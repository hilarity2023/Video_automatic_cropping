import unittest
import numpy as np
from types import SimpleNamespace
from unittest.mock import Mock
import test_player_association as fixtures
from manual_setup import validate_annotation
from player_features import MultiPlayerFeatures,combine_players
from temporal_events import segment_actions
from test_temporal_events import samples

class MultiPlayerTests(unittest.TestCase):
    def annotation(self,n=2):
        return dict(court=[[0,0],[1,0],[1,1],[0,1]],player_boxes=[[.05+i*.24,.1,.20+i*.24,.7] for i in range(n)],shuttle=None)
    def test_annotation_accepts_two_to_four_rejects_one_five_and_duplicates(self):
        for n in (2,3,4):validate_annotation(self.annotation(n))
        for n in (1,5):
            with self.assertRaises(ValueError):validate_annotation(self.annotation(n))
        a=self.annotation();a['player_boxes'][1]=a['player_boxes'][0]
        with self.assertRaises(ValueError):validate_annotation(a)
    def test_four_trackers_share_one_full_frame_inference(self):
        a=self.annotation(4);model=Mock();frame=np.zeros((100,100,3),np.uint8)
        boxes=np.asarray(a['player_boxes'])*100;pts=np.ones((4,17,3));pts[:,:,:2]=50
        result=SimpleNamespace(boxes=SimpleNamespace(xyxy=fixtures.ArrayValue(boxes)),keypoints=SimpleNamespace(data=fixtures.ArrayValue(pts)))
        model.predict.return_value=[result]
        p=MultiPlayerFeatures(model);p.initialize_manual(frame,a,0.)
        for tracker in p.trackers:tracker.camera_shift=lambda f:np.zeros(2)
        row=p.infer(frame,.1)
        self.assertEqual(model.predict.call_count,1)
        self.assertEqual([q['player_id'] for q in row['players']],[1,2,3,4])
        self.assertTrue(all(q['valid'] for q in row['players']))
        self.assertEqual(len({tuple(q['box']) for q in row['players']}),4)

    def test_automatic_setup_tracks_two_detected_players(self):
        model=Mock();frame=np.full((100,100,3),127,np.uint8)
        boxes=np.asarray([[10,20,35,85],[55,24,80,82]],float)
        pts=np.ones((2,17,3));pts[:,:,:2]=[20,45]
        result=SimpleNamespace(boxes=SimpleNamespace(xyxy=fixtures.ArrayValue(boxes)),
                               keypoints=SimpleNamespace(data=fixtures.ArrayValue(pts)))
        model.predict.return_value=[result]
        player=MultiPlayerFeatures(model,max_players=2)
        row=player.infer(frame,0.)
        self.assertEqual(len(player.trackers),2)
        self.assertEqual(len(row['players']),2)
        self.assertTrue(all(q['valid'] for q in row['players']))
        self.assertTrue(row['context_complete'])

    def test_automatic_setup_enrolls_player_who_enters_after_start(self):
        model=Mock();frame=np.full((100,100,3),127,np.uint8)
        def result(boxes):
            boxes=np.asarray(boxes,float);pts=np.ones((len(boxes),17,3));pts[:,:,:2]=[20,45]
            return SimpleNamespace(boxes=SimpleNamespace(xyxy=fixtures.ArrayValue(boxes)),
                                   keypoints=SimpleNamespace(data=fixtures.ArrayValue(pts)))
        one=result([[10,20,35,85]])
        two=result([[10,20,35,85],[55,24,80,82]])
        model.predict.side_effect=[[one],[two],[two],[two]]
        player=MultiPlayerFeatures(model,max_players=2)
        player.infer(frame,0.)
        player.infer(frame,.1)
        player.infer(frame,.2)
        row=player.infer(frame,.3)
        self.assertEqual(len(player.trackers),2)
        self.assertEqual(len(row['players']),2)
        self.assertTrue(row['context_complete'])
    def test_other_players_action_extends_rally(self):
        a=samples([(5,10)],length=25);b=samples([(12,18)],length=25)
        rows=[combine_players([x,y],x['t']) for x,y in zip(a,b)]
        records=segment_actions(rows)
        self.assertEqual(len(records),1);self.assertGreater(records[0]['last_action'],17.5)
    def test_bending_player_does_not_hide_active_teammate(self):
        a=samples([(5,10)],pickup=True);b=samples([(5,10)])
        row=combine_players([a[60],b[60]],6.)
        self.assertFalse(row['bent']);self.assertFalse(row['group_rest'])
    def test_retrieval_not_hidden_by_other_players_wrist_motion(self):
        a=samples([(5,10),(14,20)],length=27);b=samples([(5,20)],length=27)
        for q in a:
            if 11<=q['t']<=12:q.update(bent=True,body_speed=.1,foot_speed=.2,arm_speed=.3)
        rows=[combine_players([x,y],x['t']) for x,y in zip(a,b)]
        records=segment_actions(rows)
        self.assertEqual(len(records),2)
        self.assertLess(records[0]['last_action'],11)
        self.assertGreater(records[1]['first_action'],12)
        self.assertAlmostEqual(records[0]['retrieval_start'],11)

    def test_shallow_bend_low_hand_splits_without_deep_bent_flag(self):
        a=samples([(5,20)],length=27);b=samples([(5,20)],length=27)
        for q in a:
            if 11<=q['t']<=12:
                q.update(torso=.19,bent=False,body_speed=.1,foot_speed=.2,arm_speed=.3)
                q['keypoints'][9]=[.5,.75,1.]
        records=segment_actions([combine_players([x,y],x['t']) for x,y in zip(a,b)],pre_roll=3)
        self.assertEqual(len(records),2)
        self.assertLess(records[0]['last_action'],11)
        self.assertGreaterEqual(records[1]['start'],13)

    def test_missing_player_is_not_end_confirmation(self):
        a=samples([(5,10)],length=18);b=samples(length=18)
        for q in b:
            if q['t']>10:q['valid']=False
        rows=[combine_players([x,y],x['t']) for x,y in zip(a,b)]
        self.assertEqual(segment_actions(rows)[0]['status'],'incomplete')

if __name__=='__main__':unittest.main()
