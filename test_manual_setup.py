import unittest
import numpy as np
from manual_setup import validate_annotation
from player_features import PlayerFeatures
import test_player_association

class ManualSetupTests(unittest.TestCase):
    def annotation(self):
        return dict(court=[[.3,.2],[.8,.2],[.8,.8],[.3,.8]],player_box=[.4,.1,.6,.4],shuttle=None)
    def test_side_camera_no_longer_rejected_by_old_bottom_limit(self):
        p=PlayerFeatures(None)
        self.assertFalse(p.allowed_player([.4,.1,.6,.4]))
        p.initialize_manual(np.zeros((100,100,3),np.uint8),self.annotation(),100.)
        self.assertTrue(p.allowed_player([.4,.1,.6,.4]))
        self.assertFalse(p.allowed_player([.05,.1,.15,.4]))
        self.assertEqual(p.last_seen,100.)
    def test_airborne_shuttle_need_not_be_inside_ground_polygon(self):
        a=self.annotation();a['shuttle']=[.5,.05]
        self.assertEqual(validate_annotation(a),a)
    def test_bad_polygon_missing_player_and_outside_feet_rejected(self):
        for changes in [dict(court=[[0,0],[1,1]]),dict(court=[[0,0],[1,1],[0,1],[1,0]]),
                        dict(player_box=None),dict(player_box=[0,0,.1,.1]),dict(shuttle=[float('nan'),0])]:
            a=self.annotation();a.update(changes)
            with self.assertRaises(ValueError):validate_annotation(a)
    def test_manual_identity_is_used_for_pose_selection(self):
        from unittest.mock import Mock
        p=PlayerFeatures(Mock());p.camera_shift=lambda f:np.zeros(2)
        frame=np.zeros((100,100,3),np.uint8)
        p.initialize_manual(frame,self.annotation(),0.)
        p.pose.predict.return_value=[test_player_association.RecoveryTests().result([40,10,60,40])]
        row=p.infer(frame,.1)
        self.assertTrue(row['valid'])
        np.testing.assert_allclose(row['box'],[.4,.1,.6,.4])

    def test_later_reference_uses_appearance_not_stale_position(self):
        p=PlayerFeatures(None);p.initialize_manual(np.zeros((100,100,3),np.uint8),self.annotation(),30.)
        moved=np.array([.65,.3,.75,.6])
        self.assertTrue(p.allowed_player(moved))
        self.assertIsNotNone(p.association(moved,p.appearance,0.,np.zeros(2)))
        wrong=np.zeros_like(p.appearance);wrong[1]=1.
        self.assertIsNone(p.association(moved,wrong,0.,np.zeros(2)))

    def test_setup_time_independent_of_analysis_range(self):
        from auto_pipeline import build_parser,DEFAULT_SETUP_SECONDS
        a=build_parser().parse_args(['--start-seconds','100','--end-seconds','200'])
        self.assertEqual((a.setup_seconds,a.start_seconds,a.max_seconds),(DEFAULT_SETUP_SECONDS,100.,200.))
        self.assertEqual(build_parser().parse_args(['--setup-seconds','45']).setup_seconds,45.)

    def test_shuttle_click_filters_only_initial_observations(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from auto_pipeline import Vision,Settings
        v=Vision.__new__(Vision);v.c=Settings();v.device='cpu';v.ball_ids=[0]
        boxes=Mock();boxes.xyxy.cpu.return_value.tolist.return_value=[[48,8,52,12],[8,8,12,12]]
        boxes.conf.cpu.return_value.tolist.return_value=[.8,.9]
        v.ball=Mock();v.ball.predict.return_value=[SimpleNamespace(boxes=boxes)]
        v.player=Mock();v.player.infer.return_value={'valid':False};v.features=[]
        v.shuttle_hint=[.5,.1];v.shuttle_hint_time=0.
        balls,_=v.infer(np.zeros((100,100,3),np.uint8),0.)
        self.assertEqual(len(balls),1);self.assertAlmostEqual(balls[0].x,.5)
        balls,_=v.infer(np.zeros((100,100,3),np.uint8),2.)
        self.assertEqual(len(balls),2);self.assertIsNone(v.shuttle_hint)

    def test_shuttle_model_rate_limit_uses_video_timestamps(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from auto_pipeline import Vision,Settings
        v=Vision.__new__(Vision);v.c=Settings(shuttle_fps=5);v.device='cpu';v.ball_ids=[0]
        boxes=Mock();boxes.xyxy.cpu.return_value.tolist.return_value=[];boxes.conf.cpu.return_value.tolist.return_value=[]
        v.ball=Mock();v.ball.predict.return_value=[SimpleNamespace(boxes=boxes)]
        v.player=Mock();v.player.infer.return_value={'valid':False};v.features=[];v.shuttle_hint=None
        for i in range(60):v.infer(np.zeros((100,100,3),np.uint8),i/60)
        self.assertEqual(v.ball.predict.call_count,5)
        self.assertEqual(v.player.infer.call_count,60)

if __name__=='__main__':unittest.main()
