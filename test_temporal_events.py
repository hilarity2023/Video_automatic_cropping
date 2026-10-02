import unittest
from temporal_events import segment_actions,TemporalSettings,merge_action_proposals


def samples(spans=(),length=30,pickup=False):
    rows=[]
    for i in range(length*10):
        t=i/10; active=any(a<=t<=b for a,b in spans)
        pts=[[.5,.5,1.] for _ in range(17)]
        pts[11]=pts[12]=[.5,.6,1.]
        pts[9]=pts[10]=[.5,.4 if active else .6,1.]
        rows.append(dict(t=t,valid=True,box=[.3,.1,.7,.9],keypoints=pts,
                         arm_speed=1.2 if active else .1,foot_speed=.4 if active else .1,
                         spread=.12,bent=active and pickup))
    return rows


class TemporalTests(unittest.TestCase):
    def test_short_action_groups_are_stitched_before_duration_filtering(self):
        proposals=[dict(release=10.,start=9.,first_action=10.3,last_action=11.,
                        evidence_frames=5,retrieval_start=30.),
                   dict(release=13.2,start=12.2,first_action=13.5,last_action=14.5,
                        evidence_frames=5,retrieval_start=30.),
                   dict(release=15.7,start=14.7,first_action=16.,last_action=16.6,
                        evidence_frames=4,retrieval_start=30.)]
        merged=merge_action_proposals(proposals,[])
        self.assertEqual(len(merged),1)
        self.assertEqual((merged[0]['release'],merged[0]['start']),(10.,9.))
        self.assertEqual((merged[0]['first_action'],merged[0]['last_action']),(10.3,16.6))
        self.assertEqual(merged[0]['retrieval_start'],30.)
        self.assertEqual(merged[0]['evidence_frames'],14)

    def test_retrieval_or_long_ready_pause_blocks_stitching(self):
        base=dict(release=10.,start=9.,first_action=10.3,last_action=11.)
        following=dict(release=14.,start=13.,first_action=14.3,last_action=15.)
        self.assertEqual(len(merge_action_proposals([dict(base,retrieval_start=12.),following],[])),2)
        rows=[]
        for i in range(10):
            rows.append(dict(t=11.1+i*.1,valid=True,context_complete=True,group_rest=True))
        self.assertEqual(len(merge_action_proposals([base,following],rows)),2)

    def test_long_gap_stitches_only_with_intermediate_play_evidence(self):
        first=dict(release=10.,start=9.,first_action=10.3,last_action=11.)
        following=dict(release=18.7,start=17.7,first_action=18.9,last_action=20.)
        self.assertEqual(len(merge_action_proposals([first,following],[])),2)
        rows=[dict(t=14.+i*.1,valid=True,players=[dict(valid=True,overhead=True,arm_speed=1.)])
               for i in range(3)]
        merged=merge_action_proposals([first,following],rows)
        self.assertEqual(len(merged),1)
        self.assertEqual(merged[0]['last_action'],20.)
        walking=[dict(t=14.+i*.1,valid=True,players=[dict(valid=True,overhead=True,arm_speed=1.,
                    body_speed=.5,foot_speed=.6,spread=.1)]) for i in range(4)]
        self.assertEqual(len(merge_action_proposals([first,following],walking)),2)

    def test_short_low_serve_swing_seed_connects_to_following_rally(self):
        first=dict(release=10.5,start=8.,first_action=10.8,last_action=11.2)
        following=dict(release=15.,start=14.,first_action=15.8,last_action=20.)
        serve=[dict(t=t,valid=True,ball_held=False,players=[dict(valid=True,player_id=1,
            overhead=False,arm_speed=3.4,foot_speed=.3,body_speed=.12,spread=.1)])
            for t in (10.9,11.0)]
        merged=merge_action_proposals([first,following],serve)
        self.assertEqual(len(merged),1)
        self.assertAlmostEqual(merged[0]['start'],10.0)
        self.assertAlmostEqual(merged[0]['release'],10.5)
        self.assertEqual(merged[0]['first_action'],10.8)
        self.assertEqual(merged[0]['last_action'],20.)

    def test_fast_arm_while_walking_does_not_bridge_separate_proposals(self):
        first=dict(release=10.,start=9.,first_action=10.3,last_action=11.)
        following=dict(release=15.,start=14.,first_action=15.3,last_action=18.)
        walking=[dict(t=t,valid=True,ball_held=False,players=[dict(valid=True,player_id=1,
            overhead=False,arm_speed=3.4,foot_speed=.8,body_speed=.35,spread=.1)])
            for t in (10.4,10.5)]
        self.assertEqual(len(merge_action_proposals([first,following],walking)),2)

    def test_low_serve_swing_does_not_extend_a_proposal_tail(self):
        proposal=dict(release=0.,start=0.,first_action=1.,last_action=5.)
        rows=[dict(t=t,valid=True,ball_held=False,players=[dict(valid=True,player_id=1,
            overhead=False,arm_speed=1.8,foot_speed=.2,body_speed=.1,spread=.1)])
            for t in (5.1,5.2)]
        merged=merge_action_proposals([proposal],rows)
        self.assertEqual(merged[0]['last_action'],5.)

    def test_recent_active_play_extends_tail_but_retrieval_stays_a_boundary(self):
        first=dict(release=10.,start=9.,first_action=10.3,last_action=11.,retrieval_start=14.)
        following=dict(release=14.7,start=13.7,first_action=15.,last_action=16.)
        rows=[dict(t=11.2+i*.1,valid=True,ball_held=False,
                   players=[dict(valid=True,overhead=True,arm_speed=1.,bent=False)]) for i in range(3)]
        merged=merge_action_proposals([first,following],rows)
        self.assertEqual(len(merged),2)
        self.assertAlmostEqual(merged[0]['last_action'],11.4)

    def test_short_late_proposal_recovers_preceding_multi_player_strokes(self):
        proposal=dict(release=382.06,start=379.56,first_action=382.06,last_action=382.46,
                      pre_roll_available=2.5,evidence_frames=4)
        rows=[
            dict(t=377.8,valid=True,players=[dict(valid=True,player_id=1,overhead=True,
                arm_speed=1.1,foot_speed=.5,body_speed=.5,spread=.1)]),
            dict(t=378.8,valid=True,players=[dict(valid=True,player_id=2,overhead=False,
                arm_speed=1.2,foot_speed=1.5,body_speed=.5,spread=.4)]),
            dict(t=380.2,valid=True,players=[dict(valid=True,player_id=2,overhead=False,
                arm_speed=1.2,foot_speed=1.5,body_speed=.5,spread=.4)]),
            dict(t=381.9,valid=True,players=[dict(valid=True,player_id=1,overhead=True,
                arm_speed=1.1,foot_speed=.1,body_speed=.1,spread=.1)]),
        ]
        merged=merge_action_proposals([proposal],rows)
        self.assertEqual(len(merged),1)
        self.assertAlmostEqual(merged[0]['first_action'],377.8)
        self.assertAlmostEqual(merged[0]['start'],374.5)

    def test_walk_like_overhead_alone_cannot_backtrack_short_proposal(self):
        proposal=dict(release=382.06,start=379.56,first_action=382.06,last_action=382.46,
                      pre_roll_available=2.5,evidence_frames=4)
        rows=[dict(t=t,valid=True,players=[dict(valid=True,player_id=1,overhead=True,
            arm_speed=1.1,foot_speed=.5,body_speed=.5,spread=.1)]) for t in (377.8,378.8,380.2,381.9)]
        merged=merge_action_proposals([proposal],rows)
        self.assertEqual(merged[0]['first_action'],382.06)
        self.assertEqual(merged[0]['start'],379.56)

    def test_two_second_proposal_can_recover_a_multi_player_stroke_cluster(self):
        proposal=dict(release=12.,start=9.5,first_action=12.,last_action=14.2,
                      pre_roll_available=2.5,evidence_frames=10)
        rows=[
            dict(t=7.8,valid=True,players=[dict(valid=True,player_id=1,overhead=True,
                arm_speed=1.1,foot_speed=.5,body_speed=.5,spread=.1)]),
            dict(t=9.,valid=True,players=[dict(valid=True,player_id=2,overhead=False,
                arm_speed=1.2,foot_speed=1.5,body_speed=.5,spread=.4)]),
            dict(t=10.8,valid=True,players=[dict(valid=True,player_id=2,overhead=False,
                arm_speed=1.2,foot_speed=1.5,body_speed=.5,spread=.4)]),
            dict(t=11.9,valid=True,players=[dict(valid=True,player_id=1,overhead=True,
                arm_speed=1.1,foot_speed=.1,body_speed=.1,spread=.1)]),
        ]
        merged=merge_action_proposals([proposal],rows)
        self.assertAlmostEqual(merged[0]['first_action'],7.8)
        self.assertAlmostEqual(merged[0]['start'],4.5)

    def test_stale_walking_tail_is_cut_after_last_clear_stroke(self):
        proposal=dict(release=349.,start=349.,first_action=349.2,last_action=360.,evidence_frames=50)
        rows=[dict(t=350.,valid=True,players=[dict(valid=True,player_id=1,overhead=True,
                   arm_speed=1.2,foot_speed=.1,body_speed=.1,spread=.1)])]
        rows.extend(dict(t=t,valid=True,players=[dict(valid=True,player_id=1,overhead=True,
                   arm_speed=1.2,foot_speed=.6,body_speed=.5,spread=.1)])
                     for t in (352.,353.,354.))
        merged=merge_action_proposals([proposal],rows)
        self.assertAlmostEqual(merged[0]['last_action'],350.)

    def test_long_absence_return_walk_does_not_extend_rally(self):
        rows=samples([(5,11),(14,18)])
        for q in rows:
            if 10.8<=q['t']<=11.1:q['box']=[0.,.1,.2,.9]
            if 11.1<=q['t']<14:q.update(valid=False,arm_speed=0.,foot_speed=0.)
            if 14<=q['t']<=18:q.update(foot_speed=.6,body_speed=.4)
        r=segment_actions(rows)
        self.assertEqual(len(r),1)
        self.assertEqual(r[0]['status'],'keep')
        self.assertLess(r[0]['end'],13)
        self.assertFalse(r[0]['landing_observed'])

    def test_clear_athletic_return_after_absence_can_start_new_play(self):
        rows=samples([(5,11),(16,24)],length=30)
        for q in rows:
            if 10.8<=q['t']<=11.1:q['box']=[0.,.1,.2,.9]
            if 11.1<=q['t']<16:q.update(valid=False,arm_speed=0.,foot_speed=0.)
            if 16<=q['t']<=24:q.update(overhead=True,foot_speed=1.3,body_speed=.6)
        r=segment_actions(rows)
        self.assertEqual(len(r),2)
        self.assertGreater(r[-1]['last_action'],23)

    def test_rally_survives_gaps_between_actions(self):
        r=segment_actions(samples([(5,6),(8,9),(11,12)]))
        self.assertEqual(len(r),1)
        self.assertEqual(r[0]['status'],'keep')
        self.assertFalse(r[0]['landing_observed'])
        self.assertTrue(r[0]['needs_review'])

    def test_separate_rallies_and_filter_short_one(self):
        r=segment_actions(samples([(5,11),(20,21)]))
        self.assertEqual([q['status'] for q in r],['keep','too_short'])

    def test_rest_and_pickup_do_not_make_rallies(self):
        self.assertEqual(segment_actions(samples()),[])
        self.assertEqual(segment_actions(samples([(5,11)],pickup=True)),[])

    def test_fast_lunge_is_not_pickup_or_end(self):
        rows=samples([(5,6),(8,9),(11,12)])
        for q in rows:
            if 8<=q['t']<=9:q.update(bent=True,foot_speed=1.5,spread=.4)
        r=segment_actions(rows)
        self.assertEqual(len(r),1)
        self.assertGreater(r[0]['last_action'],11)

    def test_truncated_action_is_incomplete(self):
        r=segment_actions(samples([(25,29)]))
        self.assertEqual(r[0]['status'],'incomplete')
        self.assertNotIn('end',r[0])

    def test_missing_player_does_not_confirm_end(self):
        rows=samples([(5,11)])
        for q in rows:
            if q['t']>11:q['valid']=False
        self.assertEqual(segment_actions(rows)[0]['status'],'incomplete')

    def test_pre_roll_clamped_and_duration_not_inflated(self):
        r=segment_actions(samples([(0,4)]))[0]
        self.assertEqual(r['start'],0)
        self.assertEqual(r['status'],'too_short')
        self.assertEqual(r['duration'],r['end']-r['start'])

    def test_exact_threshold_and_just_under(self):
        data=samples([(5,11)])
        duration=segment_actions(data)[0]['duration']
        self.assertEqual(segment_actions(data,t_limit=duration)[0]['status'],'keep')
        self.assertEqual(segment_actions(data,t_limit=duration+.01)[0]['status'],'too_short')

    def test_broadcast_uses_sustained_preparation_before_weak_play(self):
        rows=samples([(9,11)],length=16)
        for q in rows:
            if 5<=q['t']<9:q.update(arm_speed=.2,foot_speed=.5,body_speed=.3)
        r=segment_actions(rows,broadcast=True)[0]
        self.assertTrue(r['preparation_observed'])
        self.assertLess(r['release'],5)

    def test_relaxation_and_non_play_transition_can_confirm_end(self):
        rows=samples([(5,11)],length=13)
        old=segment_actions(rows)[0]
        self.assertEqual(old['status'],'incomplete')
        r=segment_actions(rows,broadcast=True,terminal_transition=True)[0]
        self.assertEqual(r['status'],'keep')
        self.assertEqual(r['end_confirmation'],'relaxed_player_then_non_play_view')
        self.assertFalse(r['landing_observed'])

    def test_transition_alone_or_missing_player_cannot_confirm_end(self):
        for missing in (False,True):
            rows=samples([(5,12.9)],length=13)
            if missing:
                for q in rows:
                    if q['t']>11:q['valid']=False
            r=segment_actions(rows,broadcast=True,terminal_transition=True)[0]
            self.assertEqual(r['status'],'incomplete')

    def test_broadcast_pre_roll_can_include_prior_camera_view(self):
        rows=samples([(5,11)])
        rows=[q for q in rows if q['t']>=5]
        r=segment_actions(rows,scene_start=5,broadcast=True,pre_roll_floor=2)[0]
        self.assertEqual(r['start'],2)
        self.assertFalse(r['preparation_observed'])

    def test_pickup_and_standing_recovery_do_not_create_rally(self):
        rows=samples([(8,8.8)],length=16)
        for q in rows:
            if 7<=q['t']<8:
                q.update(arm_speed=.5,foot_speed=.2,body_speed=.1)
                q['keypoints'][9]=q['keypoints'][10]=[.5,.85,1.]
        self.assertEqual(segment_actions(rows),[])

    def test_retrieval_breaks_group_and_caps_tail(self):
        rows=samples([(5,10),(11.5,12.2)],length=18)
        for q in rows:
            if 10.5<=q['t']<11.5:
                q.update(arm_speed=.3,foot_speed=.2,body_speed=.1,bent=True)
                q['keypoints'][9]=q['keypoints'][10]=[.5,.85,1.]
        r=segment_actions(rows)
        self.assertEqual(len(r),1)
        self.assertLessEqual(r[0]['end'],r[0]['retrieval_start'])
        self.assertLess(r[0]['last_action'],10.5)
        self.assertEqual(r[0]['end_confirmation'],'sustained_retrieval_pose')

    def test_low_hand_walking_does_not_start_rally(self):
        rows=samples(length=20)
        for q in rows:
            if 5<=q['t']<=12:
                q.update(arm_speed=.75,foot_speed=.5,body_speed=.2)
                q['keypoints'][9]=q['keypoints'][10]=[.5,.5,1.]
        self.assertEqual(segment_actions(rows),[])

    def test_fast_footwork_continues_but_cannot_start_rally(self):
        for seeded in (False,True):
            rows=samples([(5,6)] if seeded else [],length=16)
            for q in rows:
                if 6.5<=q['t']<=9:
                    q.update(arm_speed=.4,foot_speed=1.3,body_speed=.7)
                    q['keypoints'][9]=q['keypoints'][10]=[.5,.48,1.]
            records=segment_actions(rows)
            if seeded:self.assertGreaterEqual(records[0]['last_action'],8.9)
            else:self.assertEqual(records,[])

    def test_short_observation_loss_bridges_one_rally(self):
        rows=samples([(5,6),(10,11)],length=17)
        self.assertEqual(len(segment_actions(rows)),2)
        for q in rows:
            if 6.3<=q['t']<=9.8:q['valid']=False
        self.assertEqual(len(segment_actions(rows)),1)

    def test_long_observation_loss_is_not_unbounded_bridge(self):
        rows=samples([(5,6),(13,14)],length=20)
        for q in rows:
            if 6.3<=q['t']<=12.8:q['valid']=False
        self.assertEqual(len(segment_actions(rows)),2)

    def test_detected_box_without_wrists_is_missing_action_evidence(self):
        rows=samples([(5,6),(10,11)],length=17)
        for q in rows:
            if 6.3<=q['t']<=9.8:
                q['keypoints'][9][2]=q['keypoints'][10][2]=.1
        self.assertEqual(len(segment_actions(rows)),1)

    def test_fixed_start_does_not_backtrack_to_old_quiet_teammate(self):
        rows=samples([(10,18)],length=25)
        for q in rows:
            if 8<=q['t']<10:q.update(arm_speed=.5,foot_speed=.5,body_speed=.3)
        r=segment_actions(rows,pre_roll=1)[0]
        self.assertAlmostEqual(r['release'],r['first_action']-.3)
        self.assertAlmostEqual(r['start'],r['release']-1)
        self.assertGreater(r['start'],8)

    def test_carrying_hand_walk_cannot_seed_but_real_swing_can(self):
        rows=samples([(5,8),(9,16)],length=23)
        for q in rows:
            if 5<=q['t']<=8:q.update(arm_speed=.9,foot_speed=.6,body_speed=.4)
        r=segment_actions(rows,pre_roll=1)
        self.assertEqual(len(r),1)
        self.assertGreaterEqual(r[0]['first_action'],8.8)
        self.assertGreater(r[0]['last_action'],15)

    def test_walk_back_before_serve_starts_at_serve(self):
        rows=samples([(5,10),(15,19)],length=24)
        for q in rows:
            if 15<=q['t']<17:
                q.update(arm_speed=1.4,foot_speed=.7,body_speed=.45,spread=.05)
            elif 17<=q['t']<=19:
                q.update(arm_speed=1.4,foot_speed=.2,body_speed=.1,spread=.05)
        records=segment_actions(rows,pre_roll=1)
        self.assertEqual(len(records),2)
        self.assertGreaterEqual(records[1]['first_action'],17)
        self.assertGreaterEqual(records[1]['start'],15.7)

    def test_leaning_near_ground_retrieval_ends_previous_rally(self):
        rows=samples([(5,9),(13,18)],length=23)
        for q in rows:
            if 10<=q['t']<=10.6:
                q.update(arm_speed=.2,foot_speed=.2,body_speed=.1,spread=.05)
                for k in (5,6):q['keypoints'][k]=[.42,.42,1.]
                for k in (9,10):q['keypoints'][k]=[.5,.64,1.]
        records=segment_actions(rows)
        self.assertEqual(len(records),2)
        self.assertLessEqual(records[0]['last_action'],9)
        self.assertLessEqual(records[0]['end'],records[0]['retrieval_start'])
        self.assertGreater(records[1]['first_action'],12)
        self.assertGreaterEqual(records[1]['start'],records[1]['release'])

    def test_walking_cannot_refresh_last_hit_forever(self):
        rows=samples([(5,6)],length=30)
        for q in rows:
            if 6.5<=q['t']<=23:
                q.update(arm_speed=.4,foot_speed=1.3,body_speed=.7)
                q['keypoints'][9]=q['keypoints'][10]=[.5,.48,1.]
        r=segment_actions(rows)[0]
        self.assertLess(r['last_action'],10)

    def test_ambiguous_large_arm_walk_is_kept_for_recall(self):
        rows=samples([(5,18)],length=25)
        for q in rows:
            if 5<=q['t']<=18:q.update(arm_speed=1.8,foot_speed=.9,body_speed=.5,overhead=False)
        records=segment_actions(rows)
        self.assertEqual(len(records),1)
        self.assertEqual(records[0]['start_confidence'],'uncertain')

    def test_confirmed_ground_stop_splits_even_continuous_action_proposal(self):
        rows=samples([(5,20)],length=27)
        for q in rows:
            if q['t']>12:q.update(overhead=True,arm_speed=1.5)
        records=segment_actions(rows,pre_roll=1,stop_times=[11.])
        self.assertEqual(len(records),2)
        self.assertLess(records[0]['last_action'],11)
        self.assertEqual(records[0]['ground_stop_time'],11)
        self.assertGreaterEqual(records[1]['start'],11.5)

    def test_timestamp_validation(self):
        data=samples();data[3]['t']=data[2]['t']
        with self.assertRaises(ValueError):segment_actions(data)
        with self.assertRaises(ValueError):TemporalSettings(action_gap=-1).validate()

if __name__=='__main__':unittest.main()
