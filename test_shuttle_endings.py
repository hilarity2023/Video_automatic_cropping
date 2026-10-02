import unittest,tempfile
from pathlib import Path
from auto_pipeline import Ball,Settings
from shuttle_endings import resolve_shuttle_endings,write_merged_start_log

class ShuttleEndTests(unittest.TestCase):
    def run_rows(self,points):
        c=Settings(t_limit=0,ground=((0,0),(1,0),(1,1),(0,1)))
        rows=[dict(t=t,balls=[] if x is None else [Ball(t,x,y)],wrists=[(.5,.5)]) for t,x,y in points]
        return resolve_shuttle_endings([dict(release=0,start=0)],rows,c,10)[0]
    def test_stop_requires_half_second_and_previous_motion(self):
        r=self.run_rows([(0,.5,.5),(.1,.6,.5)]+[(i/10,.6,.5) for i in range(2,8)])
        self.assertEqual(r['end_confirmation'],'shuttle_stationary');self.assertAlmostEqual(r['end'],.6)
        r=self.run_rows([(i/10,.5,.5) for i in range(10)])
        self.assertEqual(r['status'],'incomplete')
    def test_outward_exit_needs_more_than_point_eight(self):
        r=self.run_rows([(0,.5,.5),(.2,.8,.5),(.3,.98,.5)]+[(i/10,None,None) for i in range(4,13)])
        self.assertEqual(r['end_confirmation'],'outward_exit_missing');self.assertGreater(r['end']-.3,.8)
    def test_interior_disappearance_is_not_exit(self):
        r=self.run_rows([(0,.5,.5),(.1,.6,.5)]+[(i/10,None,None) for i in range(2,25)])
        self.assertEqual(r['status'],'incomplete')
    def test_brief_occlusion_does_not_count_as_stationary(self):
        r=self.run_rows([(0,.5,.5),(.1,.6,.5),(.2,.6,.5),(.3,None,None),(.4,.6,.5),(.5,.6,.5),(.6,.6,.5)])
        self.assertEqual(r['status'],'incomplete')
    def test_log_contains_only_exported_source_start_times(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'merged_starts.log';write_merged_start_log(p,[dict(start=65.25,export_end=78.125,start_confidence='uncertain'),dict(start=3661.,export_end=3670.,end_uncertain=True)],[12.88,9.])
            self.assertEqual(p.read_text(),'export=65.250-78.125s ; merged=0.000-12.880s\nexport=3661.000-3670.000s ; merged=12.880-21.880s\n')

    def test_air_does_not_end_before_longer_threshold(self):
        c=Settings(t_limit=0,ground=((0,.7),(1,.7),(1,1),(0,1)))
        for points in ([(0,.5,.4),(.1,.6,.4)]+[(i/10,.6,.4) for i in range(2,10)],
                       [(0,.5,.4),(.2,.8,.4),(.3,.98,.4)]+[(i/10,None,None) for i in range(4,11)]):
            rows=[dict(t=t,balls=[] if x is None else [Ball(t,x,y)],wrists=[(.5,.4)]) for t,x,y in points]
            self.assertEqual(resolve_shuttle_endings([dict(release=0,start=0)],rows,c,10)[0]['status'],'incomplete')
    def test_air_loss_reacquires_and_ground_stop_finishes(self):
        c=Settings(t_limit=0,ground=((0,.7),(1,.7),(1,1),(0,1)))
        points=[(0,.5,.4),(.1,.6,.4)]+[(i/10,None,None) for i in range(2,22)]+[(2.2,.5,.8),(2.3,.6,.8)]+[(i/10,.6,.8) for i in range(24,31)]
        rows=[dict(t=t,balls=[] if x is None else [Ball(t,x,y)],wrists=[]) for t,x,y in points]
        r=resolve_shuttle_endings([dict(release=0,start=0)],rows,c,10)[0]
        self.assertEqual(r['status'],'keep');self.assertGreaterEqual(r['end'],2.8)

    def test_air_ends_after_its_own_thresholds(self):
        c=Settings(t_limit=0,ground=((0,.7),(1,.7),(1,1),(0,1)))
        for points,reason,minimum in [
            ([(0,.5,.4),(.1,.6,.4)]+[(i/10,.6,.4) for i in range(2,20)],'shuttle_stationary',1.6),
            ([(0,.5,.4),(.2,.8,.4),(.3,.98,.4)]+[(i/10,None,None) for i in range(4,27)],'outward_exit_missing',1.1)]:
            rows=[dict(t=t,balls=[] if x is None else [Ball(t,x,y)],wrists=[]) for t,x,y in points]
            r=resolve_shuttle_endings([dict(release=0,start=0)],rows,c,10)[0]
            self.assertEqual(r['end_confirmation'],reason)
            self.assertGreaterEqual(r['end'],minimum)
    def test_region_transition_resets_stationary_timer(self):
        c=Settings(t_limit=0,ground=((0,.7),(1,.7),(1,1),(0,1)))
        points=[(0,.5,.69),(.1,.6,.699)]+[(i/10,.6,.699) for i in range(2,10)]+[(i/10,.6,.701) for i in range(10,14)]
        rows=[dict(t=t,balls=[Ball(t,x,y)],wrists=[]) for t,x,y in points]
        self.assertEqual(resolve_shuttle_endings([dict(release=0,start=0)],rows,c,10)[0]['status'],'incomplete')
    def test_air_thresholds_must_be_longer(self):
        for values in [dict(air_shuttle_stop_seconds=.5)]:
            with self.assertRaises(ValueError):Settings(**values).validate()

    def test_exit_time_same_inside_and_outside_ground(self):
        c=Settings(t_limit=0,ground=((0,.7),(1,.7),(1,1),(0,1)))
        ends=[]
        for y in (.4,.8):
            points=[(0,.5),(.2,.8),(.3,.98)]+[(i/10,None) for i in range(4,16)]
            rows=[dict(t=t,balls=[] if x is None else [Ball(t,x,y)],wrists=[]) for t,x in points]
            r=resolve_shuttle_endings([dict(release=0,start=0)],rows,c,10)[0]
            self.assertEqual(r['end_confirmation'],'outward_exit_missing');ends.append(r['end'])
        self.assertEqual(ends[0],ends[1]);self.assertGreater(ends[0]-.3,.8)
        self.assertLessEqual(ends[0],1.2)

    def test_never_searches_later_rally_to_finish_old_one(self):
        c=Settings(t_limit=0,ground=((0,0),(1,0),(1,1),(0,1)))
        points=[(0,.5),(.1,.6)]+[(i/10,None) for i in range(2,100)]+[(10,.5),(10.1,.6)]+[(i/10,.6) for i in range(102,110)]
        rows=[dict(t=t,balls=[] if x is None else [Ball(t,x,.5)],wrists=[]) for t,x in points]
        r=resolve_shuttle_endings([dict(release=0,start=0,last_action=1),dict(release=10,start=9,last_action=10.1)],rows,c,12)
        self.assertEqual(r[0]['status'],'incomplete');self.assertLessEqual(r[0]['observed_until'],4)
        self.assertEqual(r[1]['status'],'keep');self.assertEqual(r[1]['start'],9)


    def test_retrieval_boundary_survives_resolver_and_caps_export_tail(self):
        from auto_pipeline import plan_export_tails
        c=Settings(t_limit=0)
        r=resolve_shuttle_endings([dict(release=0,start=0,last_action=10,retrieval_start=11)],[],c,20,True)
        self.assertEqual(r[0]['retrieval_start'],11)
        plan_export_tails(r,1,30,20)
        self.assertEqual(r[0]['planned_export_end'],r[0]['end'])
        self.assertGreater(r[0]['export_duration'],0)

    def test_new_ground_stop_rejects_background_duplicates_and_active_play(self):
        from shuttle_endings import ground_stop_windows
        c=Settings(ground=((0,0),(1,0),(1,1),(0,1)))
        def rows(old=False,duplicates=False):
            return [dict(t=i/4,wrists=[],balls=([Ball(i/4,.6,.8)]*(6 if duplicates else 1)
                    if old or i>=12 else [])) for i in range(20)]
        self.assertEqual(ground_stop_windows(rows(),c),[3.5])
        self.assertEqual(ground_stop_windows(rows(True),c),[])
        self.assertEqual(ground_stop_windows(rows(duplicates=True),c),[3.5])
        active=[dict(t=i/12,valid=True,arm_speed=2.,overhead=True) for i in range(60)]
        self.assertEqual(ground_stop_windows(rows(),c,active),[])
        held=rows()
        for q in held:q['wrists']=[(.6,.8)]
        self.assertEqual(ground_stop_windows(held,c),[])

    def test_ground_stop_is_not_overridden_by_later_proposal(self):
        c=Settings(t_limit=0)
        q=resolve_shuttle_endings([dict(release=1,start=0,last_action=3,ground_stop_time=4),
                                  dict(release=8,start=7,last_action=9)],[],c,15,True)
        self.assertEqual(q[0]['end'],4)
        self.assertEqual(q[0]['end_confirmation'],'new_ground_stop_with_quiet_players')
        self.assertEqual(q[1]['status'],'keep')
        self.assertTrue(q[1]['end_uncertain'])

    def test_pickup_fallback_is_optional_and_stays_in_current_rally(self):
        c=Settings(t_limit=0)
        proposals=[dict(release=0,start=0,last_action=10,retrieval_start=11)]
        self.assertEqual(resolve_shuttle_endings(proposals,[],c,20)[0]['status'],'incomplete')
        r=resolve_shuttle_endings(proposals,[],c,20,True)[0]
        self.assertEqual(r['end'],11);self.assertEqual(r['end_confirmation'],'pose_confirmed_retrieval')
        proposals[0]['retrieval_start']=18
        unresolved=resolve_shuttle_endings(proposals,[],c,20,True)[0]
        self.assertEqual(unresolved['status'],'keep')
        self.assertTrue(unresolved['end_uncertain'])

    def test_late_retrieval_allows_one_extra_second_before_safety_buffer(self):
        c=Settings(t_limit=0,shuttle_search_tail=3)
        r=resolve_shuttle_endings([dict(release=0,start=0,last_action=10,retrieval_start=14)],[],c,20,True)[0]
        self.assertAlmostEqual(r['end'],12.5)
        self.assertLessEqual(r['end'],14-1.5)
        self.assertTrue(r['end_uncertain'])

    def test_recovered_action_head_gets_one_second_to_keep_late_rally_action(self):
        c=Settings(t_limit=0,context_fallback_tail=1.5)
        r=resolve_shuttle_endings([dict(release=0,start=0,last_action=10,
            recovered_action_head=True,retrieval_start=20)],[],c,30,True)[0]
        self.assertAlmostEqual(r['end'],12.5)

    def test_retrieval_still_caps_recovered_action_tail(self):
        c=Settings(t_limit=0,context_fallback_tail=1.5)
        r=resolve_shuttle_endings([dict(release=0,start=0,last_action=10,
            recovered_action_head=True,retrieval_start=12)],[],c,30,True)[0]
        self.assertAlmostEqual(r['end'],12.)

    def test_confirmed_retrieval_near_search_limit_ends_rally(self):
        c=Settings(t_limit=0,shuttle_search_tail=3)
        proposal=dict(release=557,start=556,last_action=560,retrieval_start=562.75)
        r=resolve_shuttle_endings([proposal],[],c,580,True)[0]
        self.assertEqual(r['end'],562.75)
        self.assertEqual(r['end_confirmation'],'pose_confirmed_retrieval')

if __name__=='__main__':unittest.main()
