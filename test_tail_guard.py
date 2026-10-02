import unittest
from temporal_events import protect_active_tails
from test_temporal_events import samples

class TailGuardTests(unittest.TestCase):
    def run_guard(self,rows,**kw):
        r=dict(status='keep',release=4.,start=2.,end=9.,duration=7.,planned_export_end=10.)
        protect_active_tails([r],rows,30,kw.pop('window_end',20),**kw)
        return r
    def activity(self,until=10.6):
        rows=samples(length=16)
        for q in rows:
            if 9.5<=q['t']<=until:
                q.update(arm_speed=.4,foot_speed=.5,body_speed=.3,spread=.2)
                q['keypoints'][9]=q['keypoints'][10]=[.5,.4,1.]
        return rows
    def test_continued_ready_pose_extends_until_relaxation(self):
        r=self.run_guard(self.activity())
        self.assertGreater(r['planned_export_end'],10.5)
        self.assertLess(r['planned_export_end'],11.5)
        self.assertTrue(r['tail_guard_rest_confirmed'])
        self.assertEqual((r['end'],r['duration']),(9.,7.))
    def test_rest_walking_and_one_pose_spike_do_not_extend(self):
        for rows in (samples(length=16),self.activity(until=9.5)):
            self.assertEqual(self.run_guard(rows)['planned_export_end'],10.)
    def test_unresolved_play_is_bounded_and_flagged(self):
        r=self.run_guard(self.activity(until=15))
        self.assertEqual(r['planned_export_end'],11.5)
        self.assertTrue(r['tail_guard_unresolved'])
    def test_scene_window_video_limits(self):
        for kw in ({'scene_cuts':[10.4]},{'window_end':10.4},{'video_duration':10.4}):
            r=self.run_guard(self.activity(until=15),**kw)
            self.assertLessEqual(r['planned_export_end'],10.4)
    def test_pickup_and_next_serve_limit_extension(self):
        for extra in ({'retrieval_start':10.4},{}):
            r=dict(status='keep',release=4.,start=2.,end=9.,duration=7.,planned_export_end=10.,**extra)
            later=[] if extra else [dict(status='incomplete',release=10.4)]
            protect_active_tails([r]+later,self.activity(until=15),30,20)
            self.assertLessEqual(r['planned_export_end'],10.4)
    def test_short_candidates_never_promoted(self):
        records=[dict(status='too_short',end=9.),dict(status='incomplete')]
        protect_active_tails(records,self.activity(),30,20)
        self.assertTrue(all('planned_export_end' not in r for r in records))
    def test_disabled_guard(self):
        self.assertEqual(self.run_guard(self.activity(),max_extension=0)['planned_export_end'],10.)
if __name__=='__main__':unittest.main()
