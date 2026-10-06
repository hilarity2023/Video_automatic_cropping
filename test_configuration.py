import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from auto_pipeline import ROOT,Settings,build_parser,load_settings,validate_window,analyse

class ConfigurationTests(unittest.TestCase):
    def test_automatic_pipeline_does_not_export_short_rounds_around_pickups(self):
        import cv2,numpy as np
        from test_temporal_events import samples
        rows=samples([(2,5),(8,11),(14,17)],length=24)
        for q in rows:
            if 5.5<=q['t']<=6.5 or 11.5<=q['t']<=12.5:
                q.update(bent=True,arm_speed=2.,foot_speed=.2,body_speed=.1)
                q['keypoints'][9]=q['keypoints'][10]=[.5,.85,1.]
        class FakeVision:
            def __init__(self,*args):self.features=[];self.next_pose=0.
            def infer(self,frame,t):
                row=rows[min(round(t*10),len(rows)-1)]
                self.features.append(dict(row,t=t))
                return [],[]
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as tmp:
            source=Path(tmp)/'short_rounds.mp4'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(64,48))
            self.assertTrue(writer.isOpened())
            for _ in range(240):writer.write(np.full((48,64,3),127,np.uint8))
            writer.release()
            args=build_parser().parse_args([str(source),'-o',tmp,'--end-seconds','24',
                '--frame-stride','1','--view-mode','fixed','--no-manual-setup',
                '--pre-roll','5','--post-roll','10','--t-limit','8'])
            with patch('auto_pipeline.Vision',FakeVision),patch('auto_pipeline.export_merged') as exporter:
                report=analyse(args,load_settings(args))
                exporter.assert_not_called()
            self.assertEqual(report['merged_segment_count'],0)
            self.assertEqual(len(report['segments']),3)
            self.assertTrue(all(q['status']=='too_short' for q in report['segments']))
            self.assertFalse(list(Path(tmp).rglob('highlights.mp4')))

    def test_project_uses_python_clip_defaults(self):
        args=build_parser().parse_args([])
        c=load_settings(args)
        self.assertEqual((c.pre_roll,c.t_limit),(Settings().pre_roll,Settings().t_limit))

    def test_cli_overrides_json_and_python(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as tmp:
            config=Path(tmp)/'config.json'
            config.write_text(json.dumps(dict(pre_roll=3,t_limit=8)))
            args=build_parser().parse_args(['--config',str(config),'--pre-roll','1.5','--t-limit','5'])
            c=load_settings(args)
            self.assertEqual((c.pre_roll,c.t_limit),(1.5,5))

    def test_window_and_old_alias(self):
        p=build_parser()
        for flag in ['--end-seconds','--max-seconds']:
            args=p.parse_args(['--start-seconds','100',flag,'200'])
            validate_window(args)
            self.assertEqual((args.start_seconds,args.max_seconds),(100,200))
        for a,b in [(-1,100),(100,100),(200,100),(float('nan'),200),(0,float('inf'))]:
            args=p.parse_args([]);args.start_seconds=a;args.max_seconds=b
            with self.assertRaises(ValueError):validate_window(args)

    def test_tail_never_promotes_short_or_incomplete_candidates(self):
        from auto_pipeline import plan_export_tails
        records=[dict(status='keep',start=10.,end=15.,duration=5.),
                 dict(status='too_short',start=20.,end=24.,duration=4.),
                 dict(status='incomplete',start=28.)]
        plan_export_tails(records,2.,30.,100.)
        self.assertAlmostEqual(records[0]['planned_export_end'],17.+1/30)
        self.assertEqual(records[0]['end'],15.)
        self.assertEqual(records[0]['duration'],5.)
        self.assertNotIn('planned_export_end',records[1])
        self.assertNotIn('planned_export_end',records[2])

    def test_tail_clamps_to_scene_window_and_eof(self):
        from auto_pipeline import plan_export_tails
        for window,cuts,duration,expected in [(100,[16],120,16),(16,[],120,16),(100,[],16,16)]:
            r=dict(status='keep',start=10.,end=15.)
            plan_export_tails([r],2.,30.,window,cuts,duration)
            self.assertEqual(r['planned_export_end'],expected)
            self.assertEqual(r['post_roll_available'],1.)
        r=dict(status='keep',start=10.,end=15.)
        plan_export_tails([r],0.,30.,100.)
        self.assertAlmostEqual(r['planned_export_end'],15.+1/30)
        Settings(post_roll=0).validate()
        with self.assertRaises(ValueError):Settings(post_roll=-1).validate()

    def test_post_roll_cli(self):
        args=build_parser().parse_args(['--post-roll','2.5'])
        self.assertEqual(load_settings(args).post_roll,2.5)

    def test_pose_imgsz_defaults_to_higher_resolution_and_validates(self):
        self.assertEqual(Settings().pose_imgsz,768)
        with self.assertRaises(ValueError):Settings(pose_imgsz=750).validate()

    def test_five_frame_stride_at_30_and_60_fps(self):
        import cv2,numpy as np
        for fps in (30,60):
            times=[]
            class FakeVision:
                def __init__(self,*args):self.features=[];self.next_pose=0
                def infer(self,frame,t):times.append(t);return [],[]
            with self.subTest(fps=fps),tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as tmp:
                source=Path(tmp)/'stride.mp4'
                writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),fps,(64,48))
                for i in range(fps*4):writer.write(np.zeros((48,64,3),np.uint8))
                writer.release()
                args=build_parser().parse_args([str(source),'-o',tmp,'--start-seconds','1','--end-seconds','3','--frame-stride','5','--dry-run'])
                c=load_settings(args)
                with patch('auto_pipeline.Vision',FakeVision):report=analyse(args,c)
                self.assertEqual(len(times),2*fps//5)
                self.assertTrue(np.allclose(np.diff(times),5/fps))
                self.assertEqual(report['effective_analysis_fps'],fps/5)
                self.assertTrue(report['shuttle_detection_enabled'])
                self.assertEqual(list(Path(tmp).rglob('*.jsonl')),[])
                self.assertEqual(list(Path(tmp).rglob('segments.json')),[])

    def test_tail_stops_before_retrieval(self):
        from auto_pipeline import plan_export_tails
        r=dict(status='keep',start=10.,end=15.,retrieval_start=15.4)
        plan_export_tails([r],2.,30.,100.)
        self.assertEqual(r['planned_export_end'],15.4)

    def test_real_video_seek_preserves_original_timestamps(self):
        import cv2,numpy as np
        times=[]
        class FakeVision:
            def __init__(self,*args):self.features=[];self.next_pose=0
            def infer(self,frame,t):times.append(t);return [],[]
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as tmp:
            source=Path(tmp)/'seek.mp4'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),5,(64,48))
            self.assertTrue(writer.isOpened())
            for i in range(1005):writer.write(np.full((48,64,3),i%20,np.uint8))
            writer.release()
            args=build_parser().parse_args([str(source),'-o',tmp,'--start-seconds','100','--end-seconds','200','--frame-stride','5','--dry-run'])
            c=load_settings(args)
            with patch('auto_pipeline.Vision',FakeVision):report=analyse(args,c)
            self.assertEqual(report['analysis_start'],100)
            self.assertEqual(report['analysis_end'],200)
            self.assertEqual(len(times),100)
            self.assertAlmostEqual(times[0],100)
            self.assertAlmostEqual(times[-1],199.)
            self.assertTrue(all(100<=t<200 for t in times))

if __name__=='__main__':unittest.main()
