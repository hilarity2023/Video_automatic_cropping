import math
import unittest
from auto_pipeline import Ball, Events, Settings, build_parser, DEFAULT_INPUT, DEFAULT_OUTPUT

def simulate(release=4.0,flight=5.0,limit=8):
    engine=Events(Settings(pre_roll=3.,t_limit=limit))
    wrists=[(.4,.3)]
    for i in range(9):
        t=release-.4+i*.05;engine.step(t,[Ball(t,.4,.3)],wrists)
    n=round(flight/.05)
    for i in range(1,n+1):
        t=release+i*.05;x=.4+.12*math.sin(i*.5);y=.3+.5*i/n
        engine.step(t,[Ball(t,x,y)],wrists)
    for i in range(1,10):
        t=release+flight+i*.05;engine.step(t,[Ball(t,x,.8)],wrists)
    return engine

class AutomaticTests(unittest.TestCase):
    def test_cli_defaults(self):
        a=build_parser().parse_args([])
        from pathlib import Path
        self.assertEqual(a.input,Path(DEFAULT_INPUT))
        self.assertEqual(a.output,Path(DEFAULT_OUTPUT))
        from auto_pipeline import DEFAULT_END_SECONDS
        self.assertEqual(a.max_seconds,DEFAULT_END_SECONDS)
        self.assertFalse(a.dry_run)
    def test_whole_rally(self):
        r=simulate(flight=9).records[0]
        self.assertEqual(r['status'],'keep')
        self.assertAlmostEqual(r['start'],1)
        self.assertAlmostEqual(r['end'],13)
        self.assertAlmostEqual(r['qualifying_duration'],9)
        self.assertAlmostEqual(r['duration'],12)
    def test_under_eight_dropped(self):
        self.assertEqual(simulate(flight=4.8).records[0]['status'],'too_short')
    def test_start_clipped(self):
        r=simulate(release=1).records[0]
        self.assertEqual(r['start'],0)
        self.assertEqual(r['status'],'too_short')
    def test_t_limit_override(self):
        self.assertEqual(simulate(flight=4.8,limit=4.5).records[0]['status'],'keep')
    def test_ball_far_from_hands_cannot_start_rally(self):
        e=Events(Settings())
        for i in range(100):
            t=i*.05;e.step(t,[Ball(t,.4+.1*math.sin(i*.5),.3)],[(.8,.8)])
        self.assertIsNone(e.release)
        self.assertEqual(e.records,[])
    def test_one_frame_near_hand_is_insufficient(self):
        e=Events(Settings());e.step(0,[Ball(0,.4,.3)],[(.4,.3)])
        for i in range(1,20):e.step(i*.05,[Ball(i*.05,.6,.4)],[(.4,.3)])
        self.assertIsNone(e.release)
    def test_loss_is_not_landing(self):
        e=Events(Settings());e.release=1;e.last=Ball(2,.4,.8)
        e.step(4,[],[])
        self.assertEqual(e.records[0]['status'],'incomplete')
        self.assertNotIn('end',e.records[0])
    def test_analysis_limit_is_not_landing(self):
        e=Events(Settings());e.release=98;e.finish_unknown(99.9,'limit')
        self.assertEqual(e.records[0]['status'],'incomplete')
    def test_scene_cut_clears_active_rally(self):
        e=Events(Settings());e.release=1;e.reset_scene(5)
        self.assertIsNone(e.release);self.assertEqual(e.scene_start,5)
        self.assertEqual(e.records[0]['reason'],'scene_change')
    def test_invalid_parameters(self):
        for value in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):Settings(t_limit=value).validate()
    def test_no_hand_no_release(self):
        e=Events(Settings())
        for i in range(100):e.step(i*.05,[Ball(i*.05,.4,.3)],[])
        self.assertIsNone(e.release)

    def test_bending_pickup_rejected(self):
        from auto_pipeline import eligible_wrists
        k=[[50,50,1.] for _ in range(17)]
        k[5]=k[6]=[50,48,1.];k[11]=k[12]=[50,50,1.]
        k[9]=k[10]=[50,90,1.]
        self.assertEqual(eligible_wrists(k,[0,0,100,100],100,100),[])
        k[5]=k[6]=[50,20,1.];k[9]=k[10]=[50,50,1.]
        self.assertEqual(len(eligible_wrists(k,[0,0,100,100],100,100)),2)
    def test_known_static_distractor_removed(self):
        from auto_pipeline import StaticTargets
        import numpy as np
        filt=StaticTargets(Settings())
        frame=np.zeros((270,480,3),np.uint8)
        for i in range(11):result=filt.filter(frame,[Ball(i*.1,.4,.5)],[],i*.1)
        self.assertEqual(result,[])
        result=filt.filter(frame,[Ball(1.1,.4,.5)],[(.4,.5)],1.1)
        self.assertEqual(len(result),1)
    def test_real_export_preserves_duration_and_audio(self):
        import cv2,subprocess,tempfile
        from pathlib import Path
        from auto_pipeline import ROOT,ffmpeg_path,export
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as tmp:
            source=Path(tmp)/'synthetic.mp4';target=Path(tmp)/'out.mp4'
            subprocess.run([ffmpeg_path(),'-v','error','-nostdin','-f','lavfi','-i',
                'testsrc2=size=160x90:rate=30:duration=10','-f','lavfi','-i',
                'sine=frequency=1000:duration=10','-c:v','libx264','-c:a','aac','-shortest',str(source)],check=True)
            export(source,target,1,9)
            cap=cv2.VideoCapture(str(target))
            self.assertTrue(cap.isOpened())
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS),8,places=2)
            cap.release()
            subprocess.run([ffmpeg_path(),'-v','error','-nostdin','-i',str(target),
                '-map','0:a:0','-f','null','-'],check=True,capture_output=True)

if __name__=='__main__':unittest.main()
