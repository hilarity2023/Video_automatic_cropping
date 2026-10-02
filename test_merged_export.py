import subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from auto_pipeline import ROOT,export_merged,ffmpeg_path

class MergedExportTests(unittest.TestCase):
    def test_order_duration_audio_and_no_leftover_parts(self):
        for with_audio in (False,True):
            with self.subTest(audio=with_audio),tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as tmp:
                folder=Path(tmp);silent=folder/'silent.mp4'
                writer=cv2.VideoWriter(str(silent),cv2.VideoWriter_fourcc(*'mp4v'),25,(160,90))
                self.assertTrue(writer.isOpened())
                for i in range(150):
                    frame=np.zeros((90,160,3),np.uint8)
                    frame[:,:,2 if i<75 else 0]=240
                    writer.write(frame)
                writer.release();source=silent
                if with_audio:
                    source=folder/'source.mp4'
                    subprocess.run([ffmpeg_path(),'-v','error','-nostdin','-i',str(silent),
                        '-f','lavfi','-i','sine=frequency=1000:duration=6',
                        '-c:v','copy','-c:a','aac','-shortest',str(source)],check=True,capture_output=True)
                target=folder/'highlights.mp4'
                export_merged(source,target,[(4.,5.),(1.,2.)])
                cap=cv2.VideoCapture(str(target));self.assertTrue(cap.isOpened())
                duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS)
                self.assertAlmostEqual(duration,2.,delta=.15)
                for t,channel in [(.3,2),(1.6,0)]:
                    cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=cap.read()
                    self.assertTrue(ok)
                    self.assertGreater(float(frame[:,:,channel].mean()),200)
                cap.release()
                subprocess.run([ffmpeg_path(),'-v','error','-nostdin','-i',str(target),
                    '-f','null','-'],check=True,capture_output=True)
                if with_audio:
                    subprocess.run([ffmpeg_path(),'-v','error','-nostdin','-i',str(target),
                        '-map','0:a:0','-f','null','-'],check=True,capture_output=True)
                self.assertFalse(list(folder.glob('.merge_*')))
                self.assertFalse(list(folder.glob('*.partial.mp4')))
                self.assertFalse(list(folder.glob('part_*.mp4')))

    def test_failure_cleans_scratch_and_preserves_existing_target(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as tmp:
            folder=Path(tmp);target=folder/'highlights.mp4';target.write_bytes(b'previous')
            with patch('auto_pipeline.export',side_effect=RuntimeError('encoder failure')):
                with self.assertRaises(RuntimeError):export_merged(folder/'source.mp4',target,[(1,2)])
            self.assertEqual(target.read_bytes(),b'previous')
            self.assertEqual(list(folder.iterdir()),[target])

    def test_no_empty_output(self):
        with self.assertRaises(ValueError):export_merged(Path('source'),Path('out'),[])

if __name__=='__main__':unittest.main()
