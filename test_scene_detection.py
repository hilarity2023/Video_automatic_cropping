import unittest
import cv2,numpy as np
from scene_detection import SceneDetector,broadcast_wide_view
class SceneTests(unittest.TestCase):
    def test_camera_movement_not_scene_cut(self):
        rng=np.random.default_rng(42)
        frame=rng.integers(0,256,(90,160,3),dtype=np.uint8)
        d=SceneDetector();d.step(frame,0)
        self.assertFalse(d.step(np.roll(frame,50,axis=1),.1)[0])
    def test_true_colour_scene_change_and_debounce(self):
        d=SceneDetector()
        a=np.full((90,160,3),(0,160,0),np.uint8)
        b=np.full((90,160,3),(0,0,160),np.uint8)
        self.assertFalse(d.step(a,0)[0])
        self.assertTrue(d.step(b,1)[0])
        self.assertFalse(d.step(a,1.1)[0])
    def test_wide_court_vs_closeup(self):
        f=np.zeros((180,320,3),np.uint8)
        cv2.fillConvexPoly(f,np.array([[110,45],[210,45],[310,179],[10,179]]),(60,140,60))
        self.assertTrue(broadcast_wide_view(f))
        self.assertFalse(broadcast_wide_view(np.full_like(f,(20,20,80))))
        self.assertFalse(broadcast_wide_view(np.full_like(f,(60,140,60))))
if __name__=='__main__':unittest.main()
