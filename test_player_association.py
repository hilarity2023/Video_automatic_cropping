import unittest
import numpy as np
from player_features import PlayerFeatures
class AssociationTests(unittest.TestCase):
    def tracker(self):
        p=PlayerFeatures(None);p.appearance=np.array([1.,0.]);p.last_box=[.4,.3,.6,.7];p.last_seen=0
        return p
    def test_nearby_changed_appearance_can_be_retained(self):
        p=self.tracker()
        self.assertIsNotNone(p.association(np.array([.43,.4,.58,.7]),np.array([.12,.9928]),.17,np.zeros(2)))
    def test_identical_clothes_do_not_allow_teleport(self):
        p=self.tracker()
        self.assertIsNone(p.association(np.array([.8,.3,1.,.7]),np.array([1.,0.]),.17,np.zeros(2)))
    def test_small_nearby_person_with_wrong_scale_rejected(self):
        p=self.tracker()
        self.assertIsNone(p.association(np.array([.48,.65,.52,.7]),np.array([1.,0.]),.17,np.zeros(2)))
    def test_long_loss_requires_appearance_evidence(self):
        p=self.tracker()
        self.assertIsNone(p.association(np.array([.4,.3,.6,.7]),np.array([.12,.9928]),2,np.zeros(2)))
class ArrayValue:
    def __init__(self,value):self.value=np.asarray(value,dtype=float)
    def cpu(self):return self
    def numpy(self):return self.value

class RecoveryTests(unittest.TestCase):
    def result(self,box=None,offset=(0,0)):
        from types import SimpleNamespace
        boxes=[] if box is None else [box]
        pts=np.zeros((0,17,3)) if box is None else np.ones((1,17,3))
        if box is not None:
            pts[0,:,:2]=[50-offset[0],50-offset[1]]
        return SimpleNamespace(boxes=SimpleNamespace(xyxy=ArrayValue(boxes)),keypoints=SimpleNamespace(data=ArrayValue(pts)))
    def tracker(self,results):
        from unittest.mock import Mock
        p=PlayerFeatures(Mock());p.pose.predict.side_effect=[[r] for r in results]
        p.camera_shift=lambda f:np.zeros(2)
        p.appearance=np.zeros(144);p.appearance[0]=1
        p.last_box=[.4,.3,.6,.7];p.last_seen=0.
        return p
    def test_crop_coordinates_return_to_full_image(self):
        p=self.tracker([self.result(),self.result([15,30,35,70],(25,0))])
        q=p.infer(np.zeros((100,100,3),np.uint8),.17)
        self.assertTrue(q['valid']);self.assertEqual(q['tracking_source'],'recovery_crop')
        np.testing.assert_allclose(q['box'],[.4,.3,.6,.7])
        np.testing.assert_allclose(q['keypoints'][9][:2],[.5,.5])
        self.assertEqual(p.pose.predict.call_count,2)
    def test_bending_recovery_burst_is_bounded(self):
        p=self.tracker([self.result(),self.result([15,30,35,70],(25,0))])
        q=p.infer(np.zeros((100,100,3),np.uint8),.17)
        self.assertTrue(q['bent'])
        self.assertAlmostEqual(p.next_recovery,.17)
        self.assertAlmostEqual(p.recovery_burst_until,.52)
        self.assertAlmostEqual(p.recovery_burst_cooldown,1.17)
        p.pose.predict.side_effect=[[self.result()],[self.result([15,30,35,70],(25,0))]]
        p.infer(np.zeros((100,100,3),np.uint8),.6)
        self.assertAlmostEqual(p.next_recovery,.85)
        self.assertAlmostEqual(p.recovery_burst_until,.52)

    def test_no_second_inference_when_full_frame_succeeds(self):
        p=self.tracker([self.result([40,30,60,70])])
        self.assertTrue(p.infer(np.zeros((100,100,3),np.uint8),.17)['valid'])
        self.assertEqual(p.pose.predict.call_count,1)
    def test_crop_background_person_below_size_floor_rejected(self):
        p=self.tracker([self.result(),self.result([15,40,20,50],(25,0))])
        self.assertFalse(p.infer(np.zeros((100,100,3),np.uint8),.17)['valid'])

    def test_no_stale_crop_after_long_absence(self):
        p=self.tracker([self.result()])
        self.assertFalse(p.infer(np.zeros((100,100,3),np.uint8),2.1)['valid'])
        self.assertEqual(p.pose.predict.call_count,1)

if __name__=='__main__':unittest.main()
