"""Scene checks that distinguish global appearance change from player motion."""
import cv2
import numpy as np

class SceneDetector:
    def __init__(self):
        self.previous=None;self.histogram=None;self.last_cut=-float('inf')
    def step(self,frame,t):
        small=cv2.resize(frame,(160,90))
        hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
        histogram=cv2.calcHist([hsv],[0,1],None,[24,16],[0,180,0,256])
        cv2.normalize(histogram,histogram,alpha=1,norm_type=cv2.NORM_L1)
        cut=False;distance=0.;change=0.
        if self.previous is not None:
            diff=np.abs(small.astype(float)-self.previous.astype(float)).mean(axis=2)
            change=float(diff.mean())
            distance=float(cv2.compareHist(self.histogram,histogram,cv2.HISTCMP_BHATTACHARYYA))
            # Movement alone leaves the global court/stands colour distribution similar.
            cut=change>28 and float((diff>30).mean())>.35 and distance>.45 and t-self.last_cut>=.5
            if cut:self.last_cut=t
        self.previous=small;self.histogram=histogram
        return bool(cut),dict(appearance_distance=distance,pixel_change=change)


def broadcast_wide_view(frame):
    """Conservative green-court trapezoid gate, not a trained replay classifier."""
    hsv=cv2.cvtColor(cv2.resize(frame,(320,180)),cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([30,35,35]),np.array([100,255,255]))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    components,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not components:return False
    contour=max(components,key=cv2.contourArea)
    if cv2.contourArea(contour)<320*180*.22:return False
    fill=np.zeros_like(mask);cv2.drawContours(fill,[cv2.convexHull(contour)],-1,255,-1)
    def span(y):
        xs=np.flatnonzero(fill[int(y*179)]>0)
        return (xs[-1]-xs[0])/320 if len(xs) else 0.
    top,bottom=span(.40),span(.85)
    return bool(.18<top<.75 and bottom>.55 and bottom>top*1.12)
