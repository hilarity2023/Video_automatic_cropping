"""Reusable pose features and camera-motion compensation for rally segmentation."""
import cv2
import numpy as np

class PlayerFeatures:
    def __init__(self,pose,device='cpu',imgsz=640):
        self.pose=pose;self.device=device;self.imgsz=imgsz;self.appearance=None;self.last_box=None
        self.last_seen=None;self.velocity=np.zeros(2)
        self.previous_gray=None;self.previous_points=None;self.previous_t=None
        self.manual_court=None;self.manual_height=None;self.reference_appearance=None;self.next_recovery=0.;self.recovery_burst_until=-1.;self.recovery_burst_cooldown=-1.
    def initialize_manual(self,frame,annotation,t):
        from manual_setup import validate_annotation
        validate_annotation(annotation)
        self.manual_court=np.asarray(annotation['court'],np.float32)
        box=np.asarray(annotation['player_box'],float);h,w=frame.shape[:2]
        x1,y1,x2,y2=box*[w,h,w,h];bh=y2-y1
        crop=frame[max(0,int(y1+.25*bh)):min(h,int(y1+.6*bh)),max(0,int(x1)):min(w,int(x2))]
        if not crop.size:raise ValueError('主球员标注框没有有效图像')
        hist=cv2.calcHist([cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)],[0,1],None,[18,8],[0,180,0,256]).ravel()
        hist/=max(float(np.linalg.norm(hist)),1e-6)
        self.appearance=hist;self.reference_appearance=hist.copy();self.last_box=box.tolist();self.last_seen=t
        self.manual_height=float(box[3]-box[1]);self.velocity=np.zeros(2)
        self.previous_points=None;self.previous_t=None

    def initialize_auto(self,frame,t,box,points):
        """Seed a tracker from one full-frame pose detection."""
        h,w=frame.shape[:2]
        box=np.asarray(box,float)
        normalized=box/np.array([w,h,w,h])
        x1,y1,x2,y2=box
        crop=frame[max(0,int(y1+.25*(y2-y1))):min(h,int(y1+.6*(y2-y1))),
                   max(0,int(x1)):min(w,int(x2))]
        if not crop.size:return False
        hist=cv2.calcHist([cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)],[0,1],None,[18,8],[0,180,0,256]).ravel()
        hist/=max(float(np.linalg.norm(hist)),1e-6)
        self.appearance=hist;self.reference_appearance=hist.copy()
        self.last_box=normalized.tolist();self.last_seen=float(t)
        self.velocity=np.zeros(2)
        self.previous_points=np.asarray(points,float)/np.array([w,h,1])
        self.previous_t=float(t)
        return True

    def allowed_player(self,box):
        x1,y1,x2,y2=box
        if self.manual_court is not None:
            # Court is a ground-plane constraint on feet, never an airborne-ball mask.
            foot=((float(x1)+float(x2))/2,float(y2))
            return (y2-y1>=max(.03,self.manual_height*.5) and
                    cv2.pointPolygonTest(self.manual_court,foot,True)>=-.015)
        return y2-y1>=.12 and y2>=.48 and (x1+x2)/2>=(.12 if self.appearance is None else .08)

    def camera_shift(self,frame):
        gray=cv2.cvtColor(cv2.resize(frame,(480,270)),cv2.COLOR_BGR2GRAY)
        shift=np.zeros(2)
        if self.previous_gray is not None:
            corners=cv2.goodFeaturesToTrack(self.previous_gray,160,.02,10)
            if corners is not None:
                nxt,status,_=cv2.calcOpticalFlowPyrLK(self.previous_gray,gray,corners,None)
                if nxt is not None and np.count_nonzero(status)>=12:
                    delta=(nxt-corners).reshape(-1,2)[status.ravel()==1]
                    shift=np.median(delta,axis=0)/[480,270]
        self.previous_gray=gray
        return shift
    def association(self,box,hist,t,shift):
        """Appearance plus bounded predicted movement; avoid jumping to neighbours."""
        if self.appearance is None:return float(box[3]-box[1])
        # A reference frame at another time provides appearance, not current position.
        if self.manual_court is not None and self.previous_points is None and abs(t-self.last_seen)>.5:
            colour=float(np.dot(self.appearance,hist))
            ratio=(box[3]-box[1])/max(self.manual_height,.01)
            return colour if colour>=.5 and .5<ratio<2.5 else None
        prior=np.asarray(self.last_box);elapsed=max(.01,t-self.last_seen)
        center=np.array([(box[0]+box[2])/2,box[3]])
        previous=np.array([(prior[0]+prior[2])/2,prior[3]])
        predicted=previous+self.velocity*min(elapsed,.5)+shift
        distance=float(np.linalg.norm(center-predicted))
        colour=float(np.dot(self.appearance,hist))
        if self.reference_appearance is not None and elapsed>.6:colour=max(colour,float(np.dot(self.reference_appearance,hist)))
        gate=.075+.25*min(elapsed,1.)
        ratio=(box[3]-box[1])/max(prior[3]-prior[1],.01)
        if distance>gate or not .45<ratio<2.2:return None
        # Turning around or shrinking mixes more court pixels into a shirt crop.
        if colour<.25 and not (elapsed<.6 and colour>=.08 and distance<gate*.6):return None
        return 2*colour-4*distance-.2*abs(float(np.log(ratio)))

    def infer(self,frame,t,result=None,excluded=None,camera_motion=None):
        h,w=frame.shape[:2];shift=self.camera_shift(frame) if camera_motion is None else camera_motion
        r=result if result is not None else self.pose.predict(frame,imgsz=self.imgsz,conf=.25,verbose=False,device=self.device)[0]
        candidates=[]
        def collect(result,offset=(0,0),source='full'):
            if result.keypoints is None:return
            for raw_box,raw_pts in zip(result.boxes.xyxy.cpu().numpy(),result.keypoints.data.cpu().numpy()):
                box=raw_box.copy();pts=raw_pts.copy()
                box+=np.array([offset[0],offset[1],offset[0],offset[1]])
                pts[:,:2]+=np.array(offset)
                normalized=box/np.array([w,h,w,h])
                if any(box_iou(normalized,other)>.5 for other in (excluded or [])):continue
                x1,y1,x2,y2=box;bh=y2-y1
                if not self.allowed_player(box/np.array([w,h,w,h])):continue
                crop=frame[max(0,int(y1+.25*bh)):min(h,int(y1+.6*bh)),max(0,int(x1)):min(w,int(x2))]
                if crop.size==0:continue
                hist=cv2.calcHist([cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)],[0,1],None,[18,8],[0,180,0,256]).ravel()
                hist/=max(float(np.linalg.norm(hist)),1e-6)
                score=self.association(box/np.array([w,h,w,h]),hist,t,shift)
                if source=='recovery_crop' and self.appearance is not None and max(float(np.dot(self.appearance,hist)),float(np.dot(self.reference_appearance,hist)) if self.reference_appearance is not None else 0)<.5:continue
                if score is not None:candidates.append((score,box,pts,hist,source))
        collect(r)
        if not candidates and self.last_box is not None and self.last_seen is not None and t>=self.last_seen and (self.manual_court is None or t-self.last_seen<=.3 or t>=self.next_recovery) and (t-self.last_seen<=2. or (self.manual_court is not None and t>=self.next_recovery)):
            self.next_recovery=t+.25
            previous=np.asarray(self.last_box)
            center=(previous[:2]+previous[2:])/2+self.velocity*min(t-self.last_seen,.4)
            size=np.maximum((previous[2:]-previous[:2])*(4 if t-self.last_seen>2 else 2.5),[.4,.45] if t-self.last_seen>2 else [.25,.35])
            lo=np.maximum(center-size/2,0);hi=np.minimum(center+size/2,1)
            x1,y1=np.floor(lo*[w,h]).astype(int);x2,y2=np.ceil(hi*[w,h]).astype(int)
            crop=frame[y1:y2,x1:x2]
            if crop.size:
                recovered=self.pose.predict(crop,imgsz=self.imgsz,conf=.15,verbose=False,device=self.device)[0]
                collect(recovered,(x1,y1),'recovery_crop')
        if not candidates:return dict(t=t,valid=False,camera=shift.tolist())
        score,box,pts,hist,tracking_source=max(candidates,key=lambda q:q[0])
        colour=float(np.dot(self.appearance,hist)) if self.appearance is not None else 1.
        if self.appearance is None:self.appearance=hist
        elif colour>=.5:
            self.appearance=.98*self.appearance+.02*hist
            self.appearance/=np.linalg.norm(self.appearance)
        box=box/np.array([w,h,w,h])
        if self.last_box is not None and self.last_seen is not None and 0<t-self.last_seen<.6:
            prior=np.asarray(self.last_box)
            previous=np.array([(prior[0]+prior[2])/2,prior[3]])
            center=np.array([(box[0]+box[2])/2,box[3]])
            velocity=(center-previous-shift)/(t-self.last_seen)
            speed=float(np.linalg.norm(velocity))
            if speed>.8:velocity*=.8/speed
            self.velocity=.5*self.velocity+.5*velocity
        else:self.velocity=np.zeros(2)
        self.last_box=box.tolist();self.last_seen=t
        pts=pts/np.array([w,h,1])
        height=box[3]-box[1];width=box[2]-box[0]
        def mean(ids):
            good=[pts[i,:2] for i in ids if pts[i,2]>.25]
            return np.mean(good,axis=0) if good else None
        hip=mean([11,12]);shoulder=mean([5,6]);ankle=mean([15,16])
        torso=(hip[1]-shoulder[1])/height if hip is not None and shoulder is not None else None
        feetspread=abs(pts[15,0]-pts[16,0])/height if min(pts[15,2],pts[16,2])>.25 else 0.
        hands=[pts[i,:2] for i in [9,10] if pts[i,2]>.3]
        overhead=bool(shoulder is not None and any(p[1]<shoulder[1]-.04*height for p in hands))
        bent=bool(torso is not None and torso<.13)
        # Briefly confirm a recovered bending pose at the normal sampled cadence.
        # A fixed burst and cooldown prevent continuous expensive crop inference.
        if tracking_source=='recovery_crop' and bent and t>=self.recovery_burst_cooldown:
            self.recovery_burst_until=t+.35
            self.recovery_burst_cooldown=t+1.
        if t<self.recovery_burst_until:
            self.next_recovery=t
        arm_speed=foot_speed=body_speed=0.
        if self.previous_points is not None and self.previous_t is not None and 0<t-self.previous_t<.4:
            dt=t-self.previous_t
            old=self.previous_points
            valid=(old[:,2]>.3)&(pts[:,2]>.3)
            oldhip=np.mean(old[11:13,:2],axis=0);newhip=np.mean(pts[11:13,:2],axis=0)
            body_speed=float(np.linalg.norm(newhip-oldhip-shift)/dt/max(height,.08))
            arm=[float(np.linalg.norm((pts[k,:2]-newhip)-(old[k,:2]-oldhip))/dt/max(height,.08)) for k in [9,10] if valid[k]]
            foot=[float(np.linalg.norm(pts[k,:2]-old[k,:2]-shift)/dt/max(height,.08)) for k in [15,16] if valid[k]]
            arm_speed=max(arm,default=0.);foot_speed=max(foot,default=0.)
        self.previous_points=pts;self.previous_t=t
        return dict(t=t,valid=True,tracking_source=tracking_source,box=box.tolist(),keypoints=pts.tolist(),camera=shift.tolist(),torso=torso,
                    spread=float(feetspread),overhead=overhead,bent=bent,arm_speed=arm_speed,foot_speed=foot_speed,body_speed=body_speed)


def box_iou(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/max(union,1e-9)


def combine_players(rows,t):
    """Use the strongest observed player's own features, never cross-player velocities."""
    visible=[q for q in rows if q.get('valid')]
    if not visible:return dict(t=t,valid=False,players=rows,context_complete=False,group_rest=False)
    def evidence(q):
        pts=np.asarray(q['keypoints']);height=max(q['box'][3]-q['box'][1],.01)
        hips=[p[1] for p in pts[11:13] if p[2]>.25];hands=[p[1] for p in pts[9:11] if p[2]>.3]
        if not hips or not hands:return (-1.,False,False)
        hand=(np.mean(hips)-min(hands))/height
        arm=q.get('arm_speed',0);foot=q.get('foot_speed',0);body=q.get('body_speed',0);spread=q.get('spread',0)
        strong=(arm>.65 and hand>.18) or (arm>1. and hand>.08) or (spread>.25 and foot>.8 and arm>.9)
        pickup=q.get('bent',False) and ((foot<.8 and body<.65 and spread<.3) or (spread<.3 and body<1.5 and foot<2.))
        ready=(hand>.12 and spread>.12) or (hand>.18 and arm>.3) or (foot>.9 and body>.45 and hand>.08)
        # An active teammate overrides another player's bending/standing still.
        score=(10 if strong and not pickup else 5 if ready and not pickup else 2 if pickup else 0)+min(arm,3)*.1
        rest=arm<.35 and foot<.3 and body<.2 and not ready
        return score,rest,True
    signals=[evidence(q) for q in rows if q.get('valid')]
    chosen=max(visible,key=lambda q:evidence(q)[0])
    complete=len(visible)==len(rows) and all(x[2] for x in signals)
    return dict(chosen,players=rows,context_complete=complete,group_rest=complete and all(x[1] for x in signals))


class MultiPlayerFeatures:
    def __init__(self,pose,device='cpu',max_players=2,imgsz=640):
        self.pose=pose;self.device=device;self.max_players=max(1,int(max_players));self.imgsz=imgsz
        self.trackers=[];self.auto_tracking=True;self.pending=[]
    def initialize_manual(self,frame,annotation,t):
        from manual_setup import validate_annotation
        validate_annotation(annotation)
        boxes=annotation.get('player_boxes') or [annotation['player_box']]
        self.trackers=[];self.pending=[];self.auto_tracking=False
        for box in boxes:
            tracker=PlayerFeatures(self.pose,self.device,self.imgsz)
            tracker.initialize_manual(frame,dict(court=annotation['court'],player_box=box,shuttle=None),t)
            self.trackers.append(tracker)
    def infer(self,frame,t):
        # One full-frame inference shared by every identity; local recovery is only
        # attempted for missing identities. Reserve boxes to prevent duplicate tracks.
        result=self.pose.predict(frame,imgsz=self.imgsz,conf=.25,verbose=False,device=self.device)[0]
        if not self.trackers:
            h,w=frame.shape[:2]
            detections=[]
            if result.keypoints is not None:
                for raw_box,raw_pts in zip(result.boxes.xyxy.cpu().numpy(),result.keypoints.data.cpu().numpy()):
                    normalized=raw_box/np.array([w,h,w,h])
                    if normalized[3]-normalized[1]>=.12 and normalized[3]>=.48 and (normalized[0]+normalized[2])/2>=.12:
                        detections.append((float(normalized[3]-normalized[1]),raw_box,raw_pts))
            for _,box,points in sorted(detections,key=lambda row:row[0],reverse=True):
                normalized=box/np.array([w,h,w,h])
                if any(box_iou(normalized,tracker.last_box)>.5 for tracker in self.trackers):continue
                tracker=PlayerFeatures(self.pose,self.device,self.imgsz)
                if tracker.initialize_auto(frame,t,box,points):self.trackers.append(tracker)
                if len(self.trackers)>=self.max_players:break
            if not self.trackers:
                return dict(t=t,valid=False,players=[],context_complete=False,group_rest=False)
        rows=[];used=[]
        shift=self.trackers[0].camera_shift(frame) if self.trackers else np.zeros(2)
        for index,tracker in enumerate(self.trackers):
            q=tracker.infer(frame,t,result=result,excluded=used,camera_motion=shift);q['player_id']=index+1
            if q.get('valid'):used.append(q['box'])
            rows.append(q)
        if self.auto_tracking and len(self.trackers)<self.max_players and result.keypoints is not None:
            h,w=frame.shape[:2];available=[]
            for raw_box,raw_pts in zip(result.boxes.xyxy.cpu().numpy(),result.keypoints.data.cpu().numpy()):
                normalized=raw_box/np.array([w,h,w,h])
                if (normalized[3]-normalized[1]<.12 or normalized[3]<.48
                        or (normalized[0]+normalized[2])/2<.12):continue
                crop=frame[max(0,int(raw_box[1]+.25*(raw_box[3]-raw_box[1]))):min(h,int(raw_box[1]+.6*(raw_box[3]-raw_box[1]))),
                           max(0,int(raw_box[0])):min(w,int(raw_box[2]))]
                if not crop.size:continue
                hist=cv2.calcHist([cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)],[0,1],None,[18,8],[0,180,0,256]).ravel()
                hist/=max(float(np.linalg.norm(hist)),1e-6)
                if any(box_iou(normalized,box)>.35 for box in used):continue
                if any(tracker.association(normalized,hist,t,shift) is not None for tracker in self.trackers):continue
                available.append((normalized,raw_box,raw_pts,hist))

            self.pending=[q for q in self.pending if t-q['last_seen']<=.6]
            matched=set()
            for normalized,box,points,hist in available:
                center=np.array([(normalized[0]+normalized[2])/2,normalized[3]])
                options=[(float(np.linalg.norm(center-q['center'])),index,q) for index,q in enumerate(self.pending)
                         if index not in matched and float(np.dot(hist,q['hist']))>=.35]
                match=min(options,key=lambda row:row[0]) if options else None
                if match is None or match[0]>.14:
                    self.pending.append(dict(center=center,hist=hist,box=box.copy(),points=points.copy(),
                                             hits=1,last_seen=float(t)))
                    matched.add(len(self.pending)-1)
                    continue
                _,index,pending=match;matched.add(index)
                pending.update(center=center,hist=hist,box=box.copy(),points=points.copy(),
                               hits=pending['hits']+1,last_seen=float(t))
                if pending['hits']<3:continue
                tracker=PlayerFeatures(self.pose,self.device,self.imgsz)
                if not tracker.initialize_auto(frame,t,pending['box'],pending['points']):continue
                tracker.camera_shift(frame)
                player_id=len(self.trackers)+1;self.trackers.append(tracker)
                q=tracker.infer(frame,t,result=result,excluded=used,camera_motion=shift);q['player_id']=player_id
                if q.get('valid'):used.append(q['box'])
                rows.append(q)
                self.pending=[]
                if len(self.trackers)>=self.max_players:break
        return combine_players(rows,t)
