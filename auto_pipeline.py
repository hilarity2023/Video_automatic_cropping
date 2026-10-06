"""Automatic shuttle/pose based rally cutting. No per-video event timestamps."""
from __future__ import annotations
import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# A configured virtual environment must use its own installed dependencies.
if sys.prefix == sys.base_prefix and (ROOT / '.runtime').is_dir():
    sys.path.insert(0, str(ROOT / '.runtime'))
(ROOT / '.runtime' / 'yolo_config').mkdir(parents=True, exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.runtime' / 'yolo_config'))

DEFAULT_INPUT = ROOT / 'input.mp4'
DEFAULT_START_SECONDS = 0.0
DEFAULT_END_SECONDS = 11450.0
DEFAULT_SETUP_SECONDS = 12.0  # 人工标注使用原视频第几秒的画面；不改变分析起止时间；最好选一个球和人都在画面内的时间
DEFAULT_OUTPUT = ROOT / 'output'


@dataclass
class Settings:
    # 主要参数
    post_roll: float = 2.0  # 导出时在估计结束后额外保留的时间（秒）；不计入 t_limit 的时长筛选，可能受捡球/边界限制
    pre_roll: float = 2.5  # 固定机位多留发球准备时间；受分析起点限制，不计入回合时长筛选
    t_limit: float = 8.0  # 实际回合必须超过此时长（秒）；前滚、后滚及兜底延长均不计入
    frame_stride: int = 5  # 每隔多少个原视频帧检测一次；越大越快，但更容易漏掉瞬间动作（不改变输出视频帧率）

    # 默认 context 模式按球员动作分段；标注“严格模式”的参数用于 --mode strict 的羽毛球事件判定。
    # 羽毛球模型始终启用，回合结束由球停止或向外出画消失确认；context模式允许持续捡球姿态作为兜底。
    # 归一化坐标：x=像素横坐标/画面宽度，y=像素纵坐标/画面高度，左上角为 (0, 0)。
    # 配置文件和命令行可以覆盖此处默认值；实际生效参数以启动输出为准。
    analysis_fps: float = 15.0  # 旧版分析帧率兼容参数（帧/秒）；当前不控制采样，实际由 frame_stride 决定
    pose_fps: float = 10.0  # 旧版姿态帧率兼容参数（帧/秒）；当前不控制采样，校验仍要求不大于 analysis_fps
    shuttle_fps: float = 5.0  # 羽毛球模型每秒最多检测次数；球员仍按frame_stride采样
    shuttle_search_tail: float = 3.0  # 最后对打动作后查找球结束证据的最长时间，禁止跨越后续回合
    context_fallback_tail: float = 1.5  # context模式未确认落地/捡球时，最后击球后的最大片尾；避免延伸进捡球
    imgsz: int = 1280  # 羽毛球模型推理尺寸（像素），至少 320 且为 32 的倍数；不控制球员姿态模型尺寸
    pose_imgsz: int = 768  # 姿态模型尺寸；提高远侧球员召回率，768 相对 640 通常只增加少量耗时
    confidence: float = 0.18  # 羽毛球检测置信度下限（0~1）；越低候选越多、误检也可能增加
    pose_confidence: float = 0.35  # 旧版姿态置信度兼容参数；当前姿态检测未读取此值，修改它不会改变检测阈值
    max_players: int = 2  # 无人工标注时自动跟踪的球员数上限；人工模式以标注人数为准
    hold_seconds: float = 0.12  # 严格模式：球在手附近持续达到此时间（秒），才可作为持球证据
    hand_radius: float = 0.035  # 严格模式及羽毛球静态过滤：球与手的邻近距离阈值（归一化坐标距离）
    depart_distance: float = 0.025  # 严格模式：相对持球位置的最小离手距离，也是确认飞行的位移阈值（归一化距离）
    flight_confirm: float = 0.12  # 严格模式：疑似离手后继续观察的最短时间（秒），用于确认飞行
    min_flight: float = 0.8  # 严格模式的最短飞行时间；context结束确认也须晚于最后对打动作至少此时间（秒）
    landing_confirm: float = 0.30  # 严格模式：球在地面区域持续静止达到此时间（秒），才确认落地
    still_radius: float = 0.009  # 严格模式：相邻球位置及落地静止位置的距离容差（归一化距离）
    max_gap: float = 0.30  # 严格模式：持球或轨迹连续观测允许的最大间隔（秒）；必须小于 lost_timeout
    lost_timeout: float = 1.5  # 严格模式：球丢失超过此时间（秒）后判为未知结束，不作为已确认落地
    static_seconds: float = 0.8  # 启用羽毛球检测时：候选持续静止达到此时间（秒）后作为背景过滤，手附近最多额外保留同等时长
    max_idle_flight: float = 2.0  # 严格模式：飞行中的球长期缺少明显位移超过此时间（秒）时，判为未知结束
    max_speed: float = 2.0  # 严格模式：轨迹匹配允许的最大球速（归一化坐标距离/秒），用于排除位置突跳
    shuttle_stop_seconds: float = 0.5  # 羽毛球持续静止达到此时间（秒）后确认回合结束
    stop_followup_seconds: float = 1.8  # 静止球/地面停球候选需再观察此时长，后续击球可否决结束
    shuttle_exit_seconds: float = 0.8  # 所有区域统一：球向画面外飞出后，消失超过此时间（秒）确认结束
    air_shuttle_stop_seconds: float = 1.5  # 地面区域外持续静止的结束阈值（秒），必须大于地面阈值
    shuttle_exit_margin: float = 0.04  # 出画判断的图像边缘带宽（归一化坐标）
    court: tuple = ((0.05,0.03),(0.96,0.03),(0.98,0.98),(0.05,0.98))  # 严格模式：允许羽毛球出现的球场多边形顶点，使用归一化 (x, y) 坐标
    players: tuple = ((0.12,0.52),(0.36,0.46),(0.67,0.50),(0.98,1.0),(0.1,1.0))  # 旧版球员区域多边形兼容参数；当前主球员跟踪未使用此区域
    ground: tuple = ((0.13,0.52),(0.55,0.51),(0.98,0.68),(0.96,0.98),(0.17,0.98))  # 严格模式：用于确认落地的地面区域多边形顶点，使用归一化 (x, y) 坐标

    def validate(self):
        if self.air_shuttle_stop_seconds<=self.shuttle_stop_seconds:
            raise ValueError('地面区域外的静止时间阈值必须大于地面静止阈值')
        for k,v in asdict(self).items():
            if k in {'court','players','ground'}:
                if len(v)<3 or any(len(p)!=2 or any(not math.isfinite(q) or not 0<=q<=1 for q in p) for p in v):
                    raise ValueError(f'{k}: expected normalized polygon')
            elif not math.isfinite(v) or v < 0 or (k not in {'pre_roll','post_roll','t_limit'} and v==0):
                raise ValueError(f'invalid {k}')
        if not isinstance(self.frame_stride,int) or isinstance(self.frame_stride,bool) or self.frame_stride<1:
            raise ValueError('frame_stride must be a positive integer')
        if self.pose_fps>self.analysis_fps or self.max_gap>=self.lost_timeout:
            raise ValueError('pose_fps <= analysis_fps; max_gap < lost_timeout')
        if not 0<self.confidence<=1 or not 0<self.pose_confidence<=1:
            raise ValueError('confidence must be in (0,1]')
        if (self.imgsz<320 or self.imgsz%32 or not isinstance(self.pose_imgsz,int)
                or self.pose_imgsz<320 or self.pose_imgsz%32 or not isinstance(self.max_players,int)):
            raise ValueError('imgsz and pose_imgsz >=320 and divisible by32; max_players is an integer')


@dataclass
class Ball:
    t: float
    x: float
    y: float
    conf: float=1.0


def dist(a,b):
    return math.hypot(a.x-b.x,a.y-b.y)


def inside(x,y,polygon):
    result=False
    for i,(x1,y1) in enumerate(polygon):
        x2,y2=polygon[(i+1)%len(polygon)]
        if (y1>y)!=(y2>y) and x<(x2-x1)*(y-y1)/(y2-y1)+x1:
            result=not result
    return result


class Events:
    """Hand proximity + departure, then observed descent + ground rest.

    Unknown ends are rejected. A hand-proximity proxy is still not a trained
    release-action classifier; diagnostic evidence is recorded explicitly.
    """
    def __init__(self,c):
        self.c=c; self.records=[]; self.scene_start=0.0
        self.clear()

    def clear(self):
        self.anchor=None; self.held_since=None; self.held_last=None
        self.departure=None; self.release=None; self.last=None
        self.history=deque(); self.rest_anchor=None; self.rest_since=None
        self.activity_anchor=None; self.activity_time=None

    def finish_unknown(self,t,reason):
        if self.release is not None:
            self.records.append(dict(status='incomplete',release=self.release,
                                     observed_until=t,reason=reason))
        self.clear()

    def reset_scene(self,t):
        self.finish_unknown(t,'scene_change')
        self.scene_start=t

    def _nearest(self,balls):
        dt=max(1e-6,balls[0].t-self.last.t)
        possible=[p for p in balls if dist(p,self.last)<=self.c.max_speed*dt+0.01]
        return min(possible,key=lambda p:dist(p,self.last)-p.conf*.003) if possible else None

    def step(self,t,balls,wrists):
        c=self.c
        balls=[p for p in balls if inside(p.x,p.y,c.court)]
        if self.release is None and self.departure is None:
            # Track a ball actually observed near a player's wrist, not a white speck.
            near=[p for p in balls if any(math.hypot(p.x-x,p.y-y)<=c.hand_radius for x,y in wrists)]
            if self.anchor is not None and self.held_last is not None and t-self.held_last<=c.max_gap:
                moving=[p for p in balls if c.depart_distance<=dist(p,self.anchor)<=c.max_speed*(t-self.held_last)+.03
                        and not any(math.hypot(p.x-x,p.y-y)<c.hand_radius*.7 for x,y in wrists)]
                if self.held_last-self.held_since+1e-9>=c.hold_seconds and moving:
                    p=min(moving,key=lambda p:dist(p,self.anchor))
                    self.departure=self.held_last; self.last=p; self.history=deque([self.anchor,p])
                    return p
            if near:
                p=min(near,key=lambda p:dist(p,self.anchor)) if self.anchor else max(near,key=lambda p:p.conf)
                if self.anchor is None or dist(p,self.anchor)>c.hand_radius or t-self.held_last>c.max_gap:
                    self.held_since=t
                self.anchor=p; self.held_last=t
                return p
            if self.held_last is not None and t-self.held_last>c.max_gap:
                self.clear()
            return None
        p=self._nearest(balls) if balls and self.last is not None else None
        if p is None:
            self.rest_since=None; self.rest_anchor=None
            if self.last is not None and t-self.last.t>c.lost_timeout:
                self.finish_unknown(self.last.t,'tracking_lost_not_landing')
            return None
        previous=self.last
        if t-previous.t>c.max_gap:
            self.rest_since=None; self.rest_anchor=None; self.history.clear()
            if self.release is None:
                self.clear(); return None
        self.last=p;self.history.append(p)
        while self.history and t-self.history[0].t>0.65:self.history.popleft()
        if self.release is None:
            if t-self.departure>=c.flight_confirm:
                if len(self.history)>=3 and max(dist(p,q) for q in self.history)>=c.depart_distance:
                    self.release=self.departure
                    self.activity_anchor=p;self.activity_time=t
                else:self.clear()
            return p
        if self.activity_anchor is None or dist(p,self.activity_anchor)>c.still_radius*2:
            self.activity_anchor=p;self.activity_time=t
        elif t-self.activity_time>c.max_idle_flight:
            self.finish_unknown(t,'static_track_without_landing_evidence');return None
        if t-self.release<c.min_flight:return p
        # Do not treat loss, a turn near the apex, or EOF as landing.
        descending=any(p.y-q.y>0.015 for q in self.history if t-q.t>=.08)
        slow=dist(p,previous)<=c.still_radius
        if slow and inside(p.x,p.y,c.ground):
            if self.rest_since is None and descending:
                self.rest_since=previous.t;self.rest_anchor=previous
            if self.rest_since is not None:
                if dist(p,self.rest_anchor)>c.still_radius:
                    self.rest_since=None;self.rest_anchor=None
                elif t-self.rest_since+1e-9>=c.landing_confirm:
                    start=max(self.scene_start,self.release-c.pre_roll,0.0)
                    end=self.rest_since
                    self.records.append(dict(release=self.release,landing=end,start=start,end=end,
                        duration=end-start,
                        evidence='shuttle_near_wrist_then_departure; descent_then_ground_rest',
                        pre_roll_available=self.release-start))
                    from shuttle_endings import filter_rally_duration
                    filter_rally_duration(self.records[-1],c.t_limit)
                    self.clear()
        else:self.rest_since=None;self.rest_anchor=None
        return p


class StaticTargets:
    """Reject balls/distractors already stationary before a rally.

    Background camera translation is estimated from image corners; this is a
    temporal filter, not a manually specified ball mask or set of event times.
    """
    def __init__(self,c):
        self.c=c;self.tracks=[];self.gray=None

    def filter(self,frame,balls,wrists,t):
        import cv2
        import numpy as np
        gray=cv2.cvtColor(cv2.resize(frame,(480,270)),cv2.COLOR_BGR2GRAY)
        dx=dy=0.
        if self.gray is not None:
            features=cv2.goodFeaturesToTrack(self.gray,120,.02,12)
            if features is not None and len(features)>=8:
                nxt,status,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,features,None)
                if nxt is not None:
                    valid=status.ravel()==1
                    if valid.sum()>=8:
                        movement=(nxt-features).reshape(-1,2)[valid]
                        dx,dy=np.median(movement,axis=0)/np.array([480,270])
        self.gray=gray
        for track in self.tracks:
            track['x']+=float(dx);track['y']+=float(dy)
        self.tracks=[q for q in self.tracks if t-q['last']<1.0]
        used=set();out=[]
        for p in sorted(balls,key=lambda p:p.conf,reverse=True):
            matches=[(math.hypot(p.x-q['x'],p.y-q['y']),i,q) for i,q in enumerate(self.tracks) if i not in used]
            match=min(matches,key=lambda row:row[0]) if matches else None
            if match is not None and match[0]<.012:
                _,i,q=match;used.add(i)
                if match[0]>.004:q['since']=t
                static=t-q['since']>=self.c.static_seconds
                q.update(x=p.x,y=p.y,last=t)
            else:
                self.tracks.append(dict(x=p.x,y=p.y,last=t,since=t));used.add(len(self.tracks)-1);static=False
            near_hand=any(math.hypot(p.x-x,p.y-y)<=self.c.hand_radius for x,y in wrists)
            if not static or (near_hand and t-q['since']<2*self.c.static_seconds):out.append(p)
        return out


def eligible_wrists(kpts,box,w,h):
    height=box[3]-box[1]
    shoulders=[q[1] for q in kpts[5:7] if q[2]>=.35]
    hips=[q[1] for q in kpts[11:13] if q[2]>=.35]
    if not shoulders or not hips:return []
    shoulder=sum(shoulders)/len(shoulders);hip=sum(hips)/len(hips)
    if hip-shoulder < .14*height:return []
    return [(q[0]/w,q[1]/h) for q in kpts[9:11]
            if q[2]>=.35 and q[1]<=hip+.12*height]


class Vision:
    def __init__(self,c,ball_weights,pose_weights,device,detect_shuttle=True):
        from ultralytics import YOLO
        self.c=c;self.device=device
        for p in ((ball_weights,pose_weights) if detect_shuttle else (pose_weights,)):
            if not p.is_file():raise ValueError(f'Missing model: {p}; run python setup_models.py')
        self.pose=YOLO(str(pose_weights));self.ball=None;self.ball_ids=[]
        if detect_shuttle:
            self.ball=YOLO(str(ball_weights))
            self.ball_ids=[int(k) for k,v in self.ball.names.items()
                           if v.lower() in {'shuttlecock','shuttle','badminton','羽毛球'}]
            if not self.ball_ids:raise ValueError(f'Model has no shuttlecock class: {self.ball.names}')
        from player_features import MultiPlayerFeatures
        self.player=MultiPlayerFeatures(self.pose,device,max_players=c.max_players,imgsz=c.pose_imgsz)
        self.features=[]
        self.shuttle_hint=None;self.shuttle_hint_time=None
        self.next_pose=0.;self.wrists=[];self.next_shuttle=0.;self.last_shuttle_inferred=False

    def infer(self,frame,t):
        c=self.c;h,w=frame.shape[:2]
        balls=[]
        self.last_shuttle_inferred=self.ball is not None and t+1e-9>=getattr(self,'next_shuttle',0.)
        if self.last_shuttle_inferred:
            self.next_shuttle=t+1/c.shuttle_fps
            result=self.ball.predict(frame,imgsz=c.imgsz,conf=c.confidence,classes=self.ball_ids,
                                     device=self.device,verbose=False)[0]
            balls=[Ball(t,(b[0]+b[2])/2/w,(b[1]+b[3])/2/h,float(score))
                   for b,score in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist())]
        if self.shuttle_hint is not None:
            elapsed=t-self.shuttle_hint_time
            if 0<=elapsed<=1.0:
                # A click is an identity hint, never a fabricated detection or release.
                possible=[b for b in balls if math.hypot(b.x-self.shuttle_hint[0],b.y-self.shuttle_hint[1])<=c.max_speed*elapsed+.04]
                balls=sorted(possible,key=lambda b:math.hypot(b.x-self.shuttle_hint[0],b.y-self.shuttle_hint[1]))[:1]
            elif elapsed>1.0:self.shuttle_hint=None
        feature=self.player.infer(frame,t)
        self.features.append(feature)
        self.wrists=[]
        for player in feature.get('players',[feature]):
            if player.get('valid'):
                self.wrists.extend((q[0],q[1]) for q in player['keypoints'][9:11] if q[2]>=.35)
        return balls,self.wrists


def ffmpeg_path():
    import shutil
    path=shutil.which('ffmpeg')
    if path:return path
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def export(source,target,start,end):
    tmp=target.with_suffix('.partial.mp4')
    cmd=[ffmpeg_path(),'-hide_banner','-loglevel','error','-nostdin','-y',
         '-ss',f'{start:.6f}','-i',str(source),'-t',f'{end-start:.6f}',
         '-map','0:v:0','-map','0:a:0?','-c:v','libx264','-crf','20','-preset','fast',
         '-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-pix_fmt','yuv420p','-c:a','aac',
         '-movflags','+faststart',str(tmp)]
    try:
        result=subprocess.run(cmd,capture_output=True)
        if result.returncode:raise RuntimeError(result.stderr.decode('utf-8',errors='replace'))
        tmp.replace(target)
    finally:tmp.unlink(missing_ok=True)


def export_merged(source,target,intervals):
    """Encode consistent temporary parts, concatenate once, and clean all scratch files."""
    import tempfile
    import cv2
    intervals=sorted(intervals)
    if not intervals:raise ValueError('No intervals to merge')
    for index,(start,end) in enumerate(intervals,1):
        if not math.isfinite(start) or not math.isfinite(end) or start<0 or end<=start:
            raise ValueError(f'Invalid export interval #{index}: start={start!r}s, end={end!r}s; '
                             'expected finite values with 0 <= start < end')
    tmp=target.with_suffix('.partial.mp4')
    try:
        with tempfile.TemporaryDirectory(prefix='.merge_',dir=target.parent) as scratch:
            scratch=Path(scratch);lines=['ffconcat version 1.0']
            part_durations=[]
            for i,(start,end) in enumerate(intervals):
                part=scratch/f'part_{i:06d}.mp4'
                print(f'Preparing merged video: {i+1}/{len(intervals)}',flush=True)
                export(source,part,start,end)
                part_cap=cv2.VideoCapture(str(part))
                try:
                    part_fps=part_cap.get(cv2.CAP_PROP_FPS)
                    part_frames=part_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    part_durations.append(part_frames/part_fps if part_fps>0 and part_frames>0 else end-start)
                finally:part_cap.release()
                lines.append(f"file '{part.name}'")
            manifest=scratch/'parts.ffconcat'
            manifest.write_text('\n'.join(lines)+'\n',encoding='utf-8')
            result=subprocess.run([ffmpeg_path(),'-hide_banner','-loglevel','error','-nostdin','-y',
                '-f','concat','-safe','1','-i',str(manifest),'-map','0:v:0','-map','0:a:0?',
                '-c','copy','-movflags','+faststart',str(tmp)],capture_output=True)
            if result.returncode:raise RuntimeError(result.stderr.decode('utf-8',errors='replace'))
            tmp.replace(target)
            return part_durations
    finally:tmp.unlink(missing_ok=True)


def plan_export_tails(records, post_roll, fps, window_end, scene_cuts=(), video_duration=None):
    """Pad only accepted clips. Preserve detection boundaries and filtering decisions."""
    for r in records:
        if r['status']!='keep':continue
        limit=min([window_end]+[t for t in scene_cuts if t>r['end']])
        terminal_end=r.get('end_confirmation') in {
            'shuttle_stationary','outward_exit_missing','new_ground_stop_with_quiet_players',
            'pose_confirmed_retrieval','pose_tail_cap_without_confirmed_shuttle_end'}
        if terminal_end:limit=min(limit,r['end'])
        if r.get('ground_stop_time') is not None:limit=min(limit,max(r['end'],r['ground_stop_time']))
        if r.get('retrieval_start') is not None:limit=min(limit,max(r['end'],r['retrieval_start']))
        if r.get('pose_tail_capped'):limit=min(limit,r['end'])
        if video_duration is not None and video_duration>0:limit=min(limit,video_duration)
        # Also include the estimated landing frame, as before.
        effective_post_roll=0. if terminal_end else post_roll
        r['planned_export_end']=min(limit,r['end']+effective_post_roll+1/fps)
        r['post_roll_requested']=post_roll
        r['post_roll_available']=max(0.,min(effective_post_roll,limit-r['end']))
        r['export_duration']=r['planned_export_end']-r['start']
    # Padding must not duplicate the beginning of the following rally.
    kept=sorted((r for r in records if r['status']=='keep'),key=lambda r:r['start'])
    for current,following in zip(kept,kept[1:]):
        if current['planned_export_end']>following['start']:
            current['planned_export_end']=max(current['end'],following['start'])
            current['export_duration']=current['planned_export_end']-current['start']


def analyse(args,c):
    import cv2
    import numpy as np
    from scene_detection import SceneDetector,broadcast_wide_view
    from shuttle_endings import pose_active
    scene_detector=SceneDetector();view_samples=[]
    started=time.perf_counter()
    annotation=getattr(args,'annotation',None)
    annotations=[]
    shuttle_enabled=True  # End confirmation always requires shuttle observations.
    vision=Vision(c,args.weights,args.pose_weights,args.device,shuttle_enabled)
    if annotation:
        from manual_setup import read_setup_frame
        setup_time=annotation.get('frame_time',args.start_seconds)
        initial_frame,_=read_setup_frame(args.input,setup_time)
        from player_features import MultiPlayerFeatures
        vision.player=MultiPlayerFeatures(vision.pose,args.device,max_players=c.max_players,imgsz=c.pose_imgsz)
        vision.player.initialize_manual(initial_frame,annotation,setup_time)
        vision.shuttle_hint=annotation.get('shuttle');vision.shuttle_hint_time=setup_time
        c.ground=tuple(tuple(p) for p in annotation['court'])
        c.court=((0.,0.),(1.,0.),(1.,1.),(0.,1.))
        annotations.append(annotation)
    cap=cv2.VideoCapture(str(args.input))
    if not cap.isOpened():raise ValueError(f'Cannot open {args.input}')
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps<=0:cap.release();raise ValueError('Invalid video FPS')
    duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
    if duration>0 and args.max_seconds>duration:
        print(f'Requested end {args.max_seconds:.3f}s exceeds video duration {duration:.3f}s; using video end.',flush=True)
        args.max_seconds=duration
    if duration>0 and args.start_seconds>=duration:
        cap.release();raise ValueError('start-seconds is beyond video duration')
    if args.start_seconds and not cap.set(cv2.CAP_PROP_POS_MSEC,args.start_seconds*1000):
        cap.release();raise RuntimeError('Cannot seek to requested start time')
    first_frame=max(0,int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    folder=args.output/args.input.stem/time.strftime('%Y%m%d_%H%M%S')
    folder.mkdir(parents=True,exist_ok=False)
    engine=Events(c);static_filter=StaticTargets(c);previous_t=args.start_seconds-1/fps;last_analyzed=-1.;i=first_frame
    engine.scene_start=args.start_seconds
    vision.next_pose=args.start_seconds
    next_log=args.start_seconds;fallbacks=0;count=0;ball_frames=0;hand_frames=0;cuts=[];debug=None
    debug_stop_candidates=[]
    shuttle_samples=[]
    try:
        while cap.grab():
            t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if not math.isfinite(t) or (i>0 and t<=previous_t):
                t=max(i/fps,previous_t+1/fps);fallbacks+=1
            i+=1;previous_t=t
            # Keep original-video timestamps; infer only within [start, end).
            if t>=args.max_seconds:break
            if t<args.start_seconds:continue
            if (i-1-first_frame)%c.frame_stride:continue
            ok,frame=cap.retrieve()
            if not ok:raise RuntimeError(f'Frame decode failed at {t:.3f}s')
            scene_cut,scene_metrics=scene_detector.step(frame,t)
            wide=broadcast_wide_view(frame)
            view_samples.append(dict(t=t,wide=wide,scene_cut=scene_cut,**scene_metrics))
            if scene_cut:
                engine.reset_scene(t);vision.next_pose=t;vision.wrists=[];cuts.append(t);static_filter=StaticTargets(c)
                from player_features import MultiPlayerFeatures
                vision.player=MultiPlayerFeatures(vision.pose,args.device,max_players=c.max_players,imgsz=c.pose_imgsz)
                if annotation:
                    from manual_setup import select_annotation
                    annotation=select_annotation(frame,t)
                    if annotation.get('shuttle') is not None and vision.ball is None:
                        previous_features=vision.features
                        vision=Vision(c,args.weights,args.pose_weights,args.device,True)
                        vision.features=previous_features;shuttle_enabled=True
                    from player_features import MultiPlayerFeatures
                    vision.player=MultiPlayerFeatures(vision.pose,args.device,max_players=c.max_players,imgsz=c.pose_imgsz)
                    vision.player.initialize_manual(frame,annotation,t)
                    vision.shuttle_hint=annotation.get('shuttle');vision.shuttle_hint_time=t
                    c.ground=tuple(tuple(p) for p in annotation['court'])
                    annotations.append(annotation)
            balls,wrists=vision.infer(frame,t)
            inferred=getattr(vision,'last_shuttle_inferred',True)
            filtered=static_filter.filter(frame,balls,wrists,t) if shuttle_enabled and inferred else []
            point=engine.step(t,filtered,wrists) if shuttle_enabled and inferred else None
            if vision.features:
                held=(engine.release is None and engine.anchor is not None
                      and engine.held_since is not None and engine.held_last is not None
                      and engine.held_last-engine.held_since+1e-9>=c.hold_seconds
                      and t-engine.held_last<=c.max_gap)
                vision.features[-1]['ball_held'] = bool(held)
            count+=1;ball_frames+=bool(balls);hand_frames+=bool(wrists);last_analyzed=t
            if inferred:
                feature=vision.features[-1] if vision.features else {}
                shuttle_samples.append(dict(t=t,balls=balls,wrists=wrists,
                    acquisition_balls=filtered,play_active=pose_active(feature,c),
                    context_complete=bool(feature.get('context_complete',False))))
            if args.debug_video:
                canvas=cv2.resize(frame,(960,round(frame.shape[0]*960/frame.shape[1])))
                h,w=canvas.shape[:2]
                for p in balls:cv2.circle(canvas,(round(p.x*w),round(p.y*h)),4,(0,255,255),1)
                for x,y in wrists:cv2.circle(canvas,(round(x*w),round(y*h)),5,(255,0,0),2)
                if point:cv2.circle(canvas,(round(point.x*w),round(point.y*h)),8,(0,0,255),2)
                cv2.putText(canvas,f'{t:.3f}s active={engine.release is not None}',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
                if debug is None:
                    debug=cv2.VideoWriter(str(folder/'diagnostic.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps/c.frame_stride,(w,h))
                    if not debug.isOpened():raise RuntimeError('Cannot create diagnostic video')
                debug.write(canvas)
            if t>=next_log:
                print(f'Analysed {t:.1f}/{args.max_seconds:g}s; shuttle={str(ball_frames)+'/'+str(count) if shuttle_enabled else 'disabled'}; strict_events={sum("end" in r for r in engine.records)}; context_segmentation=after_analysis',flush=True)
                next_log=t+5
        engine.finish_unknown(max(0,last_analyzed),'analysis_limit_or_eof_not_landing')
    finally:
        cap.release()
        if debug is not None:debug.release()
    wide_fraction=sum(q['wide'] for q in view_samples)/max(1,len(view_samples))
    broadcast=(args.view_mode=='broadcast' or
               (args.view_mode=='auto' and wide_fraction>.15 and len(cuts)>=3))
    views_by_time={row["t"]:row for row in view_samples}
    gated_features=[]
    for feature in vision.features:
        q=dict(feature)
        sample=views_by_time[q['t']]
        if broadcast and not sample['wide']:
            q.update(valid=False,arm_speed=0.,foot_speed=0.,body_speed=0.,spread=0.,bent=False,overhead=False)
        gated_features.append(q)
    strict_records=engine.records
    if args.mode=='context':
        from temporal_events import segment_actions, TemporalSettings
        options=TemporalSettings(**args.temporal_settings)
        edges=[args.start_seconds]+cuts+[args.max_seconds]
        engine.records=[]
        for left,right in zip(edges,edges[1:]):
            features=[q for q in gated_features if left<=q['t']<right]
            following=[q for q in view_samples if right<=q['t']<right+1.]
            terminal_transition=(broadcast and right in cuts and len(following)>1
                and following[-1]['t']-following[0]['t']+1e-6>=.8
                and sum(q['wide'] for q in following)/len(following)<.2)
            from shuttle_endings import ground_stop_windows
            stops=ground_stop_windows([q for q in shuttle_samples if left<=q['t']<right],c,features) if not broadcast else []
            debug_stop_candidates.extend(dict(scene_start=left,scene_end=right,t=t) for t in stops)
            engine.records.extend(segment_actions(features,c.pre_roll,c.t_limit,options,left,right,
                broadcast=broadcast,terminal_transition=terminal_transition,
                pre_roll_floor=args.start_seconds if broadcast else None,stop_times=stops))
    from shuttle_endings import resolve_shuttle_endings
    proposals=engine.records
    action_proposals=[dict(q) for q in proposals]
    engine.records=[]
    edges=[args.start_seconds]+cuts+[args.max_seconds]
    for left,right in zip(edges,edges[1:]):
        candidates=[dict(q,start=q.get('start',max(left,q['release']-c.pre_roll))) for q in proposals if left<=q['release']<right]
        if args.mode=='context':
            from temporal_events import merge_action_proposals,TemporalSettings
            candidates=merge_action_proposals(candidates,
                [q for q in gated_features if left<=q['t']<right],
                TemporalSettings(**args.temporal_settings),
                scene_start=args.start_seconds if broadcast else left)
        rows=[q for q in shuttle_samples if left<=q['t']<right]
        # Each camera scene uses the court actually annotated for that scene.
        from dataclasses import replace
        scene_annotation=next((a for a in reversed(annotations) if a.get('frame_time',0)<=left+.1),annotations[0] if annotations else None)
        scene_c=replace(c,ground=tuple(tuple(p) for p in scene_annotation['court'])) if scene_annotation else c
        engine.records.extend(resolve_shuttle_endings(candidates,rows,scene_c,min(right,last_analyzed+1/fps),
            allow_pickup_fallback=args.mode=='context',terminal_flight=args.temporal_settings.get('terminal_flight',.8)))
    report=dict(source=str(args.input.resolve()),analysis_limit=args.max_seconds,analysis_start=args.start_seconds,analysis_end=args.max_seconds,
        last_analyzed=last_analyzed,sampled_frames=count,frames_with_shuttle=ball_frames,
        elapsed_analysis_seconds=time.perf_counter()-started,source_fps=fps,frame_stride=c.frame_stride,
        effective_analysis_fps=fps/c.frame_stride,shuttle_detection_enabled=shuttle_enabled,
        frames_with_wrists=hand_frames,timestamp_fallbacks=fallbacks,scene_changes=cuts,
        settings=asdict(c),weights=str(args.weights),pose_weights=str(args.pose_weights),
        view_mode_requested=args.view_mode,broadcast_gate_enabled=broadcast,wide_fraction=wide_fraction,
        manual_annotations=annotations,
        method=args.mode,temporal_settings=args.temporal_settings,strict_events=strict_records,
        source_size=args.input.stat().st_size,source_mtime_ns=args.input.stat().st_mtime_ns,
        accuracy_validated=False,dry_run=args.dry_run,segments=engine.records)
    plan_export_tails(engine.records,c.post_roll,fps,args.max_seconds,cuts,duration)
    if args.mode=='context' and not broadcast:
        from temporal_events import protect_active_tails
        protect_active_tails(engine.records,gated_features,fps,args.max_seconds,cuts,duration,
                             max_extension=2.0,rest_confirm=.5)
    kept_records=sorted((r for r in engine.records if r['status']=='keep'),key=lambda r:r['start'])
    kept=len(kept_records)
    report['output_mode']='merged'
    report['merged_segment_count']=kept
    saved=0
    if kept_records and not args.dry_run:
        target=folder/'highlights.mp4'
        try:
            part_durations=export_merged(args.input,target,[(r['start'],r['planned_export_end']) for r in kept_records])
            report['merged_file']=str(target)
            for order,r in enumerate(kept_records,1):
                r['merged_order']=order
                r['export_end']=r['planned_export_end']
            from shuttle_endings import write_merged_start_log
            write_merged_start_log(folder/'merged_starts.log',kept_records,part_durations)
            saved=1
        except Exception as e:
            report['export_error']=str(e);raise
    if args.debug_events:
        event_rows=[]
        for q in gated_features:
            player_rows=q.get('players',[q])
            def pickup_geometry(p):
                pts=p.get('keypoints')
                if pts is None:return None
                hips=[v for v in pts[11:13] if v[2]>.25]
                shoulders=[v for v in pts[5:7] if v[2]>.25]
                hands=[v for v in pts[9:11] if v[2]>.3]
                if not (hips and shoulders and hands):return None
                height=max(p['box'][3]-p['box'][1],.01)
                return [abs(sum(v[0] for v in shoulders)/len(shoulders)-sum(v[0] for v in hips)/len(hips))/height,
                        (max(v[1] for v in hands)-sum(v[1] for v in hips)/len(hips))/height]
            event_rows.append(dict(t=q['t'],valid=bool(q.get('valid')),
                context_complete=bool(q.get('context_complete',False)),
                group_rest=bool(q.get('group_rest',False)),ball_held=bool(q.get('ball_held',False)),
                players=[dict(player_id=p.get('player_id'),valid=bool(p.get('valid')),
                    arm_speed=float(p.get('arm_speed',0)),foot_speed=float(p.get('foot_speed',0)),
                    body_speed=float(p.get('body_speed',0)),spread=float(p.get('spread',0)),
                    overhead=bool(p.get('overhead',False)),bent=bool(p.get('bent',False)),
                    box=p.get('box'),pickup_geometry=pickup_geometry(p)) for p in player_rows]))
        debug_path=folder/'event_debug.json'
        debug_path.write_text(json.dumps(dict(stop_candidates=debug_stop_candidates,
            action_proposals=action_proposals,segments=engine.records,features=event_rows),
            ensure_ascii=False,indent=2),encoding='utf-8')
        report['event_debug_file']=str(debug_path)
    print(f'Finished: {kept} selected segments; {saved} merged video saved; dry_run={args.dry_run}. Output: {folder}',flush=True)
    for r in engine.records:
        if r['status']=='keep':
            flags=[]
            if r.get('start_confidence')=='uncertain':flags.append('start=uncertain')
            if r.get('end_uncertain'):flags.append('end=unconfirmed')
            suffix=('; review='+','.join(flags)) if flags else ''
            print(f"  export={r['start']:.3f}-{r['planned_export_end']:.3f}s; estimated_end={r['end']:.3f}s; qualifying_duration={r['qualifying_duration']:.3f}s; tail={r['post_roll_available']:.3f}s; adaptive_extension={r.get('tail_guard_extension',0):.3f}s{suffix}",flush=True)
    if not any(r['status']=='keep' for r in engine.records):
        print('No candidate passed duration/completeness checks.',flush=True)
    return report


def build_parser():
    p=argparse.ArgumentParser(description='羽毛球自动回合检测与剪辑；默认分析原视频 [0,11450) 秒')
    p.add_argument('input',nargs='?',type=Path,default=DEFAULT_INPUT)
    p.add_argument('-o','--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--start-seconds',type=float,default=DEFAULT_START_SECONDS,help='Original video start time in seconds')
    p.add_argument('--end-seconds','--max-seconds',dest='max_seconds',type=float,default=DEFAULT_END_SECONDS,
                   help='Original video end time (exclusive), not duration')
    p.add_argument('--t-limit','--t_limit',type=float,default=None)
    p.add_argument('--frame-stride',type=int,default=None,help='Pose sampling interval in source frames; fixed context defaults to 4 unless explicitly set')
    p.add_argument('--pre-roll',type=float,default=None)
    p.add_argument('--post-roll',type=float,default=None,help='Extra export tail, excluded from minimum-duration filtering')
    p.add_argument('--config',type=Path,default=ROOT/'config.auto.json')
    p.add_argument('--weights',type=Path,default=ROOT/'models'/'shuttle_yolo11s.pt')
    p.add_argument('--pose-weights',type=Path,default=ROOT/'models'/'yolo11n-pose.pt')
    p.add_argument('--device',default='cpu')
    p.add_argument('--mode',choices=['context','strict'],default='context',
                   help='context: pose-based estimated boundaries; strict: observed shuttle events only')
    p.add_argument('--view-mode',choices=['auto','fixed','broadcast'],default='auto',
                   help='Broadcast excludes close-ups and side-view shots; auto uses court layout and scene changes')
    p.add_argument('--setup-seconds',type=float,default=DEFAULT_SETUP_SECONDS,help=f'原视频中用于人工标注的时刻（默认 {DEFAULT_SETUP_SECONDS:g} 秒；不改变分析范围）')
    p.add_argument('--no-manual-setup',action='store_true',help='Skip interactive court/player/shuttle setup')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--debug-video',action='store_true')
    p.add_argument('--debug-events',action='store_true',help='Write sampled pose, held-ball, stop-candidate, and segment decision evidence')
    return p


def load_settings(args):
    if not args.config.is_file():raise ValueError(f'Config does not exist: {args.config}')
    values=json.loads(args.config.read_text(encoding='utf-8-sig'))
    args.frame_stride_configured=('frame_stride' in values or args.frame_stride is not None)
    values.pop('air_shuttle_exit_seconds',None)  # 兼容旧配置，出画时间统一由 shuttle_exit_seconds 控制
    args.temporal_settings=values.pop('temporal',{})
    from temporal_events import TemporalSettings
    TemporalSettings(**args.temporal_settings).validate()
    c=Settings(**values)
    if args.frame_stride is not None:c.frame_stride=args.frame_stride
    if args.t_limit is not None:c.t_limit=args.t_limit
    if args.pre_roll is not None:c.pre_roll=args.pre_roll
    if args.post_roll is not None:c.post_roll=args.post_roll
    c.validate()
    return c


def validate_window(args):
    if not math.isfinite(args.start_seconds) or args.start_seconds<0:
        raise ValueError("start-seconds must be finite and nonnegative")
    if not math.isfinite(args.max_seconds) or args.max_seconds<=args.start_seconds:
        raise ValueError("end-seconds must be finite and greater than start-seconds")


def main():
    p=build_parser();args=p.parse_args()
    try:
        c=load_settings(args)
        validate_window(args)
        if not args.input.is_file():raise ValueError(f'Input file does not exist: {args.input}')
        if not args.no_manual_setup:
            if args.view_mode=='broadcast':raise ValueError('人工球场标注适用于固定机位；转播模式请使用 --no-manual-setup')
            from manual_setup import read_setup_frame,select_annotation
            if not math.isfinite(args.setup_seconds) or args.setup_seconds<0:raise ValueError('setup-seconds must be finite and nonnegative')
            frame,t=read_setup_frame(args.input,args.setup_seconds)
            args.annotation=select_annotation(frame,t,video_path=args.input)
            if args.annotation is not None:
                args.view_mode='fixed'
            else:
                print('Manual setup skipped; continuing with automatic detection.',flush=True)
        if (args.mode=='context' and args.view_mode=='fixed'
                and not args.frame_stride_configured):
            c.frame_stride=4  # 5→4 is about 25% more pose samples on fixed-camera context runs.
            print('Fixed context sampling: frame_stride=4 (about 25% more pose samples).',flush=True)
        print(f'Effective settings: range=[{args.start_seconds:g}, {args.max_seconds:g})s; '
              f'frame_stride={c.frame_stride}; pre_roll={c.pre_roll:g}s; post_roll={c.post_roll:g}s; t_limit={c.t_limit:g}s; config={args.config}',flush=True)
        analyse(args,c)
        return 0
    except (ValueError,TypeError,OSError,RuntimeError,ImportError) as e:
        print(f'Error: {e}',file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
