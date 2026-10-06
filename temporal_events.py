"""Offline kinematic rally proposals. All boundaries are estimates, not ground truth.

References/annotations are deliberately not accepted by this module.
"""
from dataclasses import dataclass
import math
import numpy as np


def retrieval_pose(player):
    """Low, leaning retrieval geometry; athletic wide lunges remain playable."""
    if not player.get('valid'):return False
    if (player.get('spread',0)>=.3 or player.get('body_speed',0)>=.65
            or player.get('foot_speed',0)>=2.):return False
    if player.get('bent',False):return True
    pts=player.get('keypoints');box=player.get('box')
    if pts is None or box is None:return False
    height=max(box[3]-box[1],.01)
    hips=[v for v in pts[11:13] if v[2]>.25]
    shoulders=[v for v in pts[5:7] if v[2]>.25]
    hands=[v for v in pts[9:11] if v[2]>.3]
    if not hips or not hands:return False
    hip_y=np.mean([v[1] for v in hips])
    lowered=(max(v[1] for v in hands)-hip_y)/height
    torso=player.get('torso')
    leaning=bool(shoulders and
        (hip_y-np.mean([v[1] for v in shoulders]))/height<.25 and
        abs(np.mean([v[0] for v in shoulders])-np.mean([v[0] for v in hips]))/height>.09)
    return bool((lowered>.22 and player.get('foot_speed',0)<.8) or
                (lowered>.12 and torso is not None and torso<.23) or
                (lowered>0 and leaning))


def retrieval_frame(feature, settings):
    players=feature.get('players') or [feature]
    retrieving=any(retrieval_pose(p) for p in players)
    competing=any(p.get('valid') and not retrieval_pose(p) and
        ((p.get('overhead',False) and p.get('arm_speed',0)>settings.low_hand_arm_threshold
          and p.get('foot_speed',0)>.8) or
         (p.get('spread',0)>.3 and p.get('foot_speed',0)>1.2)) for p in players)
    return retrieving and not competing


def confirmed_retrieval_spans(features,settings):
    if not features:return []
    times=[q['t'] for q in features]
    dt=float(np.median(np.diff(times))) if len(times)>1 else 0.
    spans=[];run=[]
    for q in list(features)+[None]:
        if q is not None and retrieval_frame(q,settings):run.append(q['t'])
        else:
            if run and run[-1]-run[0]+dt+1e-9>=settings.pickup_confirm:
                spans.append((run[0],run[-1]))
            run=[]
    return spans

@dataclass
class TemporalSettings:
    smooth_seconds: float = .4
    action_gap: float = 3.0
    rally_stitch_gap: float = 8.0
    rally_stitch_evidence_gap: float = 3.0
    ready_lookback: float = 2.5
    serve_backtrack: float = .3  # 固定机位：从首次击球动作前回退；准备时间由 pre-roll 保留
    reaction_backtrack: float = .6
    terminal_flight: float = .8
    end_confirm: float = 3.0
    arm_threshold: float = .65
    hand_height: float = .08
    broadcast_ready_lookback: float = 6.0
    ready_confirm: float = .6
    relaxed_confirm: float = .6
    pickup_confirm: float = .35
    pickup_recovery: float = 1.0
    low_hand_arm_threshold: float = 1.0
    tail_guard_max: float = 1.5
    tail_guard_confirm: float = .5
    occlusion_bridge: float = 2.0

    def validate(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'invalid temporal setting {name}')
        if self.action_gap <= 0 or self.end_confirm < self.action_gap:
            raise ValueError('end_confirm must be >= action_gap > 0')


def segment_actions(features, pre_roll=3., t_limit=8., settings=None,
                    scene_start=0., scene_end=None, broadcast=False, terminal_transition=False,
                    pre_roll_floor=None, stop_times=()):
    """Bridge isolated occlusions; reject pickup and incomplete end-of-window play.

    Strong evidence combines wrist motion with raised hands, or fast wide lunges.
    Stationary preparation before the first action estimates the start; a bounded
    reaction allowance accounts for the untracked opponent's serve. The last
    action plus flight allowance estimates the end ONLY after subsequent frames
    contain no further action. These allowances are configurable heuristics.
    """
    c=settings or TemporalSettings();c.validate()
    if not features:return []
    t=np.array([q['t'] for q in features],float)
    if len(t)<2:return []
    if np.any(np.diff(t)<=0):raise ValueError('pose timestamps must increase')
    dt=float(np.median(np.diff(t)))
    radius=max(1,round(c.smooth_seconds/2/dt))
    values=lambda k:np.array([q.get(k,0) for q in features],float)
    def smooth(a):
        return np.array([np.median(a[max(0,i-radius):i+radius+1]) for i in range(len(a))])
    arm,foot,spread=[smooth(values(k)) for k in ('arm_speed','foot_speed','spread')]
    body=smooth(values('body_speed'))
    heights=[];low_hands=[];leaning_pickup=[];observable=[]
    for q in features:
        if not q.get('valid'):
            heights.append(0.);low_hands.append(False);leaning_pickup.append(False);observable.append(False);continue
        pts=np.asarray(q['keypoints']);height=max(q['box'][3]-q['box'][1],.01)
        hips=[p[1] for p in pts[11:13] if p[2]>.25]
        hip_x=[p[0] for p in pts[11:13] if p[2]>.25]
        shoulder_pts=[p for p in pts[5:7] if p[2]>.25]
        shoulders=[p[0] for p in shoulder_pts]
        hands=[p[1] for p in pts[9:11] if p[2]>.3]
        observable.append(bool(hips and hands))
        heights.append((np.mean(hips)-min(hands))/height if hips and hands else 0.)
        low_hands.append(bool(hips and hands and (max(hands)-np.mean(hips))/height>.22))
        leaning_pickup.append(bool(hips and hip_x and shoulders and hands
            and (np.mean(hips)-np.mean([p[1] for p in shoulder_pts]))/height<.25
            and abs(np.mean(shoulders)-np.mean(hip_x))/height>.09
            and (max(hands)-np.mean(hips))/height>0))
    hand=smooth(np.array(heights))
    valid=(values('valid')>0)&np.array(observable)
    context_complete=np.array([q.get('context_complete',True) for q in features],bool)
    group_rest=np.array([q.get('group_rest',True) for q in features],bool)
    strong=((((arm>c.arm_threshold)&(hand>.18)) |
             ((arm>c.low_hand_arm_threshold)&(hand>c.hand_height))) |
            ((spread>.25)&(foot>.8)&(arm>.9))) & valid
    # A carrying hand can move quickly while walking. Treat sustained shuttle
    # contact during locomotion as positive carry evidence, not a stroke.
    overhead=np.array([q.get('overhead',False) for q in features],bool)
    held=values('ball_held')>0
    held_radius=max(1,round(.25/dt))
    held_near=np.array([np.any(held[max(0,k-held_radius):min(len(t),k+held_radius+1)])
                        for k in range(len(t))],bool)
    carrying_walk=held_near&(foot>.3)&(body>.2)&(spread<.25)
    strong[carrying_walk]=False
    # Overhead is a pose, not a stroke. It needs motion evidence; a planted
    # arm swing or a wide lunge can also support a candidate start.
    overhead_swing=overhead&(arm>c.arm_threshold)&(hand>c.hand_height)
    planted_swing=(foot<.65)&(body<.3)&(arm>c.arm_threshold)&(hand>.18)
    wide_lunge=(spread>.25)&(foot>.8)&(arm>.9)
    stroke_support=overhead_swing|planted_swing|wide_lunge
    walk_like=(body>.2)&(foot>.3)&(spread<.25)&(hand>.08)
    clear_stroke=planted_swing|wide_lunge|(overhead_swing & ~walk_like)
    bent=values('bent')>0
    pickup_pose=valid & (bent | np.array(low_hands) | np.array(leaning_pickup)) & (((foot<.8) & (body<.65) & (spread<.3)) |
                 (bent & (spread<.3) & (body<1.5) & (foot<2.)))
    # Inspect every identity: fusion may select the standing player's wrist motion
    # while another player is retrieving. Missing identities are not pickup evidence.
    for k,q in enumerate(features):
        if retrieval_frame(q,c):pickup_pose[k]=True
    pickup=np.zeros(len(t),dtype=bool);pickup_spans=[]
    run=[]
    for k in range(len(t)+1):
        if k<len(t) and pickup_pose[k]:run.append(k)
        else:
            if run and t[run[-1]]-t[run[0]]+dt+1e-9>=c.pickup_confirm:
                pickup[run]=True;pickup_spans.append((float(t[run[0]]),float(t[run[-1]])))
            run=[]
    for begin,end in pickup_spans:
        recovery=(t>=begin-.3)&(t<=end+c.pickup_recovery)
        athletic=(foot>.8)&(spread>.2)&(body>.45)
        strong[recovery & ~athletic]=False
    strong[pickup]=False  # 已持续确认的捡球是回合屏障，不能被融合后的摆臂重新覆盖
    # Keep the old local protection for isolated uncertain bending detections.
    pickup_radius=max(1,round(.5/dt))
    for i in range(len(t)):
        if np.any(bent[max(0,i-pickup_radius):i+pickup_radius+1]) and foot[i]<.8:
            strong[i]=False
    # Sparse sampling makes single pose spikes disproportionately influential.
    # Retain isolated high-confidence overhead/fast-foot evidence, not wrist jitter.
    original=strong.copy();run=[]
    for k in range(len(t)+1):
        if k<len(t) and original[k]:run.append(k)
        else:
            if len(run)==1:
                z=run[0]
                if not features[z].get('overhead',False) and foot[z]<.9 and arm[z]<=c.low_hand_arm_threshold:strong[z]=False
            run=[]
    # Rapid court movement can continue an established rally even if a sampled
    # frame misses the wrist swing. It cannot start a new rally by itself.
    gap_allowance=2*float(np.percentile(np.diff(t),95))
    last_play=None
    for k in range(len(t)):
        if pickup[k]:last_play=None
        if strong[k]:last_play=float(t[k])
        elif (last_play is not None and t[k]-last_play<=c.action_gap+gap_allowance and valid[k]
              and foot[k]>.9 and body[k]>.45 and hand[k]>.08 and not pickup[k]):
            strong[k]=True  # 脚步只能短暂接续最近击球，不能刷新击球时钟
    # A long absence is not a rally bridge. On return, walking back to court
    # must not extend the previous rally; wait for rest or clear athletic action.
    missing_since=None;recovering=False;settled=0.;last_box=None
    exit_barriers=np.zeros(len(t),dtype=bool)
    for k in range(len(t)):
        if not valid[k]:
            if missing_since is None:missing_since=float(t[k])
            at_edge=last_box is not None and (last_box[0]<.02 or last_box[2]>.98)
            if at_edge and t[k]-missing_since+dt>c.occlusion_bridge:recovering=True;exit_barriers[k]=True
            continue
        missing_since=None;last_box=features[k]['box']
        if recovering:
            athletic=(features[k].get('overhead',False) and arm[k]>c.low_hand_arm_threshold
                      and foot[k]>.9) or (spread[k]>.3 and foot[k]>1.2 and hand[k]>.18)
            quiet=arm[k]<.35 and foot[k]<.3 and body[k]<.2
            settled=settled+dt if quiet else 0.
            if athletic or settled+1e-9>=c.ready_confirm:
                recovering=False;settled=0.
            else:strong[k]=False
    groups=[]
    for i in np.flatnonzero(strong):
        split=not groups
        if groups:
            previous=groups[-1][-1]
            unseen=float(np.count_nonzero(~valid[previous:i]))*dt
            rest_run=0.;clear_rest=False
            for z in range(previous+1,i):
                if valid[z] and group_rest[z] and arm[z]<.35 and foot[z]<.3 and body[z]<.2:
                    rest_run+=dt
                    if rest_run+1e-9>=c.ready_confirm:clear_rest=True
                else:rest_run=0.
            allowance=c.action_gap+gap_allowance+(0 if clear_rest else min(unseen,c.occlusion_bridge))
            # Only missing observations earn extra time; real rest/pickup never does.
            split=(t[i]-t[previous]>allowance or np.any(pickup[previous:i])
                   or np.any(exit_barriers[previous:i]) or any(t[previous]<v<=t[i] for v in stop_times))
        if split:groups.append([])
        groups[-1].append(i)
    records=[]
    end_of_data=min(float(t[-1]),scene_end) if scene_end is not None else float(t[-1])
    for group in groups:
        # A walking-looking opening followed by a distinct stroke belongs to
        # the preparation for that stroke, not to the preceding rally. A group
        # consisting only of ambiguous raised-hand walking is rejected below.
        if walk_like[group[0]]:
            first_clear=next((n for n,z in enumerate(group) if clear_stroke[z]),None)
            if first_clear is not None and first_clear>0:
                group=group[first_clear:]
        previous_stop=max((v for v in stop_times if v<=t[group[0]]),default=None)
        if previous_stop is not None:
            # After a confirmed grounded ball, retrieval cannot immediately seed
            # another rally. Wait for a clear overhead stroke or planted serve.
            while group:
                z=group[0]
                if arm[z]>c.low_hand_arm_threshold and ((overhead[z] and body[z]<.65 and foot[z]<1.5) or (foot[z]<.3 and body[z]<.2)):break
                group=group[1:]
            if not group:continue
        # Confirmed held-ball walking cannot seed a rally. Ambiguous raised-hand
        # walking is retained with a review flag; ordinary footwork remains valid.
        uncertain_head=False
        while group:
            z=group[0]
            if carrying_walk[z]:
                group=group[1:]
                continue
            walking=(body[z]>.2 and foot[z]>.3 and spread[z]<.25
                     and hand[z]<.35 and arm[z]<c.low_hand_arm_threshold
                     and not features[z].get('overhead',False))
            if not walking and stroke_support[z]:break
            if walk_like[z]:
                # Raised-hand walking alone must not create a playable rally.
                if not any(clear_stroke[v] for v in group):group=[];break
                group=group[1:]
                continue
            group=group[1:]
        if not group:continue
        i,j=group[0],group[-1]
        if len(group)*dt < .2:continue  # Ignore single-frame pose glitches.
        quiet=np.flatnonzero((t>=max(scene_start,t[i]-c.ready_lookback)) &
                            (t<t[i]) & valid & (arm<.35) & (foot<.3))
        onset=float(t[quiet[-1]]) if len(quiet) else float(t[i])
        preparation=False
        if broadcast:
            # A short pause between returns is not a fresh serve: require sustained rest.
            ready=(valid & group_rest & (arm<.35) & (foot<.3) & (body<.2) &
                   (t>=max(scene_start,t[i]-c.broadcast_ready_lookback)) & (t<t[i]))
            run_start=None;last_ready=None
            for k in np.flatnonzero(t<t[i]):
                if ready[k]:
                    if run_start is None:run_start=k
                    if t[k]-t[run_start]+dt+1e-9>=c.ready_confirm:last_ready=k
                else:run_start=None
            if last_ready is not None:
                onset=float(t[last_ready]);preparation=True
        # In a fixed view, another player's quiet pose is not serve preparation.
        # Keep pre-roll explicit instead of stacking an unbounded quiet lookup.
        release=max(scene_start,onset-c.reaction_backtrack) if broadcast else max(scene_start,float(t[i])-c.serve_backtrack)
        floor=scene_start if pre_roll_floor is None else pre_roll_floor
        start=max(floor,release-pre_roll)
        if previous_stop is not None:start=max(start,previous_stop+.5)
        previous_pickups=[end for begin,end in pickup_spans if end<float(t[i])]
        if previous_pickups:
            # Pre-roll must not reintroduce the previous rally's retrieval.
            start=max(start,min(float(t[i]),previous_pickups[-1]+c.pickup_recovery))
            if float(t[i])-previous_pickups[-1]<=6.:
                start=max(start,release)
        uncertain_head=uncertain_head or bool(walk_like[i] and not stroke_support[i])
        r=dict(release=release,start=start,first_action=float(t[i]),last_action=float(t[j]),
               boundary_method='player_action_context_estimate',
               release_observed=False,landing_observed=False,needs_review=True,
               start_confidence='uncertain' if uncertain_head else 'supported',
               start_review_reason='walking_with_raised_hand' if uncertain_head else None,
               evidence_frames=len(group),pre_roll_available=release-start,
               boundary_uncertainty="not_calibrated",preparation_observed=preparation)
        stop=next((v for v in stop_times if v>t[j]),None)
        if stop is not None:r['ground_stop_time']=stop
        # A detection gap, EOF, or a camera cut is never direct landing evidence.
        confirmation_window=c.end_confirm+c.occlusion_bridge
        after=(t>t[j]) & (t<=t[j]+confirmation_window)
        observed_after=float(np.count_nonzero(valid & context_complete & after))*dt
        normal_confirmation=end_of_data-t[j]>=c.end_confirm and observed_after>=.5
        relaxed=(valid & group_rest & (arm<c.arm_threshold) & (hand<c.hand_height) &
                 (spread<.18) & ~bent & (t>t[j]))
        relaxed_run=0.;relaxed_until=None
        for k in range(j+1,len(t)):
            if relaxed[k]:
                relaxed_run+=dt
                if relaxed_run+1e-9>=c.relaxed_confirm:relaxed_until=float(t[k])
            else:relaxed_run=0.
        transition_confirmation=(broadcast and terminal_transition and relaxed_until is not None
            and end_of_data-relaxed_until<=.3 and t[j]+c.terminal_flight<=end_of_data)
        retrieval=next((begin for begin,end in pickup_spans if begin>t[j]),None)
        retrieval_confirmation=retrieval is not None and retrieval-t[j]<=c.end_confirm
        if retrieval is not None:r['retrieval_start']=retrieval
        if not (normal_confirmation or transition_confirmation or retrieval_confirmation):
            r.update(status='incomplete',observed_until=end_of_data,
                     reason='insufficient_post_action_context')
        else:
            end=min(float(t[j]+c.terminal_flight),end_of_data)
            if retrieval is not None:end=min(end,retrieval)
            r['end_confirmation']=('sustained_retrieval_pose' if retrieval_confirmation else
                                   'relaxed_player_then_non_play_view' if transition_confirmation
                                   else 'subsequent_no_action_context')
            duration=end-start
            r.update(landing=end,end=end,duration=duration,
                     evidence='raised_wrist_motion_or_lunge; '+r['end_confirmation'])
            from shuttle_endings import filter_rally_duration
            filter_rally_duration(r,t_limit,c.terminal_flight)
            r['near_duration_threshold']=abs(r['qualifying_duration']-t_limit)<=1.
        records.append(r)
    return records


def merge_action_proposals(proposals,features,settings=None,scene_start=0.):
    """Join action fragments only across positively supported ongoing play.

    A confirmed retrieval, grounded shuttle stop, or sustained all-player ready
    pose is a hard boundary. The shuttle resolver still determines the final end.
    """
    c=settings or TemporalSettings();c.validate()
    ordered=sorted((dict(q) for q in proposals),key=lambda q:q.get('first_action',q['release']))
    if not ordered:return []
    times=np.array([q['t'] for q in features],float) if features else np.array([],float)
    dt=float(np.median(np.diff(times))) if len(times)>1 else 1/12
    retrieval_spans=confirmed_retrieval_spans(features,c)

    def pickup_between(begin,end):
        return any(left<end and right>begin for left,right in retrieval_spans)

    def settled_between(begin,end):
        if end<=begin or not len(times):return False
        run_start=None
        for q in features:
            if not begin<q['t']<end:continue
            settled=bool(q.get('valid') and q.get('context_complete') and q.get('group_rest'))
            if settled:
                if run_start is None:run_start=q['t']
                if q['t']-run_start+dt+1e-9>=c.ready_confirm:return True
            else:run_start=None
        return False

    def action_events(begin,end):
        events=[]
        for q in features:
            if not begin<q['t']<end or not q.get('valid') or q.get('ball_held',False):continue
            players=q.get('players') or [q]
            for p in players:
                if not p.get('valid') or p.get('bent',False) or retrieval_pose(p):continue
                arm=p.get('arm_speed',0);foot=p.get('foot_speed',0)
                body=p.get('body_speed',0);spread=p.get('spread',0)
                walk_like=body>.2 and foot>.3 and spread<.25
                overhead_pose=(p.get('overhead',False) and arm>c.arm_threshold)
                lunge=spread>.25 and foot>.8 and arm>.9
                if lunge:
                    events.append((float(q['t']),p.get('player_id'), 'lunge', True))
                elif overhead_pose:
                    events.append((float(q['t']),p.get('player_id'), 'overhead', not walk_like))
                elif arm>1.5 and body<=.25 and foot<=.6 and spread<.25:
                    # A compact, fast arm swing can be a serve even when the
                    # lower arm path never reaches the overhead-pose gate.
                    events.append((float(q['t']),p.get('player_id'), 'serve_swing', True))
        return events

    def action_cues(begin,end):
        return sorted(set(t for t,_,kind,strong in action_events(begin,end)
                          if strong and kind!='serve_swing'))

    def has_serve_burst(events):
        by_player={}
        for t,player,kind,strong in events:
            if kind=='serve_swing' and strong and player is not None:
                by_player.setdefault(player,[]).append(float(t))
        return any(any(.04<=right-left<=.35
                       for left,right in zip(sorted(set(times)),sorted(set(times))[1:]))
                   for times in by_player.values())

    resolved_keys=('end','landing','duration','status','end_confirmation','end_confirmed_at',
                   'evidence','reason','observed_until','end_uncertain','pose_tail_capped',
                   'qualifying_duration')

    def clear_resolution(proposal):
        for key in resolved_keys:proposal.pop(key,None)

    def trim_walk_tail(proposal):
        first=float(proposal.get('first_action',proposal['release']))
        last=float(proposal.get('last_action',first))
        if last<=first:return
        cues=action_cues(first,last+dt)
        if cues and last-cues[-1]>c.action_gap:
            proposal['last_action']=cues[-1]
            clear_resolution(proposal)

    def backtrack_short_head(proposal,previous):
        first=float(proposal.get('first_action',proposal['release']))
        last=float(proposal.get('last_action',first))
        if last-first>2.5:return
        pre_roll=float(proposal.get('pre_roll_available',proposal['release']-proposal.get('start',proposal['release'])))
        if pre_roll<.5:return
        begin=first-c.rally_stitch_gap
        barriers=[]
        barriers.extend(end+c.pickup_recovery for begin,end in retrieval_spans if end<first)
        if previous is not None:
            for key in ('retrieval_start','ground_stop_time'):
                if previous.get(key) is not None:
                    boundary=float(previous[key])
                    if boundary<=first:barriers.append(boundary)
        floor=max([scene_start]+barriers)
        events=[event for event in action_events(max(begin,floor),last+dt)
                if event[2]!='serve_swing']
        # Keep only the final cue cluster close to the short proposal. Walking
        # overheads may extend a real multi-player stroke cluster, but cannot
        # rescue a candidate without at least two independent strong cues.
        events.sort(key=lambda row:row[0])
        clusters=[]
        for event in events:
            if not clusters or event[0]-clusters[-1][-1][0]>2.0:
                clusters.append([event])
            else:
                clusters[-1].append(event)
        if not clusters:return
        cluster=clusters[-1]
        strong=[event for event in cluster if event[3]]
        strong_times=sorted(set(event[0] for event in strong))
        if (len(strong_times)<2 or strong_times[-1]-strong_times[0]<.3
                or max(event[0] for event in cluster)<first-.75):return
        strong_players={event[1] for event in strong if event[1] is not None}
        if not any(event[2]=='lunge' for event in strong) and len(strong_players)<2:return
        cue_start=min(event[0] for event in cluster)
        if any(cue_start<boundary<=first for boundary in barriers):return
        release=max(floor,min(float(proposal['release']),cue_start-max(c.serve_backtrack,.8)))
        start=max(floor,min(float(proposal.get('start',proposal['release'])),release-pre_roll))
        if cue_start>=first:return
        proposal.update(first_action=cue_start,release=release,start=start,
                        pre_roll_available=max(0.,release-start),
                        evidence_frames=int(proposal.get('evidence_frames',0))+len(strong_times),
                        recovered_action_head=True)
        clear_resolution(proposal)

    def extend_last_action(proposal,limit):
        last=float(proposal.get('last_action',proposal.get('first_action',proposal['release'])))
        end=min(float(limit),last+c.action_gap)
        for begin,finish in retrieval_spans:
            if finish>=last:end=min(end,max(last,begin))
        retrieval=proposal.get('retrieval_start')
        if retrieval is not None and float(retrieval)>last:end=min(end,float(retrieval))
        cues=action_cues(last,end)
        if len(cues)>=2 and cues[-1]-cues[0]>=.1:
            proposal['last_action']=cues[-1]
            clear_resolution(proposal)

    # Trim pose jitter and post-rally walking before stitching proposals.
    for proposal in ordered:trim_walk_tail(proposal)
    # A short proposal may begin late even when the preceding frames contain a
    # continuous, multi-player stroke sequence. Recover that observed lead-in.
    for index,proposal in enumerate(ordered):
        backtrack_short_head(proposal,ordered[index-1] if index else None)
    ordered.sort(key=lambda q:q.get('first_action',q['release']))
    merged=[]
    for current in ordered:
        if not merged:
            merged.append(current);continue
        previous=merged[-1]
        current_first=float(current.get('first_action',current['release']))
        extend_last_action(previous,current_first)
        previous_last=float(previous.get('last_action',previous.get('first_action',previous['release'])))
        retrieval=previous.get('retrieval_start')
        stop=previous.get('ground_stop_time')
        barrier=((retrieval is not None and previous_last<retrieval<=current_first) or
                 (stop is not None and previous_last<stop<=current_first) or
                 pickup_between(previous_last,current_first) or
                 settled_between(previous_last,current_first))
        gap=current_first-previous_last
        previous_first=float(previous.get('first_action',previous['release']))
        current_last=float(current.get('last_action',current_first))
        serve_lead_in=(previous_last-previous_first<=.75 and
            current_last-current_first>=1.5 and
            has_serve_burst(action_events(previous_first,current_first)))
        # Small sampling gaps and a short observed serve flight can be bridged.
        # Larger gaps need continuous action support; elapsed time alone is not play.
        cue_times=action_cues(previous_last,current_first)
        supported_gap=(len(cue_times)>=2 and
            max(np.diff([previous_last]+cue_times+[current_first]))<=c.rally_stitch_evidence_gap)
        supported_serve=serve_lead_in and gap<=c.rally_stitch_evidence_gap
        unsupported_gap=(gap>max(c.terminal_flight+.2,2*dt) and not supported_gap and not supported_serve)
        if gap>c.rally_stitch_gap or barrier or unsupported_gap:
            merged.append(current);continue

        latest=float(current.get('last_action',current_first))
        current_last=float(current.get('last_action',current_first))
        previous_start=(max(scene_start,float(previous['release'])-.5) if serve_lead_in else
                        float(previous.get('start',previous['release'])))
        previous.update(start=min(previous_start,
                                  float(current.get('start',current['release']))),
                        release=min(float(previous['release']),float(current['release'])),
                        first_action=min(float(previous.get('first_action',previous['release'])),current_first),
                        last_action=max(previous_last,latest),
                        evidence_frames=int(previous.get('evidence_frames',0))+int(current.get('evidence_frames',0)),
                        recovered_action_head=bool(previous.get('recovered_action_head') or
                                                    current.get('recovered_action_head')))
        clear_resolution(previous)
        future_retrievals=[float(q['retrieval_start']) for q in (previous,current)
                           if q.get('retrieval_start') is not None and float(q['retrieval_start'])>previous['last_action']]
        if future_retrievals:previous['retrieval_start']=min(future_retrievals)
        else:previous.pop('retrieval_start',None)
        future_stops=[float(q['ground_stop_time']) for q in (previous,current)
                      if q.get('ground_stop_time') is not None and float(q['ground_stop_time'])>previous['last_action']]
        if future_stops:previous['ground_stop_time']=min(future_stops)
        else:previous.pop('ground_stop_time',None)
    if merged:
        extend_last_action(merged[-1],float('inf'))
    for proposal in merged:
        proposal['release']=max(scene_start,float(proposal['release']))
        proposal['start']=max(scene_start,float(proposal.get('start',proposal['release'])))
    return merged


def protect_active_tails(records, features, fps, window_end, scene_cuts=(),
                         video_duration=None, max_extension=1.5, rest_confirm=.5):
    """Recheck planned cuts. Extend exports only with observed ongoing-play cues.

    This is bounded export protection, not proof of shuttle landing. Duration
    filtering and detection timestamps remain unchanged. A capped unresolved tail
    stays explicitly marked for review.
    """
    if not features or max_extension<=0:return
    t=np.array([q['t'] for q in features],float)
    dt=float(np.median(np.diff(t))) if len(t)>1 else 1/fps
    active=[];relaxed=[];valid=[]
    for q in features:
        good=bool(q.get('valid'));valid.append(good)
        if not good:
            active.append(False);relaxed.append(False);continue
        pts=np.asarray(q['keypoints']);height=max(q['box'][3]-q['box'][1],.01)
        hips=[p[1] for p in pts[11:13] if p[2]>.25]
        hands=[p[1] for p in pts[9:11] if p[2]>.3]
        if not hips or not hands:
            active.append(False);relaxed.append(False);continue
        hand=(np.mean(hips)-min(hands))/height
        foot=q.get('foot_speed',0.);body=q.get('body_speed',0.)
        spread=q.get('spread',0.);arm=q.get('arm_speed',0.)
        bent=q.get('bent',False)
        ready=(hand>.12 and spread>.12) or (hand>.18 and arm>.3)
        moving=(foot>.8 and body>.4 and hand>.06)
        active.append(bool((ready or moving) and not bent))
        relaxed.append(bool(q.get('group_rest',True) and hand<.08 and arm<.65 and spread<.18 and not bent))
    active=np.array(active);relaxed=np.array(relaxed);valid=np.array(valid)
    terminal_endings={'shuttle_stationary','outward_exit_missing','new_ground_stop_with_quiet_players',
                      'pose_confirmed_retrieval','pose_tail_cap_without_confirmed_shuttle_end'}
    for r in records:
        if r['status']!='keep':continue
        if r.get('pose_tail_capped') or r.get('end_confirmation') in terminal_endings:continue
        baseline=r['planned_export_end']
        r['tail_guard_extension']=0.
        limit=min([window_end,baseline+max_extension]+[x for x in scene_cuts if x>r['end']])
        if video_duration is not None and video_duration>0:limit=min(limit,video_duration)
        if r.get('retrieval_start') is not None:limit=min(limit,r['retrieval_start'])
        later=[q['release'] for q in records if q.get('release',-1)>baseline]
        if later:limit=min(limit,min(later))
        if limit<=baseline:continue
        # At least two successive sampled cues around the cut, not a single spike.
        near=np.flatnonzero((t>=baseline-max(.4,2*dt))&(t<=baseline+.25))
        run=0;trigger=False
        for k in near:
            run=run+1 if active[k] else 0
            if run>=2 and run*dt>=.25:trigger=True
        if not trigger:continue
        rest_since=None;confirmed=False;rest_start=None
        for k in np.flatnonzero((t>=baseline)&(t<=limit)):
            if relaxed[k]:
                if rest_since is None:rest_since=float(t[k])
                if t[k]-rest_since+dt+1e-9>=rest_confirm:
                    rest_start=rest_since;confirmed=True;break
            else:rest_since=None
        available=min(limit,float(t[-1])+1/fps)
        end=min(available,max(baseline,rest_start+.2)) if confirmed else available
        if end<=baseline:continue
        r.update(planned_export_end=end,export_duration=end-r['start'],
                 tail_guard_extension=end-baseline,
                 tail_guard_reason='ongoing_play_at_planned_cut',
                 tail_guard_rest_confirmed=confirmed,
                 tail_guard_unresolved=not confirmed,
                 needs_review=True)
