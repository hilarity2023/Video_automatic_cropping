"""Resolve pose-proposed starts using observed shuttle stopping or outward exit."""
import math


def in_ground(point,polygon):
    # Image-space ground mask supplied by the user's court polygon.
    import cv2
    import numpy as np
    return cv2.pointPolygonTest(np.asarray(polygon,np.float32),(float(point.x),float(point.y)),False)>=0


def pose_active(feature,c):
    """Conservative any-player cue used to veto a premature shuttle stop."""
    for p in feature.get('players',[feature]):
        if not p.get('valid'):continue
        pts=p.get('keypoints');hand_height=0.
        if pts is not None:
            hips=[v[1] for v in pts[11:13] if v[2]>.25]
            hands=[v[1] for v in pts[9:11] if v[2]>.3]
            box=p.get('box',[0,0,0,.01]);height=max(box[3]-box[1],.01)
            if hips and hands:hand_height=(sum(hips)/len(hips)-min(hands))/height
        arm=p.get('arm_speed',0);foot=p.get('foot_speed',0)
        if ((arm>.65 and hand_height>.08) or
                (p.get('spread',0)>.25 and foot>.8)):
            return True
    return False


def filter_rally_duration(record, t_limit, terminal_flight=.8):
    """Filter playing time, never pre-roll, confirmation wait, or export padding."""
    if 'end' not in record:return
    end=record['end']
    reason=record.get('end_confirmation','')
    if reason=='shuttle_stationary' and record.get('stop_started_at') is not None:
        end=min(end,record['stop_started_at'])
    if reason in {'pose_confirmed_retrieval','pose_tail_cap_without_confirmed_shuttle_end',
                  'sustained_retrieval_pose','subsequent_no_action_context',
                  'relaxed_player_then_non_play_view'} and 'last_action' in record:
        end=min(end,record['last_action']+terminal_flight)
    record['qualifying_duration']=max(0.,end-record['release'])
    record['status']='keep' if record['qualifying_duration']>t_limit+1e-9 else 'too_short'


def resolve_shuttle_endings(proposals, samples, c, scene_end, allow_pickup_fallback=False,
                            terminal_flight=.8):
    results=[];consumed_until=-math.inf
    ordered=sorted(proposals,key=lambda q:q.get('release',math.inf))
    for index,proposal in enumerate(ordered):
        release=proposal['release']
        if release<=consumed_until:continue
        start=proposal.get('start',max(0.,release-c.pre_roll))
        search_end=min(scene_end,proposal.get('last_action',proposal.get('end',proposal.get('observed_until',release)))+c.shuttle_search_tail)
        if index+1<len(ordered):
            following=ordered[index+1]
            boundary=following.get('first_action',following['release'])
            search_end=min(search_end,boundary)
        retrieval=proposal.get('retrieval_start')
        if allow_pickup_fallback and retrieval is not None and retrieval>=proposal.get('last_action',release):
            search_end=min(search_end,retrieval+c.shuttle_stop_seconds+1e-6)
        r={k:v for k,v in proposal.items() if k not in {
            'end','landing','duration','status','end_confirmation','evidence',
            'reason','observed_until','planned_export_end','export_duration','qualifying_duration'}}
        r.update(start=start,boundary_method='player_start_shuttle_end',landing_observed=False,needs_review=True)
        last=None;previous=None;anchor=None;moving=False;rest=None;rest_time=None;end=None;reason=None
        rest_ground=None
        for row in samples:
            t=row['t']
            if t<release:continue
            if t>=search_end:break
            balls=row['balls']
            if last is None:
                acquisition=row.get('acquisition_balls',balls)
                near=[b for b in acquisition if any(math.hypot(b.x-x,b.y-y)<=c.hand_radius*2 for x,y in row['wrists'])]
                # Wrist visibility is not required: far-side hits often hide wrists.
                candidates=near or acquisition
                if not candidates:continue
                last=max(candidates,key=lambda b:b.conf);anchor=last
                continue
            gap=t-last.t
            candidates=[b for b in balls if math.hypot(b.x-last.x,b.y-last.y)<=c.max_speed*gap+.01]
            point=min(candidates,key=lambda b:math.hypot(b.x-last.x,b.y-last.y)-b.conf*.003) if candidates else None
            if point is None:
                rest=None;rest_time=None
                outward=False
                if previous is not None:
                    dx=last.x-previous.x;dy=last.y-previous.y;m=c.shuttle_exit_margin
                    outward=((last.x<=m and dx<-.003) or (last.x>=1-m and dx>.003) or
                             (last.y<=m and dy<-.003) or (last.y>=1-m and dy>.003))
                exit_threshold=c.shuttle_exit_seconds
                if moving and outward and gap>exit_threshold:
                    if 'last_action' in proposal and t<proposal['last_action']:
                        last=None;previous=None;anchor=None;moving=False
                        continue
                    if 'last_action' in proposal and t+1e-9<proposal['last_action']+c.min_flight:continue
                    end=t;reason='outward_exit_missing';break
                if gap>max(c.lost_timeout,exit_threshold):
                    # Interior loss is occlusion/unknown, not evidence of leaving frame.
                    last=None;previous=None;anchor=None;moving=False
                continue
            if gap>c.max_gap:rest=None;rest_time=None
            if math.hypot(point.x-anchor.x,point.y-anchor.y)>=c.depart_distance:moving=True
            ground=in_ground(point,c.ground)
            if not moving and t-anchor.t>=c.static_seconds:
                # A persistent static candidate is a distractor, not the rally ball.
                last=None;previous=None;anchor=None;rest=None;rest_time=None
                continue
            if moving:
                stop_threshold=c.shuttle_stop_seconds if ground else c.air_shuttle_stop_seconds
                # Crossing the polygon boundary starts a new region-specific timer.
                if rest is None or rest_ground!=ground or math.hypot(point.x-rest.x,point.y-rest.y)>c.still_radius:
                    rest=point;rest_time=t;rest_ground=ground
                elif t-rest_time+1e-9>=stop_threshold:
                    if 'last_action' in proposal and t<proposal['last_action']:
                        # A stationary distractor during subsequent confirmed play
                        # cannot end this rally. Reacquire instead of truncating it.
                        last=None;previous=None;anchor=None;moving=False;rest=None;rest_time=None
                        continue
                    if 'last_action' in proposal and t+1e-9<proposal['last_action']+c.min_flight:
                        previous=last;last=point;continue
                    if not allow_pickup_fallback:
                        end=t;reason='shuttle_stationary';break
                    followup=getattr(c,'stop_followup_seconds',1.8)
                    future=[q for q in samples if t<q['t']<=t+followup]
                    stationary_hits=sum(any(math.hypot(b.x-point.x,b.y-point.y)<=c.still_radius*2
                                            for b in q['balls']) for q in future)
                    moved_after=any(any(c.still_radius*2<math.hypot(b.x-point.x,b.y-point.y)
                                        <=c.max_speed*(q['t']-t)+.01 for b in q['balls']) for q in future)
                    continued_play=any(q.get('play_active',False) for q in future)
                    enough_context=(len(future)>=max(5,round(followup*c.shuttle_fps*.6))
                                    and stationary_hits>=3)
                    if continued_play or moved_after:
                        # The ball may have stopped briefly between returns, but
                        # later play vetoes this candidate end.
                        last=None;previous=None;anchor=None;moving=False;rest=None;rest_time=None
                        continue
                    if not enough_context:
                        # A stop near EOF or a tracking gap is unresolved; let
                        # context mode retain the longer interval.
                        last=None;previous=None;anchor=None;moving=False;rest=None;rest_time=None
                        continue
                    end=t;reason='shuttle_stationary';break
            else:rest=None;rest_time=None;rest_ground=None
            previous=last;last=point
        ground_stop=proposal.get('ground_stop_time')
        if ground_stop is not None and release<ground_stop<search_end and (end is None or ground_stop<end):
            end=ground_stop;reason='new_ground_stop_with_quiet_players'
        retrieval=proposal.get('retrieval_start')
        if (end is None and allow_pickup_fallback and retrieval is not None
                and proposal.get('last_action',release)<=retrieval
                and retrieval<=search_end):
            # Sustained pose-confirmed retrieval is a bounded fallback when the
            # tiny shuttle cannot be tracked. It is not a claim of observed contact.
            end=retrieval;reason='pose_confirmed_retrieval'
        if (end is not None and allow_pickup_fallback and retrieval is not None
                and proposal.get('last_action',release)<=retrieval<end):
            end=retrieval;reason='pose_confirmed_retrieval'
        if end is None and allow_pickup_fallback:
            # If neither ball-stop nor retrieval pose was confirmed, bound the
            # tail after the final player action. Extending to a later rally or
            # scene end commonly includes the players retrieving the shuttle.
            fallback_limit=(min(scene_end,ordered[index+1].get('start',
                                   ordered[index+1].get('release',boundary)))
                            if index+1<len(ordered) else scene_end)
            fallback_end=min(fallback_limit,
                proposal.get('last_action',release)+c.context_fallback_tail+
                (1.0 if proposal.get('recovered_action_head') else 0.0))
            retrieval_gap=(retrieval-proposal.get('last_action',release)) if retrieval is not None else math.inf
            if c.shuttle_search_tail<retrieval_gap<=c.shuttle_search_tail+2.:
                # For a nearby but unconfirmed stop, spend at most one extra
                # second while keeping 1.5 s before the observed retrieval pose.
                safe_tail=min(c.context_fallback_tail+1.,retrieval-proposal.get('last_action',release)-1.5)
                fallback_end=min(fallback_limit,max(fallback_end,
                    proposal.get('last_action',release)+max(c.context_fallback_tail,safe_tail)))
            if fallback_end>release:
                end=fallback_end;reason='pose_tail_cap_without_confirmed_shuttle_end'
                r.update(end=end,duration=end-start,
                         end_confirmation=reason,evidence='no_confirmed_shuttle_stop_or_exit',
                         end_uncertain=True,needs_review=True,
                         observed_until=fallback_end,pose_tail_capped=True)
                consumed_until=end
            else:
                r.update(status='incomplete',observed_until=search_end,
                         reason='no_confirmed_shuttle_stop_or_exit_in_rally_window')
        elif end is None:
            r.update(status='incomplete',observed_until=search_end,reason='no_confirmed_shuttle_stop_or_exit_in_rally_window')
        else:
            consumed_until=end
            r.update(end=end,duration=end-start,
                     end_confirmation=reason,end_confirmed_at=end,evidence=reason,
                     stop_started_at=rest_time if reason=='shuttle_stationary' else None,
                     last_shuttle_seen=last.t if last else None)
        filter_rally_duration(r,c.t_limit,terminal_flight)
        results.append(r)
    return results


def write_merged_start_log(path,records,part_durations=None):
    # Part durations are measured after encoding, in the same order as concat.
    lines=[]
    merged_at=0.
    if part_durations is not None and len(part_durations)!=len(records):
        raise ValueError('Part duration count must match exported records')
    for i,r in enumerate(records):
        end=r.get('export_end',r.get('planned_export_end'))
        duration=part_durations[i] if part_durations is not None else end-r['start']
        merged_end=merged_at+duration
        lines.append(f"export={r['start']:.3f}-{end:.3f}s ; merged={merged_at:.3f}-{merged_end:.3f}s")
        merged_at=merged_end
    path.write_text(''.join(line+'\n' for line in lines),encoding='utf-8')


def ground_stop_windows(samples,c,features=None):
    """New persistent ground targets, excluding pre-existing clutter and held balls.

    A short missing observation is tolerated, but never counted as a detection.
    Fixed-camera evidence only; callers reset samples at scene boundaries.
    """
    windows=[];tracks=[]
    for i,row in enumerate(samples):
        t=row['t']
        tracks=[v for v in tracks if t-v['last']<=max(.6,c.max_gap)]
        for v in tracks:v['updated']=False
        for b in row['balls']:
            if not in_ground(b,c.ground):continue
            if any(math.hypot(b.x-x,b.y-y)<c.hand_radius*2 for x,y in row['wrists']):continue
            match=next((v for v in tracks if math.hypot(b.x-v['x'],b.y-v['y'])<=c.still_radius),None)
            if match is None:
                history=[q for q in samples[max(0,i-20):i] if t-3<=q['t']<t]
                if len(history)<4:continue
                occupied=sum(any(math.hypot(b.x-z.x,b.y-z.y)<=c.still_radius for z in q['balls']) for q in history)
                if occupied/len(history)>.25:continue
                match=dict(x=b.x,y=b.y,begin=t,last=t,hits=0,updated=False,confirmed=False)
                tracks.append(match)
            if match['updated']:continue  # duplicate boxes are not temporal evidence
            match['updated']=True;match['last']=t;match['hits']+=1
            if not match['confirmed'] and t-match['begin']>=1. and match['hits']>=4:
                end=match['begin']+c.shuttle_stop_seconds
                context=[q for q in (features or []) if end-.5<=q['t']<=end+.5]
                visible=[q for q in context if q.get('valid')]
                lookahead=getattr(c,'stop_followup_seconds',1.8)
                future=[q for q in (features or []) if end<q['t']<=end+lookahead]
                future_visible=[q for q in future if q.get('valid') and q.get('context_complete',True)]
                continued_play=any(pose_active(q,c) for q in future_visible)
                quiet_context=(features is None or (len(visible)>=4 and len(visible)>=.8*len(context)
                    and len(future)>=max(5,round(lookahead*6*.6))
                    and len(future_visible)>=.8*len(future)
                    and not continued_play
                    and sum(q.get('arm_speed',0)>1.2 or (q.get('overhead',False) and q.get('arm_speed',0)>.8) for q in visible)<=.3*len(visible)))
                if quiet_context and (not windows or end-windows[-1]>1.):windows.append(end)
                match['confirmed']=True
    return windows
