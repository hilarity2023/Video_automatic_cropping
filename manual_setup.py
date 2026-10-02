"""Interactive fixed-camera setup. Coordinates are normalized to the source frame."""
import math
import cv2
import numpy as np


def validate_annotation(value):
    court=value.get('court') or [];box=value.get('player_box') or [];ball=value.get('shuttle')
    def point(p):return len(p)==2 and all(math.isfinite(x) and 0<=x<=1 for x in p)
    if len(court)<3 or not all(point(p) for p in court):raise ValueError('球场至少需要三个有效顶点')
    contour=np.asarray(court,np.float32)
    if abs(cv2.contourArea(contour))<.001:raise ValueError('球场区域过小，请重新框选')
    # Reject crossed boundaries; vertices must follow the court perimeter.
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    for i in range(len(court)):
        a,b=court[i],court[(i+1)%len(court)]
        for j in range(i+1,len(court)):
            if j==i+1 or (i==0 and j==len(court)-1):continue
            c,d=court[j],court[(j+1)%len(court)]
            if cross(a,b,c)*cross(a,b,d)<=0 and cross(c,d,a)*cross(c,d,b)<=0:
                raise ValueError('球场边界不能交叉，请沿边界顺序点击')
    boxes=value.get('player_boxes')
    if boxes is not None:
        if not 2<=len(boxes)<=4:raise ValueError('请分别框选2–4名球员（单打2人，双打4人）')
    else:boxes=[box]  # Compatibility for previous single-player reports.
    for box in boxes:
        if len(box)!=4 or not all(math.isfinite(x) and 0<=x<=1 for x in box):raise ValueError('请框选球员')
        if box[2]-box[0]<.005 or box[3]-box[1]<.015:raise ValueError('球员框过小，请包含全身和脚')
        foot=((box[0]+box[2])/2,box[3])
        if cv2.pointPolygonTest(contour,foot,True)<-.03:raise ValueError('球员脚下应位于所选球场内，请检查球场或球员框')
    for i,a in enumerate(boxes):
        for b in boxes[i+1:]:
            overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            smaller=min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1]))
            if overlap/max(smaller,1e-9)>.8:raise ValueError('球员框重复或重叠过多，请分别框选每个人')
    if ball is not None and not point(ball):raise ValueError('羽毛球坐标无效')
    return value


def select_annotation(frame,t=0.,video_path=None):
    """Collect manual hints, or return None when automatic setup is selected."""
    import tkinter as tk
    from tkinter import messagebox
    from PIL import Image,ImageTk
    root=tk.Tk();root.title('羽毛球自动剪辑 — 人工初始化')
    h,w=frame.shape[:2]
    scale=min(1.,(root.winfo_screenwidth()-100)/w,(root.winfo_screenheight()-240)/h)
    dw,dh=max(1,round(w*scale)),max(1,round(h*scale))
    photo=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((dw,dh)))
    state={'step':0,'court':[],'player_boxes':[],'player_box':None,'shuttle':None,
           'drag':None,'result':None,'skipped':False}
    label=tk.StringVar();tk.Label(root,textvariable=label,font=('Microsoft YaHei',11),wraplength=dw).pack(padx=10,pady=8)
    canvas=tk.Canvas(root,width=dw,height=dh,highlightthickness=0);canvas.pack()
    hints=['① 沿球场地面边界依次点击顶点（至少3点），多边形内部视为地面、外部为空中，请贴合地面边界。右键撤销，完成后点“下一步”。',
           '② 逐个拖框选中2–4名球员（单打2人、双打4人），包含全身和脚。松开鼠标添加，右键撤销最后一个。',
           '③ 点击羽毛球中心；看不清或不在画面内时点“跳过羽毛球”。球检测始终启用，标注只帮助初始匹配。',
           '④ 检查：绿色=场地，橙色=各球员，红色=羽毛球。确认后开始分析。']
    def draw():
        canvas.delete('all');canvas.create_image(0,0,anchor='nw',image=photo)
        points=[(x*dw,y*dh) for x,y in state['court']]
        if len(points)>=2:canvas.create_line(*[v for p in points+[points[0]] for v in p],fill='#00ff80',width=2)
        for x,y in points:canvas.create_oval(x-4,y-4,x+4,y+4,fill='#00ff80')
        for number,box in enumerate(state['player_boxes']+([state['player_box']] if state['player_box'] else []),1):
            a,b,c,d=box;canvas.create_rectangle(a*dw,b*dh,c*dw,d*dh,outline='#ffb000',width=3)
            canvas.create_text(a*dw,b*dh,anchor='sw',text=f'球员 {number}',fill='#ffb000')
        if state['shuttle']:
            x,y=state['shuttle'];x*=dw;y*=dh
            canvas.create_oval(x-7,y-7,x+7,y+7,outline='red',width=2)
            canvas.create_line(x-12,y,x+12,y,fill='red');canvas.create_line(x,y-12,x,y+12,fill='red')
        label.set(f'原视频 {t:.2f} 秒（标注参考帧）\n'+hints[state['step']])
        next_button.config(text='确认并开始分析' if state['step']==3 else '下一步')
        skip_button.config(state='normal' if state['step']==2 else 'disabled')
    def xy(event):return [min(1,max(0,event.x/dw)),min(1,max(0,event.y/dh))]
    def click(event):
        p=xy(event)
        if state['step']==0:state['court'].append(p)
        elif state['step']==1 and len(state['player_boxes'])<4:state['drag']=p;state['player_box']=None
        elif state['step']==2:state['shuttle']=p
        draw()
    def drag(event):
        if state['step']==1 and state['drag'] is not None:
            a,b=state['drag'];c,d=xy(event);state['player_box']=[min(a,c),min(b,d),max(a,c),max(b,d)];draw()
    def release(event):
        drag(event)
        if state['step']==1 and state['drag'] is not None:
            box=state['player_box']
            if box and box[2]-box[0]>=.005 and box[3]-box[1]>=.015:state['player_boxes'].append(box)
            state['player_box']=None;state['drag']=None;draw()
    def undo(event=None):
        if state['step']==0 and state['court']:state['court'].pop()
        elif state['step']==1:
            state['player_box']=None;state['drag']=None
            if state['player_boxes']:state['player_boxes'].pop()
        elif state['step']==2:state['shuttle']=None
        draw()
    def advance(skip=False):
        try:
            if state['step']==0:
                if len(state['court'])<3:raise ValueError('请至少点击三个球场顶点')
            elif state['step']==1:
                validate_annotation({k:state[k] for k in ('court','player_boxes','shuttle')})
            elif state['step']==2:
                if skip:state['shuttle']=None
                elif state['shuttle'] is None:raise ValueError('请点击羽毛球，或选择“跳过羽毛球”')
            else:
                state['result']=validate_annotation({k:state[k] for k in ('court','player_boxes','shuttle')})
                state['result']['frame_time']=t;root.destroy();return
            state['step']+=1;draw()
        except (ValueError,TypeError) as e:messagebox.showerror('请检查标注',str(e),parent=root)
    def next_frame():
        nonlocal photo,t
        try:
            new_frame,new_t=read_setup_frame(video_path,t+.5)
            if new_t<=t:raise RuntimeError('已到视频末尾，无法继续前进')
            new_photo=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(new_frame,cv2.COLOR_BGR2RGB)).resize((dw,dh)))
        except (ValueError,RuntimeError,OSError) as e:
            messagebox.showinfo('无法前进',str(e),parent=root)
            return
        photo=new_photo;t=new_t
        # All coordinates must belong to the displayed frame, not the old one.
        state.update(step=0,court=[],player_boxes=[],player_box=None,shuttle=None,drag=None)
        draw()
    def back():state['step']=max(0,state['step']-1);state['drag']=None;draw()
    def skip_setup():
        state['skipped']=True
        root.destroy()
    navigation=tk.Frame(root);navigation.pack(pady=4)
    tk.Button(navigation,text='前进0.5秒',command=next_frame,
              state='normal' if video_path is not None else 'disabled').pack(side='left',padx=8)
    tk.Label(navigation,text='可连续点击选帧；换帧后清空标注，请选好画面再标定。').pack(side='left')
    bar=tk.Frame(root);bar.pack(pady=8)
    tk.Button(bar,text='跳过全部标注，自动检测',command=skip_setup).pack(side='left',padx=8)
    tk.Button(bar,text='上一步',command=back).pack(side='left',padx=8)
    tk.Button(bar,text='撤销/清除当前标记',command=undo).pack(side='left',padx=8)
    skip_button=tk.Button(bar,text='跳过羽毛球',command=lambda:advance(True));skip_button.pack(side='left',padx=8)
    next_button=tk.Button(bar,command=advance);next_button.pack(side='left',padx=8)
    tk.Button(bar,text='取消运行',command=root.destroy).pack(side='left',padx=8)
    canvas.bind('<Button-1>',click);canvas.bind('<B1-Motion>',drag);canvas.bind('<ButtonRelease-1>',release);canvas.bind('<Button-3>',undo)
    root.bind('<Escape>',lambda e:root.destroy());draw();root.mainloop()
    if state['skipped']:return None
    if state['result'] is None:raise RuntimeError('已取消人工标注，未开始剪辑')
    return state['result']


def read_setup_frame(path,start):
    cap=cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():raise ValueError(f'Cannot open {path}')
        fps=cap.get(cv2.CAP_PROP_FPS);count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps>0 and count>0 and start>=count/fps:raise ValueError('视频不足指定标注时间，请用 --setup-seconds 指定更早的秒数')
        if start and not cap.set(cv2.CAP_PROP_POS_MSEC,start*1000):raise RuntimeError('无法定位分析起始帧')
        ok,frame=cap.read()
        if not ok:raise RuntimeError('无法读取分析起始帧')
        return frame,max(start,cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
    finally:cap.release()
