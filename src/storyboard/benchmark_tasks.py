"""Predeclared P8 five-task specification; only story text enters model prompt."""
import copy

def tasks(group_id,names):
    a,b=names;rows=[]
    def rel(n,kind,param):return dict(kind='relation',scene=n,subject='a',object='b',relation=kind,parameters=param)
    def event(n,action):
        r=dict(kind='event',scene=n,subject='a',action=action)
        if action in ('approach','retreat'):r.update(object='b',min_change=.2)
        return r
    def common():return [dict(kind='pose',scene=n,subject='b',reference='initial') for n in range(1,4)]
    def visible():return [dict(kind='visibility',scene=n,subject=s,visible=True) for n in range(1,4) for s in ['a','b']]
    def views(focus=None):return [dict(kind='shot',scene=n,template='focus' if focus==n else 'overview',targets=['b'] if focus==n else []) for n in range(1,4)]
    rows.append(dict(id='T1',story=f'三幕：{a}始终在{b}左侧附近（left_of gap>=0.2U，near<=0.5U），第二、三幕{a}保持前一幕位置。{b}始终在输入位置不动，两者始终可见，全用overview全景。',requirements=common()+visible()+views()+[rel(n,'left_of',{'gap':.2}) for n in range(1,4)]+[rel(n,'near',{'near_max':.5}) for n in range(1,4)]+[event(2,'hold'),event(3,'hold')]))
    rows.append(dict(id='T2',story=f'三幕：第一幕{a}在{b}右侧附近（gap>=0.2U，near<=0.5U）；第二幕{a}相对{b}退远至少0.2U，最终仍在右侧且far间距>=1.5U；第三幕保持第二幕位置。{b}始终保持输入位置。两者始终可见，全用overview全景。',requirements=common()+visible()+views()+[rel(1,'right_of',{'gap':.2}),rel(1,'near',{'near_max':.5}),rel(2,'right_of',{'gap':.2}),rel(2,'far',{'far_min':1.5}),event(2,'retreat'),event(3,'hold')]))
    rows.append(dict(id='T3',story=f'三幕overview全景：第一幕{a}与{b}保持输入位置且均可见；第二幕{a}消失；第三幕{a}在输入位置重新出现。{b}始终在输入位置可见且不动。不需要其他空间变化。',requirements=common()+views()+[dict(kind='visibility',scene=n,subject='a',visible=n!=2) for n in range(1,4)]+[dict(kind='visibility',scene=n,subject='b',visible=True) for n in range(1,4)]+[dict(kind='pose',scene=n,subject='a',reference='initial') for n in [1,3]]+[event(2,'disappear'),event(3,'appear')]))
    rows.append(dict(id='T4',story=f'三幕中{a}和{b}都保持输入位置、始终可见。第一幕overview全景；第二幕只将镜头改为focus关注{b}并完整展示；第三幕回到overview全景。没有物体运动。',requirements=common()+visible()+views(2)+[dict(kind='pose',scene=n,subject='a',reference='initial') for n in range(1,4)]))
    rows.append(dict(id='T5',story=f'三幕overview全景：第一幕{a}在{b}左侧远处，far间距>=2.5U；第二幕{a}相对{b}靠近至少0.2U，但仍在左侧且far间距>=1.5U；第三幕保持第二幕位置。{b}始终保持输入位置，两者始终可见。',requirements=common()+visible()+views()+[rel(1,'left_of',{'gap':.2}),rel(1,'far',{'far_min':2.5}),rel(2,'left_of',{'gap':.2}),rel(2,'far',{'far_min':1.5}),event(2,'approach'),event(3,'hold')]))
    return [dict(**r,group_id=group_id,instance_ids=['a','b']) for r in rows]
