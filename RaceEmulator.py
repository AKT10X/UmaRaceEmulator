import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import math
from stats_interface import characters

# --- 定数設定 ---
dt = 0.0666  # フレーム時間 [s]
initial_speed = 3.0  # スタート時速度 [m/s]
dash_acc = 24.0  # スタートダッシュ追加加速度 [m/s²]
lane_width = 11.25  # レーン幅 [m]
speed_factor = 4  # アニメーション再生倍率

# --- コース読み込み ---
def load_course(path='コース対座標.csv'):
    df = pd.read_csv(path, header=None)
    return df.iloc[:,0].values, df.iloc[:,1].values, df.iloc[:,2].values

distances, xs, zs = load_course()
course_length = distances[-1]

# --- 基準データ ---
rho0 = np.linalg.norm([xs[1]-xs[0], zs[1]-zs[0]])/(distances[1]-distances[0])
reference_speed = 20.0 - (course_length-2000.0)/1000.0
decel_map = {0:-1.2,1:-0.8,2:-1.0,3:-1.0}
phase_coeff = {
    '逃げ':[1.0,0.98,0.962,0.962],
    '先行':[0.978,0.991,0.975,0.975],
    '差し':[0.938,0.998,0.994,0.994],
    '追込':[0.931,1.0,1.0,1.0],
    '大逃げ':[1.063,0.962,0.95,0.95],
}
dist_corr={'S':1.05,'A':1.0,'B':0.9,'C':0.8,'D':0.6,'E':0.4,'F':0.2,'G':0.1}
total_sections=24
section_length=course_length/total_sections

def get_phase(d):
    sec=int(d/section_length)+1
    sec=min(max(sec,1),total_sections)
    if sec<=4: return 0
    if sec<=16: return 1
    if sec<=20: return 2
    return 3

# --- 位置計算 ---
def calculate_positions(d, lanes):
    r_list=[]
    for di, lane in zip(d, lanes):
        idx=np.clip(np.searchsorted(distances,di)-1,0,len(distances)-2)
        d0,d1=distances[idx],distances[idx+1]
        x0,z0=xs[idx],zs[idx]
        x1,z1=xs[idx+1],zs[idx+1]
        alpha=(di-d0)/(d1-d0)
        base=np.array([x0,z0])*(1-alpha)+np.array([x1,z1])*alpha
        seg=np.array([x1-x0,z1-z0])
        u=seg/np.linalg.norm(seg)
        normal=np.array([u[1],-u[0]])
        r_list.append(base+normal*(lane*lane_width))
    return r_list

# --- rho 計算 ---
def calculate_rho(r_list, prev_r, d, d_prev):
    rho=np.empty_like(d)
    for i in range(len(d)):
        delta=d[i]-d_prev[i]
        if prev_r[i] is None or delta<=0:
            rho[i]=rho0
        else:
            rho[i]=np.linalg.norm(r_list[i]-prev_r[i])/delta
    return rho

# --- スキル効果計算 (一度のみ発動) ---
def process_skills(stats, dist_i, rank_i, timers, used, dt):
    speed_eff=0.0
    accel_eff=0.0
    activated=[]
    for j, skill in enumerate(stats.skills):
        # 未発動かつ条件満足時に発動
        if not used[j] and dist_i>=skill.activation_pos and \
           (skill.rank_condition is None or rank_i==skill.rank_condition):
            timers[j]=skill.base_duration*(course_length/1000.0)
            used[j]=True
            activated.append(skill.name)
        # 持続中効果
        if timers[j]>0.0:
            if skill.effect_type=='accel': accel_eff+=skill.effect
            else: speed_eff+=skill.effect
            timers[j]-=dt
    return speed_eff, accel_eff, activated

# --- 速度更新 ---
def update_velocity(old_v, params, start_dash_flag):
    new_v=old_v
    if params['t']>0:
        if start_dash_flag:
            new_v=old_v+(params['reg_acc']+params['dash_acc']+params['accel_eff'])*params['dt']
            cap=0.85*params['reference_speed']
            if new_v>=cap:
                new_v, start_dash_flag=cap, False
        else:
            if old_v<params['target_v']:
                new_v=old_v+(params['reg_acc']+params['accel_eff'])*params['dt']
                new_v=min(new_v,params['target_v'])
            else:
                new_v=old_v+params['decel_val']*params['dt']
                new_v=max(new_v,params['target_v'])
            new_v=max(new_v,params['min_spd'])
        new_v=min(new_v,30.0)
    return new_v, start_dash_flag

# --- シミュレーション ---
def simulate(chars):
    n=len(chars)
    lanes=[c.stats.initial_lane for c in chars]
    powers=[c.stats.power for c in chars]
    gutses=[c.stats.guts for c in chars]
    spds=[c.stats.speed for c in chars]

    d=np.full(n,distances[0],dtype=float)
    v=np.full(n,initial_speed)
    start_dash=np.ones(n,bool)
    prev_r=[None]*n; d_prev=np.zeros(n)
    timers=[[0.0]*len(c.stats.skills) for c in chars]
    used=[[False]*len(c.stats.skills) for c in chars]

    times=[]; pos_list=[[] for _ in range(n)]; speed_list=[[] for _ in range(n)]; dist_list=[[] for _ in range(n)]; act_list=[[] for _ in range(n)]
    t=0.0
    while np.any(d<course_length):
        ranks=np.empty(n,int)
        for idx,runner in enumerate(np.argsort(-d)): ranks[runner]=idx+1

        r_list=calculate_positions(d,lanes)
        rho_vals=calculate_rho(r_list,prev_r,d,d_prev)
        times.append(t)

        for i,c in enumerate(chars):
            min_spd=0.85*reference_speed+math.sqrt(200.0*gutses[i])*0.001
            reg_acc=0.0006*math.sqrt(500.0*powers[i])
            ph=get_phase(d[i]); decel_val=decel_map[ph]
            coeff=phase_coeff[c.stats.running_style][ph]
            base_tgt=reference_speed*coeff if ph<2 else reference_speed*coeff+math.sqrt(500.0*spds[i])*dist_corr[c.stats.distance_apt]*0.002

            speed_eff, accel_eff, activated = process_skills(c.stats,d[i],ranks[i],timers[i],used[i],dt)
            act_list[i].append(activated)
            target_v=base_tgt+speed_eff

            params={'reg_acc':reg_acc,'dash_acc':dash_acc,'accel_eff':accel_eff,'target_v':target_v,'min_spd':min_spd,'decel_val':decel_val,'t':t,'dt':dt,'reference_speed':reference_speed}
            v[i],start_dash[i]=update_velocity(v[i],params,start_dash[i])

            pos_list[i].append(r_list[i]); speed_list[i].append(v[i]); dist_list[i].append(d[i])

        for i in range(n): d_prev[i],prev_r[i]=d[i],r_list[i]; d[i]+=(v[i]*dt)/max(1.0,rho_vals[i]/rho0)
        t+=dt
    return np.array(times), pos_list, speed_list, dist_list, act_list

# --- 実行 ---
if __name__=='__main__':
    times, pos_list, speed_list, dist_list, act_list = simulate(characters)
    # CSV
    rec=[]
    for i,c in enumerate(characters):
        for f,tt in enumerate(times):
            rec.append({'character':c.name,'frame':f,'time_s':tt,'speed_m_s':speed_list[i][f],'distance_m':dist_list[i][f],'activated':'/'.join(act_list[i][f])})
    pd.DataFrame(rec).to_csv('race_data.csv',index=False)
    # GIF
    colors=plt.cm.tab10(np.linspace(0,1,len(characters)))
    fig,ax=plt.subplots(figsize=(4,4))
    ax.plot(xs,zs,color='gray',linewidth=1)
    lines=[ax.plot([],[],marker='o',linestyle='',label=c.name,color=colors[i])[0] for i,c in enumerate(characters)]
    text=ax.text(0.02,0.98,'',transform=ax.transAxes,va='top',ha='left',fontsize=8,bbox=dict(facecolor='white',alpha=0.7))
    ax.set_aspect('equal');ax.axis('off')
    def init():
        for ln in lines: ln.set_data([],[])
        text.set_text('')
        return lines+[text]
    def update(frame):
        for i,ln in enumerate(lines):
            x,z = pos_list[i][frame]
            ln.set_data([x],[z])
        txt = ' '.join(f"{c.name}:{speed_list[i][frame]:.1f}m/s" + (f" ({act_list[i][frame][0]})" if act_list[i][frame] else '') for i,c in enumerate(characters))
        text.set_text(txt)
        return lines+[text]
    ani=FuncAnimation(fig,update,frames=len(times),init_func=init,blit=True)
    ani.save('race_animation.gif',writer=PillowWriter(fps=speed_factor/dt),dpi=80)
    print('Saved CSV & GIF')
