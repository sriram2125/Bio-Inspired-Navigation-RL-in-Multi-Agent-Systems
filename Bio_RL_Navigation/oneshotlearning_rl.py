import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter

# ==========================================
# 1. CONFIGURATION & PHYSICS
# ==========================================
MAP_BOUNDS = {
    'x_min': -10, 'x_max': 60,
    'y_min': -10, 'y_max': 60
}

OBSTACLES = [
    (18, 18, 7.0),
    (32, 32, 6.0)
]

NEST_POS = np.array([0.0, 0.0])
FOOD_POS = np.array([45.0, 45.0])

def check_collision(pos):
    """Returns True if pos hits Obstacle OR Wall"""
    x, y = pos[0], pos[1]
    if (x < MAP_BOUNDS['x_min'] or x > MAP_BOUNDS['x_max'] or 
        y < MAP_BOUNDS['y_min'] or y > MAP_BOUNDS['y_max']):
        return True
    for (ox, oy, r) in OBSTACLES:
        dist = np.linalg.norm(pos - np.array([ox, oy]))
        if dist < r + 0.5: 
            return True
    return False

# ==========================================
# 2. RL AGENT (Homing Recovery)
# ==========================================
class RLAnt:
    def __init__(self, start_pos, target_pos, learning_rate=0.1, discount=0.95, epsilon=1.0):
        self.q_table = np.zeros((24, 3)) 
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.decay = 0.95 
        
        self.start_pos = np.array(start_pos, dtype=float)
        self.target = np.array(target_pos, dtype=float)
        self.pos = self.start_pos.copy()
        self.heading = np.random.uniform(0, 2*np.pi)
        
    def reset(self):
        self.pos = self.start_pos.copy()
        self.heading = np.random.uniform(0, 2*np.pi)
        return self.get_state()

    def get_state(self):
        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]
        target_heading = np.arctan2(dy, dx)
        angle_diff = (target_heading - self.heading + np.pi) % (2 * np.pi) - np.pi
        sector = int((angle_diff + np.pi) / (2 * np.pi) * 12)
        sector = min(max(sector, 0), 11)
        
        look_dist = 5.0
        lx = self.pos[0] + look_dist * np.cos(self.heading)
        ly = self.pos[1] + look_dist * np.sin(self.heading)
        obstacle_detected = 1 if check_collision(np.array([lx, ly])) else 0
        
        return sector + (12 * obstacle_detected)

    def step(self, training=True):
        state = self.get_state()
        if training and np.random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            action = np.argmax(self.q_table[state])

        steering = (action - 1) * 0.5 
        new_heading = self.heading + steering
        speed = 2.0
        new_pos = self.pos.copy()
        new_pos[0] += speed * np.cos(new_heading)
        new_pos[1] += speed * np.sin(new_heading)
        
        collision = check_collision(new_pos)
        reward = -1 
        done = False
        
        if collision:
            reward += -20 
            self.heading += np.pi + np.random.uniform(-0.5, 0.5) 
        else:
            prev_dist = np.linalg.norm(self.target - self.pos)
            self.pos = new_pos
            self.heading = new_heading
            curr_dist = np.linalg.norm(self.target - self.pos)
            reward += (prev_dist - curr_dist) * 10 
            if curr_dist < 4.0: 
                reward += 200 
                done = True
                
        if training:
            next_state = self.get_state()
            best_next = np.argmax(self.q_table[next_state])
            td_target = reward + self.gamma * self.q_table[next_state][best_next]
            self.q_table[state][action] += self.lr * (td_target - self.q_table[state][action])
        return done

# ==========================================
# 3. LIVE SIMULATION
# ==========================================
def run_live_and_save():
    history = {'scout': [], 'scout_return': [], 'learner': [], 'trials': [], 'best': ([], [])}

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(MAP_BOUNDS['x_min'], MAP_BOUNDS['x_max'])
    ax.set_ylim(MAP_BOUNDS['y_min'], MAP_BOUNDS['y_max'])
    ax.set_xticks([]); ax.set_yticks([])
    
    # Draw Static Elements
    for spine in ax.spines.values(): spine.set_linewidth(2)
    for (ox, oy, r) in OBSTACLES:
        ax.add_patch(Circle((ox, oy), r, color='#404040', alpha=0.9))
    ax.plot(NEST_POS[0], NEST_POS[1], 'go', markersize=12, label='Nest')
    ax.plot(FOOD_POS[0], FOOD_POS[1], 'r*', markersize=15, label='Food')
    
    # --- PLOT LINES ---
    ln_scout_out, = ax.plot([], [], 'b-', alpha=0.3, label='Scout Search')
    ln_scout_ret, = ax.plot([], [], color='darkblue', linewidth=1.5, alpha=0.6, label='Scout Return')
    ln_learn, = ax.plot([], [], 'g-', linewidth=4, label='Learner (Stigmergy)')
    ln_lost_out, = ax.plot([], [], color='orange', linestyle='--', linewidth=2, label='Lost (Outbound)')
    ln_trial, = ax.plot([], [], 'm:', linewidth=0.8, alpha=0.5, label='RL Trials')
    ln_optim, = ax.plot([], [], color='orange', linewidth=1.5, label='RL Optimal Path')
    
    status_txt = ax.text(0.98, 0.95, "Initializing...", transform=ax.transAxes, 
                        ha='right', va='top', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
    print("Starting Live View...")

    # --- PHASE 1: SCOUT (Random Search) ---
    status_txt.set_text("Phase 1: Scout Search (Random)")
    plt.pause(0.5)
    
    curr = NEST_POS.copy()
    sx, sy = [curr[0]], [curr[1]]
    found = False
    step = 0
    while not found and step < 600:
        angle = np.arctan2(FOOD_POS[1]-curr[1], FOOD_POS[0]-curr[0]) + np.random.uniform(-1.5, 1.5)
        test = curr + np.array([2.0*np.cos(angle), 2.0*np.sin(angle)])
        if not check_collision(test):
            curr = test
            sx.append(curr[0]); sy.append(curr[1])
            if np.linalg.norm(curr - FOOD_POS) < 3.0: found = True
        step += 1
        # LIVE UPDATE
        if step % 5 == 0:
            ln_scout_out.set_data(sx, sy)
            plt.pause(0.001)
            
    ln_scout_out.set_data(sx, sy)
    history['scout'] = (sx, sy)

    # --- PHASE 1: SCOUT RETURN (Homing + Avoidance) ---
    status_txt.set_text("Phase 1: Scout Return (Creating Trail)")
    plt.pause(0.5)
    
    rx, ry = [curr[0]], [curr[1]]
    home = False
    step = 0
    current_heading = np.arctan2(NEST_POS[1]-curr[1], NEST_POS[0]-curr[0])
    
    while not home and step < 600:
        vec_to_nest = NEST_POS - curr
        desired_heading = np.arctan2(vec_to_nest[1], vec_to_nest[0])
        speed = 2.0
        test_pos_straight = curr + speed * np.array([np.cos(desired_heading), np.sin(desired_heading)])
        
        if not check_collision(test_pos_straight):
            curr = test_pos_straight
            current_heading = desired_heading
        else:
            avoid_heading = current_heading + np.random.uniform(0.5, 2.0) 
            test_pos_avoid = curr + speed * np.array([np.cos(avoid_heading), np.sin(avoid_heading)])
            if not check_collision(test_pos_avoid):
                curr = test_pos_avoid
                current_heading = avoid_heading
            else:
                current_heading -= 1.0 
        rx.append(curr[0]); ry.append(curr[1])
        if np.linalg.norm(curr - NEST_POS) < 3.0: home = True
        step += 1
        
        # LIVE UPDATE
        if step % 5 == 0:
            ln_scout_ret.set_data(rx, ry)
            plt.pause(0.001)
            
    history['scout_return'] = (rx, ry)

    # --- PHASE 2: LEARNER ---
    status_txt.set_text("Phase 2: Learner (One-Shot Stigmergy)")
    plt.pause(0.5)
    
    lx = rx[::-1] 
    ly = ry[::-1]
    
    # LIVE UPDATE (Animate Learner following path)
    for i in range(0, len(lx), 3):
        ln_learn.set_data(lx[:i], ly[:i])
        plt.pause(0.01)
    ln_learn.set_data(lx, ly)
    history['learner'] = (lx, ly)

    # --- PHASE 3: LOST AGENT OUTBOUND ---
    status_txt.set_text("Phase 3: Lost Agent (Follows Trail)")
    plt.pause(0.5)
    
    # LIVE UPDATE (Animate Lost Outbound)
    for i in range(0, len(lx), 3):
        ln_lost_out.set_data(lx[:i], ly[:i])
        plt.pause(0.01)
    ln_lost_out.set_data(lx, ly)
    plt.pause(0.5)

    # --- PHASE 3: LOST AGENT RL (Inbound) ---
    status_txt.set_text("Trail Evaporated! RL Starting...")
    ln_lost_out.set_alpha(0.3)
    plt.pause(0.5)
    
    agent = RLAnt(start_pos=FOOD_POS, target_pos=NEST_POS)
    episodes = 40
    best_len = 9999
    
    for e in range(episodes):
        agent.reset()
        ep_x, ep_y = [agent.pos[0]], [agent.pos[1]]
        steps_taken = 0
        success = False
        
        status_txt.set_text(f"RL Episode {e+1}/{episodes}\nEps: {agent.epsilon:.2f}")
        
        for _ in range(150): 
            done = agent.step(training=True)
            ep_x.append(agent.pos[0]); ep_y.append(agent.pos[1])
            steps_taken += 1
            
            # LIVE UPDATE (Show RL Trial)
            ln_trial.set_data(ep_x, ep_y)
            plt.pause(0.001)
            
            if done:
                success = True
                break
        
        history['trials'].append((list(ep_x), list(ep_y)))
        if success and steps_taken < best_len:
            best_len = steps_taken
            history['best'] = (list(ep_x), list(ep_y))
            # Show new best path
            ln_optim.set_data(ep_x, ep_y)
            plt.pause(0.1)
            
        agent.epsilon *= agent.decay
    
    ln_trial.set_data([], []) # Clear trials to show final result

    status_txt.set_text("Simulation Done. Generating GIF...")
    print("Logic Complete. Building Animation...")
    plt.ioff()
    
    save_gif(fig, ax, history, ln_scout_out, ln_scout_ret, ln_learn, ln_lost_out, ln_trial, ln_optim, status_txt)

def save_gif(fig, ax, history, l_s_out, l_s_ret, l_learn, l_lost_out, l_trial, l_optim, txt_obj):
    # Clear lines for GIF replay
    l_s_out.set_data([], [])
    l_s_ret.set_data([], [])
    l_learn.set_data([], [])
    l_lost_out.set_data([], [])
    l_trial.set_data([], [])
    l_optim.set_data([], [])

    sx, sy = history['scout']
    rx, ry = history['scout_return']
    lx, ly = history['learner']
    trials = history['trials']
    bx, by = history['best']
    
    # Frames
    f_scout = len(sx) // 4
    f_ret = len(rx) // 4
    f_learn = len(lx) // 4
    f_lost_out = len(lx) // 4
    f_trials = len(trials)
    
    # PAUSE DURATION (frames) - Allows time for screenshots during GIF replay
    PAUSE = 45 
    
    total_frames = f_scout + f_ret + PAUSE + f_learn + PAUSE + f_lost_out + PAUSE + (f_trials * 2) + PAUSE + 20
    
    def update(frame):
        curr_f = frame
        
        # --- PHASE 1: SCOUT OUTBOUND ---
        if curr_f < f_scout:
            txt_obj.set_text("Phase 1: Scout Search (Random)")
            idx = curr_f * 4
            l_s_out.set_data(sx[:idx], sy[:idx])
            return l_s_out, txt_obj
        curr_f -= f_scout
        l_s_out.set_data(sx, sy) 
        
        # --- PHASE 1: SCOUT RETURN ---
        if curr_f < f_ret:
            txt_obj.set_text("Phase 1: Scout Return (Creating Trail)")
            idx = curr_f * 4
            l_s_ret.set_data(rx[:idx], ry[:idx])
            return l_s_out, l_s_ret, txt_obj
        curr_f -= f_ret
        l_s_ret.set_data(rx, ry)
        
        # PAUSE 1
        if curr_f < PAUSE:
            return l_s_out, l_s_ret, txt_obj
        curr_f -= PAUSE
        
        # --- PHASE 2: LEARNER ---
        if curr_f < f_learn:
            txt_obj.set_text("Phase 2: Learner (One-Shot Stigmergy)")
            l_s_out.set_alpha(0.1)
            l_s_ret.set_alpha(0.3)
            idx = curr_f * 4
            l_learn.set_data(lx[:idx], ly[:idx])
            return l_s_out, l_s_ret, l_learn, txt_obj
        curr_f -= f_learn
        l_learn.set_data(lx, ly)
        
        # PAUSE 2
        if curr_f < PAUSE:
            return l_s_out, l_s_ret, l_learn, txt_obj
        curr_f -= PAUSE
        
        # --- PHASE 3: LOST OUTBOUND ---
        if curr_f < f_lost_out:
            txt_obj.set_text("Phase 3: Lost Agent (Follows Trail)")
            l_learn.set_alpha(0.2)
            idx = curr_f * 4
            l_lost_out.set_data(lx[:idx], ly[:idx])
            return l_s_out, l_s_ret, l_learn, l_lost_out, txt_obj
        curr_f -= f_lost_out
        l_lost_out.set_data(lx, ly)
        
        # PAUSE 3
        if curr_f < PAUSE:
            txt_obj.set_text("Trail Evaporated! RL Starting...")
            l_lost_out.set_alpha(0.3)
            return l_s_out, l_s_ret, l_learn, l_lost_out, txt_obj
        curr_f -= PAUSE

        # --- PHASE 3: RL TRIALS ---
        if curr_f < f_trials * 2:
            txt_obj.set_text("Phase 3: RL Recovery (Learning Path)")
            idx = curr_f // 2
            if idx < len(trials):
                acc_x, acc_y = [], []
                for i in range(idx + 1):
                    tx, ty = trials[i]
                    acc_x.extend(tx); acc_x.append(np.nan)
                    acc_y.extend(ty); acc_y.append(np.nan)
                l_trial.set_data(acc_x, acc_y)
            return l_s_out, l_s_ret, l_learn, l_lost_out, l_trial, txt_obj
        curr_f -= (f_trials * 2)

        # --- FINAL RESULT ---
        l_trial.set_data([], [])
        l_optim.set_data(bx, by)
        txt_obj.set_text("Simulation Complete")
        return l_s_out, l_s_ret, l_learn, l_lost_out, l_optim, txt_obj

    ani = FuncAnimation(fig, update, frames=total_frames, blit=True, interval=30)
    writer = PillowWriter(fps=30)
    ani.save("Stigmergy_Live_Output.gif", writer=writer)
    print("Success! Saved 'Stigmergy_Live_Output.gif'")
    plt.close()

if __name__ == "__main__":
    run_live_and_save()