"""
Continue training script for fighter policy.

Usage:
    python continue_training.py

What it does:
    1. Loads existing fighter_policy.pt and fighter_value.pt
    2. Continues PPO training from where you left off
    3. Allows you to train for additional steps with different hyperparameters
    4. Useful for: extending training, fine-tuning, or resuming interrupted runs

Options:
    - Set ADDITIONAL_STEPS to control how much more training
    - Set OVERRIDE_LR to change learning rate (useful for fine-tuning)
    - Set new reward weights or hyperparameters
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import os

# Import the environment from your training script
import missile_intercept as sim

# ===========================================================================
# CONFIGURATION - CUSTOMIZE THESE
# ===========================================================================

# How many additional steps to train?
ADDITIONAL_STEPS = 1_000_000  # Add 2M more steps

# Learning rate (set to None to keep original, or specify new value)
OVERRIDE_LR = None  # e.g., 1e-4 for fine-tuning, None to keep 3e-4

# Device
DEVICE = "mps"  # or "cpu"

# Checkpoints to load
POLICY_CHECKPOINT = "fighter_policy.pt"
VALUE_CHECKPOINT = "fighter_value.pt"

# Where to save continued training
POLICY_OUTPUT = "fighter_policy_continued.pt"
VALUE_OUTPUT = "fighter_value_continued.pt"

# Training hyperparameters (can override)
OVERRIDE_CONFIG = {
    # "gamma": 0.99,
    # "clip_eps": 0.15,  # Tighter clipping for fine-tuning
    # "entropy_coef": 0.005,  # Less exploration
}

# ===========================================================================
# Import environment from train_fighter
# ===========================================================================

# Copy the FighterEnv class from train_fighter.py
# (Or import it if you refactor train_fighter to be importable)
#TRAINING_MISSILE_INITIAL_SPEED = 300.0  # was 750 # m/s, approximate post-boost speed
TRAINING_MISSILE_INITIAL_SPEED = 750.0

class FighterEnv:
    """
    Fast RL environment:
      - State:  12D observation from fighter POV
      - Action: 4D (dir_r, dir_u, extra, mag_raw) -> world accel via _action_to_accel
      - Reward: +time alive +distance bonus, big -ve on kill/crash
    """

    def __init__(self, dt=0.02, max_time=40.0, seed=0):
        self.dt = dt
        self.max_steps = int(max_time / dt)
        self.rng = np.random.default_rng(seed)

        # altitude band similar to your fighter
        self.alt_band = (9000.0, 13000.0)
        self.kill_radius = sim.KILL_DIST  # Use same value as main sim (50m)

        self.step_count = 0

        # For heuristic expert (BC)
        self.altitude_cmd = None
        self.last_alt_update = 0.0

        # For missile acceleration estimation
        self.prev_fighter_vel = None

    # ---------- reset / observation ----------

    def reset(self):
        # Sample start geometry similar to sample_random_starts
        ac_start, mi_start = sim.sample_random_starts(self.rng)

        self.fighter_pos = ac_start.astype(float).copy()
        self.fighter_vel = np.array(
            [sim.TARG_INITIAL_SPEED, 0.0, 0.0],
            dtype=float
        )

        self.missile_pos = mi_start.astype(float).copy()

        rel = self.fighter_pos - self.missile_pos
        R0 = np.linalg.norm(rel)
        if R0 < 1e-6:
            rel = np.array([1.0, 0.0, 0.0], dtype=float)
            R0 = 1.0
        los = rel / R0
        self.missile_vel = TRAINING_MISSILE_INITIAL_SPEED * los

        self.step_count = 0

        # heuristic pilot state
        alt_min, alt_max = self.alt_band
        self.altitude_cmd = np.clip(self.fighter_pos[2], alt_min, alt_max)
        self.last_alt_update = 0.0

        #Reset missile accel estimation
        self.prev_fighter_vel = None
    
        return self._get_obs()

    def _get_obs(self):
        """
        12D obs (same structure as TargetAircraft._build_observation):

            [ r/30000 (3),
              v_rel/1000 (3),
              own_vel/400 (3),
              R/30000 (1),
              speed/400 (1),
              (alt - mid_alt) / (span/2) (1) ]
        """
        r = self.missile_pos - self.fighter_pos
        v_rel = self.missile_vel - self.fighter_vel

        R = np.linalg.norm(r)
        speed = np.linalg.norm(self.fighter_vel)

        r_norm = r / 30000.0
        v_norm = v_rel / 1000.0
        v_self_norm = self.fighter_vel / 400.0

        alt_min, alt_max = self.alt_band
        alt_mid = 0.5 * (alt_min + alt_max)
        alt_span = max(1.0, alt_max - alt_min)
        alt_norm = (self.fighter_pos[2] - alt_mid) / (alt_span / 2.0)

        R_norm = R / 30000.0
        speed_norm = speed / 400.0

        obs = np.concatenate([
            r_norm,
            v_norm,
            v_self_norm,
            np.array([R_norm, speed_norm, alt_norm], dtype=float)
        ])
        return obs.astype(np.float32)

    # ---------- action mapping (same idea as FighterRLPolicy) ----------

    def _action_to_accel(self, action):
        """
        Convert 4D action to world-frame accel vector.

        action: np.array([dir_r, dir_u, extra, mag_raw])
        """
        a = np.asarray(action, dtype=float)
        dir_raw = a[:3]
        mag_raw = a[3]

        # Build local frame around fighter velocity
        v = self.fighter_vel
        speed = np.linalg.norm(v)
        if speed < 1e-6:
            fwd = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            fwd = v / speed

        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        right = np.cross(fwd, world_up)
        r_norm = np.linalg.norm(right)
        if r_norm < 1e-6:
            right = np.array([0.0, 1.0, 0.0], dtype=float)
            r_norm = 1.0
        right /= r_norm
        up_local = np.cross(right, fwd)

        d_r, d_u, _ = dir_raw
        dir_local = d_r * right + d_u * up_local
        d_norm = np.linalg.norm(dir_local)
        if d_norm < 1e-6:
            return np.zeros(3, dtype=float)
        dir_unit = dir_local / d_norm

        # magnitude: tanh squash then 0..max_total_accel
        mag_scale = math.tanh(mag_raw)          # (-1,1)
        mag_scale = max(0.0, mag_scale)        # no braking (you could allow it)
        max_total_accel = sim.TARG_MAX_G * 9.81
        accel_mag = mag_scale * max_total_accel

        return accel_mag * dir_unit

    # ---------- heuristic "expert" for BC ----------

    def _expert_accel_world(self, t):
        """
        Rough hand-coded evasive strategy:
          - keep altitude in a band
          - turn hard perpendicular to LOS and away from missile
          - add some noise/jink
        Returns a desired acceleration vector in WORLD coordinates.
        """
        max_total_accel = sim.TARG_MAX_G * 9.81

        # Geometry
        r = self.missile_pos - self.fighter_pos
        R = np.linalg.norm(r)
        if R < 1e-6:
            return np.zeros(3, dtype=float)
        los = r / R

        speed = np.linalg.norm(self.fighter_vel)
        if speed < 1e-3:
            fwd = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            fwd = self.fighter_vel / speed

        # Altitude control (update cmd every few seconds)
        if t - self.last_alt_update > 5.0:
            alt_min, alt_max = self.alt_band
            self.altitude_cmd = self.rng.uniform(alt_min, alt_max)
            self.last_alt_update = t

        alt_error = self.altitude_cmd - self.fighter_pos[2]
        kz = 0.01
        a_z_cmd = np.clip(kz * alt_error, -2.0 * 9.81, 2.0 * 9.81)
        vertical_accel = np.array([0.0, 0.0, a_z_cmd], dtype=float)

        # Threat based on range
        if R > 25000.0:
            threat = 0.2
        elif R > 8000.0:
            threat = 0.5
        else:
            threat = 1.0

        # Beam/break away from missile LOS projected perpendicular to fighter fwd
        los_away = -los
        lateral_raw = los_away - np.dot(los_away, fwd) * fwd
        lat_norm = np.linalg.norm(lateral_raw)
        if lat_norm < 1e-5:
            tmp = np.array([0.0, 0.0, 1.0], dtype=float)
            lateral_raw = np.cross(fwd, tmp)
            lat_norm = np.linalg.norm(lateral_raw)
            if lat_norm < 1e-5:
                lateral_raw = np.array([0.0, 1.0, 0.0], dtype=float)
                lat_norm = 1.0
        lateral_dir = lateral_raw / lat_norm
        a_lat_mag = threat * max_total_accel * 0.9
        lateral_accel = a_lat_mag * lateral_dir

        # Jink: small random lateral accel perpendicular to fwd
        jink_mag = 0.2 * 9.81
        jink = jink_mag * self.rng.standard_normal(3)
        jink = jink - np.dot(jink, fwd) * fwd

        a_cmd = vertical_accel + lateral_accel + jink

        # Enforce g-limit including gravity, like sim.TargetAircraft
        total = a_cmd + sim.GRAVITY
        total_mag = np.linalg.norm(total)
        if total_mag > max_total_accel:
            scale = max_total_accel / total_mag
            total = total * scale
            a_cmd = total - sim.GRAVITY

        return a_cmd

    def expert_action(self, t):
        """
        Convert the expert's desired acceleration vector into an action in the
        same 4D parameterization used by the policy:

            action = [dir_r, dir_u, extra, mag_raw]
        """
        a_des = self._expert_accel_world(t)

        # Build same local frame as _action_to_accel
        v = self.fighter_vel
        speed = np.linalg.norm(v)
        if speed < 1e-6:
            fwd = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            fwd = v / speed

        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        right = np.cross(fwd, world_up)
        r_norm = np.linalg.norm(right)
        if r_norm < 1e-6:
            right = np.array([0.0, 1.0, 0.0], dtype=float)
            r_norm = 1.0
        right /= r_norm
        up_local = np.cross(right, fwd)

        # Project desired accel into lateral plane (right/up)
        a_r = np.dot(a_des, right)
        a_u = np.dot(a_des, up_local)
        mag = math.sqrt(a_r ** 2 + a_u ** 2)

        max_total_accel = sim.TARG_MAX_G * 9.81
        if max_total_accel < 1e-6:
            mag_scale = 0.0
        else:
            mag_scale = np.clip(mag / max_total_accel, 0.0, 0.999)

        # Invert tanh: mag_scale = tanh(mag_raw) -> mag_raw = atanh(mag_scale)
        mag_raw = 0.5 * math.log((1.0 + mag_scale) / (1.0 - mag_scale + 1e-8))

        # dir_r, dir_u are just proportional to those components
        dir_r = a_r
        dir_u = a_u
        extra = 0.0  # unused

        return np.array([dir_r, dir_u, extra, mag_raw], dtype=np.float32)

    # ---------- step function ----------

    def step(self, action, t):
        """
        Apply action, step physics, compute reward.
        NOW WITH REALISTIC MISSILE MODEL matching the main sim!
        Returns: obs, reward, done, info
        """
        self.step_count += 1
        
        # Fighter dynamics
        a_cmd = self._action_to_accel(action)
        a_total_fighter = a_cmd + sim.GRAVITY
        
        self.fighter_vel = self.fighter_vel + a_total_fighter * self.dt
        self.fighter_pos = self.fighter_pos + self.fighter_vel * self.dt
        
        # Keep speed near nominal cruise
        fighter_speed = np.linalg.norm(self.fighter_vel)
        if fighter_speed > 1e-3:
            desired_speed = sim.TARG_INITIAL_SPEED
            self.fighter_vel *= desired_speed / fighter_speed
        
        # Clamp ground
        terminated = False
        if self.fighter_pos[2] < 0.0:
            self.fighter_pos[2] = 0.0
            terminated = True
        
        # =====================================================================
        # REALISTIC MISSILE DYNAMICS (matches main sim)
        # =====================================================================
        
        # Calculate current missile mass and thrust
        t_flight = self.step_count * self.dt
        
        # Mass depletion
        total_burn = sim.MISS_BOOST_TIME + sim.MISS_SUSTAIN_TIME
        if t_flight >= total_burn:
            missile_mass = sim.MISS_MASS_BURNOUT
        else:
            fuel_mass = sim.MISS_MASS_INITIAL - sim.MISS_MASS_BURNOUT
            burn_fraction = t_flight / total_burn
            missile_mass = sim.MISS_MASS_INITIAL - fuel_mass * burn_fraction
        
        # Thrust profile
        if t_flight < sim.MISS_BOOST_TIME:
            thrust_mag = sim.MISS_BOOST_THRUST
        elif t_flight < sim.MISS_BOOST_TIME + sim.MISS_SUSTAIN_TIME:
            thrust_mag = sim.MISS_SUSTAIN_THRUST
        else:
            thrust_mag = 0.0
        
        # Geometry
        r = self.fighter_pos - self.missile_pos
        R = np.linalg.norm(r)
        if R < 1e-6:
            R = 1e-6
        los = r / R
        
        # Closing speed
        rel_vel = self.fighter_vel - self.missile_vel
        Vc = -np.dot(rel_vel, los)
        
        # Augmented PN guidance (matches main sim)
        v_rel_perp = rel_vel - np.dot(rel_vel, los) * los
        n_dot = v_rel_perp / R
        
        # Use same N_PN as main sim
        N_pn = 5.0  # Changed from 4.0 to match sim
        a_pn = N_pn * max(Vc, 0.0) * n_dot
        
        # Estimate target acceleration (simplified APN)
        if self.prev_fighter_vel is None:
            a_targ_est = np.zeros(3, dtype=float)
        else:
            a_targ_est = (self.fighter_vel - self.prev_fighter_vel) / self.dt
            
            # Clamp estimated accel to prevent spikes
            a_est_mag = np.linalg.norm(a_targ_est)
            if a_est_mag > 20.0 * 9.81:
                a_targ_est = a_targ_est * (20.0 * 9.81 / a_est_mag)
        
        # Update velocity history
        self.prev_fighter_vel = self.fighter_vel.copy()
        
        a_targ_perp = a_targ_est - np.dot(a_targ_est, los) * los
        
        # APN command
        a_guidance = a_pn + 0.5 * N_pn * a_targ_perp
        
        # G-limit guidance command
        a_guid_mag = np.linalg.norm(a_guidance)
        max_missile_accel = sim.MISS_MAX_G * 9.81
        if a_guid_mag > max_missile_accel:
            a_guidance = a_guidance * (max_missile_accel / a_guid_mag)
        
        # Thrust acceleration (along velocity)
        missile_speed = np.linalg.norm(self.missile_vel)
        if missile_speed > 1e-6:
            thrust_dir = self.missile_vel / missile_speed
        else:
            thrust_dir = np.array([1.0, 0.0, 0.0])
        
        a_thrust = (thrust_mag / missile_mass) * thrust_dir
        
        # Drag acceleration
        altitude = max(0.0, self.missile_pos[2])
        mach = missile_speed / sim.get_speed_of_sound(altitude)
        
        # Mach-dependent drag coefficient
        if mach < 0.8:
            cd = sim.MISS_CD_SUBSONIC
        elif mach < 1.2:
            t_drag = (mach - 0.8) / 0.4
            cd = sim.MISS_CD_SUBSONIC + t_drag * (sim.MISS_CD_TRANSONIC - sim.MISS_CD_SUBSONIC)
        else:
            cd = sim.MISS_CD_SUPERSONIC + 0.1 / mach
        
        drag_force = sim.calculate_drag(self.missile_vel, altitude, cd, sim.MISS_REF_AREA)
        a_drag = drag_force / missile_mass
        
        # Total missile acceleration
        a_missile_total = a_thrust + a_drag + a_guidance + sim.GRAVITY
        
        # Integrate missile motion
        self.missile_vel = self.missile_vel + a_missile_total * self.dt
        self.missile_pos = self.missile_pos + self.missile_vel * self.dt
        
        if self.missile_pos[2] < 0.0:
            self.missile_pos[2] = 0.0
        
        # =====================================================================
        # REWARD FUNCTION - Tuned for immediate threat response
        # =====================================================================
        
        reward = 0.0
        
        # Convert distance to km for readability
        dist_km = R / 1000.0
        
        # 1. Survival reward (scaled by threat level)
        if dist_km > 15.0:
            reward += 2.0 * self.dt  # Double reward when safe
        else:
            reward += self.dt
        
        # 2. Distance-based rewards (tiered system)
        if dist_km > 20.0:
            reward += 10.0  # HUGE bonus for staying far (>20km)
        elif dist_km > 15.0:
            reward += 5.0   # Good distance (15-20km)
        elif dist_km > 10.0:
            reward += 2.0   # Acceptable distance (10-15km)
        elif dist_km > 5.0:
            reward += 0.5   # Danger zone (5-10km)
        else:
            # Critical danger (<5km) - massive penalty
            danger_factor = (5.0 - dist_km) / 5.0
            reward -= 10.0 * danger_factor
        
        # 3. Reward aggressive maneuvering when threatened
        if dist_km < 12.0:
            accel_mag = np.linalg.norm(a_cmd)
            max_accel = sim.TARG_MAX_G * 9.81
            g_usage = accel_mag / max_accel
            reward += g_usage * 3.0  # Strong reward for pulling Gs
        
        # 4. Reward for perpendicular aspect (beaming)
        if dist_km < 15.0 and missile_speed > 1e-3:
            # Calculate angle between fighter velocity and LOS
            fwd = self.fighter_vel / max(fighter_speed, 1e-3)
            cos_aspect = abs(np.dot(fwd, los))
            # Reward being perpendicular (cos ≈ 0)
            beam_reward = (1.0 - cos_aspect) * 2.0
            reward += beam_reward
        
        # 5. Penalty for low altitude (might crash during maneuvers)
        alt = self.fighter_pos[2]

        # soft band: wants 9–13 km, hates going below 7 km,
        # REALLY hates going below 4 km.
        if alt < 7000.0:
            reward -= 5.0 * self.dt
        if alt < 5000.0:
            reward -= 20.0 * self.dt
        if alt < 3000.0:
            reward -= 50.0 * self.dt
        
        
        # 6. Death penalty (MUCH larger)
        if R < self.kill_radius:
            reward -= 500.0  # Increased from -100
            terminated = True
        
        # Time limit
        truncated = False
        if self.step_count >= self.max_steps:
            truncated = True
            # Bonus for surviving full episode
            reward += 100.0
        
        done = terminated or truncated
        info = {"R": R, "terminated": terminated, "truncated": truncated}
        return self._get_obs(), float(reward), done, info


# Import networks
from missile_intercept import FighterPolicyNet

class ValueNet(nn.Module):
    def __init__(self, obs_dim=12, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32, device=values.device)
    last_adv = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv
    returns = advantages + values[:-1]
    return advantages, returns


# ===========================================================================
# CONTINUE TRAINING FUNCTION
# ===========================================================================

def continue_training():
    """Continue training from existing checkpoints"""
    
    # Check if checkpoints exist
    if not os.path.exists(POLICY_CHECKPOINT):
        print(f"ERROR: {POLICY_CHECKPOINT} not found!")
        print("Train a model first using train_fighter.py")
        return
    
    if not os.path.exists(VALUE_CHECKPOINT):
        print(f"WARNING: {VALUE_CHECKPOINT} not found. Starting with fresh value network.")
        value_exists = False
    else:
        value_exists = True
    
    print("=" * 60)
    print("CONTINUING FIGHTER TRAINING")
    print("=" * 60)
    print(f"Loading policy from: {POLICY_CHECKPOINT}")
    if value_exists:
        print(f"Loading value from:  {VALUE_CHECKPOINT}")
    print(f"Training for:        {ADDITIONAL_STEPS:,} additional steps")
    print(f"Device:              {DEVICE}")
    print("=" * 60)
    
    # Setup
    env = FighterEnv(dt=0.02, max_time=40.0, seed=0)
    obs_dim = 12
    act_dim = 4
    
    # Determine network size from checkpoint
    checkpoint = torch.load(POLICY_CHECKPOINT, map_location=DEVICE)
    # Infer hidden_dim from first layer weight shape
    first_layer_key = 'net.0.weight'
    if first_layer_key in checkpoint:
        hidden_dim = checkpoint[first_layer_key].shape[0]
        print(f"Detected network size: hidden_dim={hidden_dim}")
    else:
        hidden_dim = 256  # Default
        print(f"Could not detect size, using hidden_dim={hidden_dim}")
    
    # Create networks
    policy = FighterPolicyNet(obs_dim=obs_dim, hidden_dim=hidden_dim).to(DEVICE)
    value_net = ValueNet(obs_dim=obs_dim, hidden_dim=hidden_dim).to(DEVICE)
    
    # Load policy weights
    policy.load_state_dict(checkpoint)
    policy.train()
    print(f"✓ Loaded policy weights")
    
    # Load value weights if available
    if value_exists:
        value_checkpoint = torch.load(VALUE_CHECKPOINT, map_location=DEVICE)
        value_net.load_state_dict(value_checkpoint)
        print(f"✓ Loaded value weights")
    else:
        print(f"✗ Starting with untrained value network")
    
    value_net.train()
    
    # Policy distribution (restore or create new)
    log_std = nn.Parameter(torch.zeros(act_dim, device=DEVICE))
    
    # Optimizer with optional learning rate override
    lr = OVERRIDE_LR if OVERRIDE_LR is not None else 3e-4
    params = list(policy.parameters()) + list(value_net.parameters()) + [log_std]
    optimizer = optim.Adam(params, lr=lr)
    print(f"✓ Optimizer created with lr={lr}")
    
    # Training config
    config = {
        "gamma": 0.99,
        "lam": 0.95,
        "clip_eps": 0.2,
        "entropy_coef": 0.01,
        "vf_coef": 0.5,
        "train_iters": 10,
        "minibatch_size": 4096,
        "steps_per_rollout": 16384,
    }
    
    # Apply overrides
    config.update(OVERRIDE_CONFIG)
    print(f"\nTraining config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    # Plotting
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ep_returns_plot = []
    plot_x = []
    
    global_step = 0
    episode_returns = []
    episode_lengths = []
    
    obs = env.reset()
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(DEVICE)
    t_global = 0.0
    
    print("Starting PPO training from checkpoint...")
    print("-" * 60)
    
    while global_step < ADDITIONAL_STEPS:
        # Rollout
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []
        
        steps_this_rollout = 0
        
        while steps_this_rollout < config["steps_per_rollout"] and global_step < ADDITIONAL_STEPS:
            with torch.no_grad():
                mu = policy(obs_t)
                std = log_std.exp().expand_as(mu)
                dist = torch.distributions.Normal(mu, std)
                action_t = dist.sample()
                logp_t = dist.log_prob(action_t).sum(-1)
                v_t = value_net(obs_t)
            
            action = action_t.cpu().numpy()[0]
            next_obs, reward, done, info = env.step(action, t_global)
            t_global += env.dt
            
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logp_t.cpu().numpy()[0])
            rew_buf.append(reward)
            done_buf.append(done)
            val_buf.append(v_t.cpu().numpy()[0])
            
            global_step += 1
            steps_this_rollout += 1
            
            if len(episode_returns) == 0:
                episode_returns.append(0.0)
                episode_lengths.append(0)
            
            episode_returns[-1] += reward
            episode_lengths[-1] += 1
            
            if done:
                obs = env.reset()
                episode_returns.append(0.0)
                episode_lengths.append(0)
            else:
                obs = next_obs
            
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(DEVICE)
        
        # Bootstrap
        with torch.no_grad():
            v_last = value_net(obs_t)
        val_buf.append(v_last.cpu().numpy()[0])
        
        # Convert to tensors
        obs_tensor = torch.from_numpy(np.array(obs_buf, dtype=np.float32)).to(DEVICE)
        act_tensor = torch.from_numpy(np.array(act_buf, dtype=np.float32)).to(DEVICE)
        logp_old_tensor = torch.from_numpy(np.array(logp_buf, dtype=np.float32)).to(DEVICE)
        rew_tensor = torch.from_numpy(np.array(rew_buf, dtype=np.float32)).to(DEVICE)
        done_tensor = torch.from_numpy(np.array(done_buf, dtype=np.bool_)).to(DEVICE)
        val_tensor = torch.from_numpy(np.array(val_buf, dtype=np.float32)).to(DEVICE)
        
        # GAE
        adv_tensor, ret_tensor = compute_gae(
            rew_tensor, val_tensor, done_tensor,
            gamma=config["gamma"], lam=config["lam"]
        )
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        
        # PPO updates
        dataset_size = len(obs_tensor)
        idxs = np.arange(dataset_size)
        
        for _ in range(config["train_iters"]):
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, config["minibatch_size"]):
                end = start + config["minibatch_size"]
                mb_idx = idxs[start:end]
                
                mb_obs = obs_tensor[mb_idx]
                mb_act = act_tensor[mb_idx]
                mb_logp_old = logp_old_tensor[mb_idx]
                mb_adv = adv_tensor[mb_idx]
                mb_ret = ret_tensor[mb_idx]
                
                mu = policy(mb_obs)
                std = log_std.exp().expand_as(mu)
                dist = torch.distributions.Normal(mu, std)
                
                logp = dist.log_prob(mb_act).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                v_pred = value_net(mb_obs)
                
                ratio = torch.exp(logp - mb_logp_old)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - config["clip_eps"], 
                                   1.0 + config["clip_eps"]) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (mb_ret - v_pred).pow(2).mean()
                
                loss = (actor_loss + config["vf_coef"] * critic_loss - 
                       config["entropy_coef"] * entropy)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Logging
        if len(episode_returns) > 0:
            mean_ret = float(np.mean(list(episode_returns)[-10:]))
            mean_len = float(np.mean(list(episode_lengths)[-10:]))
        else:
            mean_ret = 0.0
            mean_len = 0.0
        
        print(f"[CONTINUE] steps={global_step}/{ADDITIONAL_STEPS} | "
              f"return={mean_ret:.1f} | len={mean_len:.1f}")
        
        ep_returns_plot.append(mean_ret)
        plot_x.append(global_step)
        ax.clear()
        ax.plot(plot_x, ep_returns_plot)
        ax.set_xlabel("Additional env steps")
        ax.set_ylabel("Mean return (last 10 eps)")
        ax.set_title(f"Continued Training from {POLICY_CHECKPOINT}")
        ax.grid(True, alpha=0.3)
        plt.pause(0.01)
        
        # Save checkpoints
        torch.save(policy.state_dict(), POLICY_OUTPUT)
        torch.save(value_net.state_dict(), VALUE_OUTPUT)
    
    plt.ioff()
    plt.show()
    
    print("\n" + "=" * 60)
    print("CONTINUED TRAINING COMPLETE")
    print("=" * 60)
    print(f"Saved policy to:     {POLICY_OUTPUT}")
    print(f"Saved value net to:  {VALUE_OUTPUT}")
    print(f"Final mean return:   {mean_ret:.1f}")
    print(f"Final mean length:   {mean_len:.1f}")
    print("=" * 60)
    
    # Optionally overwrite original checkpoint
    response = input("\nOverwrite original checkpoint? (y/n): ").strip().lower()
    if response == 'y':
        torch.save(policy.state_dict(), POLICY_CHECKPOINT)
        torch.save(value_net.state_dict(), VALUE_CHECKPOINT)
        print(f"✓ Overwrote {POLICY_CHECKPOINT} and {VALUE_CHECKPOINT}")
    else:
        print(f"✓ Kept separate checkpoint: {POLICY_OUTPUT}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    continue_training()