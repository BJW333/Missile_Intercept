"""
3D Missile–Target Pursuit Simulation (with Evasion, Decoys, PN Guidance, and Seeker Confusion)

This script simulates:
- A target aircraft flying a piecewise path: straight -> turn -> straight.
- Evasive "jink" maneuvers overlaid on the base trajectory.
- Decoys that are deployed mid-flight and drift away.
- A missile that:
    - Launches from a fixed position.
    - Uses a noisy signal-based seeker that can lock onto real target OR decoys.
    - Uses Proportional Navigation (PN) guidance to steer toward whatever it's locked on.
    - Can intercept the real target OR be fooled and intercept a decoy.

Visualization:
- 3D plot with ground plane.
- Target, missile, and decoy trajectories.
- HUD showing:
    - Time
    - Distance to REAL target
    - Current seeker lock (real vs decoy)
    - Intercept status (real hit, decoy hit, or none)

This file also supports:
- Randomized starting positions for aircraft and missile.
- A Monte Carlo loop to estimate probabilities of:
    - real target intercept,
    - decoy intercept,
    - complete miss.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # ensures 3D projection is registered

# ============================================================================
# GLOBAL SIMULATION PARAMETERS
# ============================================================================

#GRAVITY
GRAVITY = np.array([0.0, 0.0, -9.81])  # m/s²

# --- Target trajectory timing ---
Straight_time = 25       # Duration of first straight segment (s)
curve_time = 25          # Duration of the turning (curved) segment (s)
Straight_time2 = 25      # Duration of second straight segment (s)

# --- Speeds ---
targ_vel = 300           # Target true airspeed (m/s), magnitude only, direction. Current measurement: Subsonic aircraft (~Mach 0.9)
                         # is given by the piecewise geometry below. 
miss_vel = 1200          # Missile speed (m/s); for simplicity, assumed constant magnitude.
                         # Missile (~Mach 3.5)
                         
# --- Turn geometry ---
turn_angle = -np.pi * 4 / 3   # Total turn angle in radians; sign defines turn direction.
tmax = 75                     # Total simulated time (s); after this, sim stops.
dt = 0.001                    # Time step for numerical integration (s).
animation_interval = 5        # Milliseconds between animation frames (for plotting only).

# --- 3D orientation of the turn plane ---
yz_angle = -np.pi / 12        # Tilt of the turning plane into YZ plane (radians).
                              # A nonzero tilt makes the turn truly 3D instead of flat in XY.

# --- Default start positions (used if you don't randomize) ---
aircraft_start_loc = np.array([0.0, 0.0, 12000.0])   # Default aircraft initial position (m).
missile_start_loc = np.array([13000.0, 12000.0, 0.0])  # Default missile launch position (m).

# --- Missile launch & kill radius ---
missile_launch_time = 0.0       # Time when missile is allowed to start moving (s).
kill_dist = 15.0  # Kill radius (m); Typical proximity fuze lethal radius (m) if within this, we treat as intercept.

# --- Extra vertical shaping of the curve segment ---
climb_rate_curve = -0.001       # Vertical shaping factor applied only during turn segment.
                                # Controls climb/dive behavior during the curve.


# ============================================================================
# EVASIVE MANEUVER PARAMETERS
# ============================================================================

t_evasion_start = 30.0    # Time when aircraft begins sinusoidal evasive "jink" (s).
jink_amp = 500.0          # Amplitude of jink maneuver (m).
jink_freq = 0.2           # Frequency of jink (Hz); 0.2 Hz ~ one cycle every 5 seconds.


# ============================================================================
# DECOY PARAMETERS
# ============================================================================

n_decoys = 3               # Number of decoys the aircraft will release.
decoy_deploy_time = 35.0   # Time when decoys are deployed (s).
decoy_base_speed = 200.0   # Average drift speed of each decoy after deployment (m/s).
decoy_speed_sigma = 50.0   # Standard deviation around base speed for decoy drift (m/s).


# ============================================================================
# SEEKER / SENSOR MODEL PARAMETERS
# ============================================================================

true_target_rcs = 1.0    # Relative signal strength (RCS-like) of real target.
decoy_rcs = 0.8          # Relative signal strength of decoys (slightly weaker).
noise_std = 0.05         # Standard deviation of Gaussian noise in seeker signal model.


# ============================================================================
# GUIDANCE PARAMETERS (Proportional Navigation)
# ============================================================================

N_pn = 3.0   # Navigation constant N (Pn ~ 2–5 in many real systems).
             # Higher N => more aggressive lateral acceleration toward LOS rate.


# ============================================================================
# CAMERA / VIEW SETTINGS
# ============================================================================

base_azim = 45           # Initial azimuth angle (deg) of 3D view.
base_elev = 20           # Initial elevation angle (deg) of 3D view.
camera_rate = 0.03       # Rate of azimuth change for camera orbit (unused here but kept).


# ============================================================================
# GLOBALS FOR TARGET GEOMETRY
# (Used by target_location() to remember where each segment starts)
# ============================================================================

# These hold the starting positions and centers for each piecewise segment.
curve_start_x = None
curve_start_y = None
curve_start_z = None
curve_initialized = False    # Flags to ensure curve start is only computed once.

straight2_start_x = None
straight2_start_y = None
straight2_start_z = None
straight2_initialized = False

# Radius of the turn derived from: arc_length = speed * time = radius * |angle|.
# sign of turn_angle is handled in param, so we use full turn_angle here.
radius = (targ_vel * curve_time) / abs(turn_angle)
center_x = None
center_y = None
center_z = None


# ============================================================================
# RANDOMIZED START HELPERS
# ============================================================================

def sample_random_starts(rng):
    """
    Sample random starting positions for aircraft and missile using some
    simple geometry constraints.

    Aircraft:
      - Starts at XY ≈ (0, 0).
      - Altitude uniformly in [9 km, 13 km].

    Missile:
      - Starts at a random bearing around the aircraft.
      - Horizontal range uniformly in [15 km, 30 km].
      - Altitude uniformly in [0, 5 km].

    Parameters
    ----------
    rng : np.random.Generator
        Numpy random number generator instance.

    Returns
    -------
    aircraft_start_loc, missile_start_loc : np.ndarray, np.ndarray
        3D start positions for aircraft and missile respectively.
    """

    # Aircraft position
    aircraft_alt = rng.uniform(9000.0, 13000.0) # altitude band.
    
    r_ac = rng.uniform(0.0, 5000.0)  # Smaller spread # up to 15 km from origin
    bearing_ac = rng.uniform(0.0, 2*np.pi)  # random bearing around origin
    
    ax0 = r_ac * np.cos(bearing_ac) # X coord
    ay0 = r_ac * np.sin(bearing_ac) # Y coord
    
    aircraft_start = np.array([ax0, ay0, aircraft_alt]) #Aircraft start pos.
    
    # Missile: Bias toward forward hemisphere (head-on to beam aspects)
    r = rng.uniform(10000.0, 25000.0) # radial distance from aircraft.
    # Favor intercept angles: +/- 90° from aircraft heading (+X direction)
    
    bearing = rng.uniform(-np.pi/2, np.pi/2) + np.pi  # # azimuth angle around aircraft. # Ahead of aircraft
    miss_alt = rng.uniform(5000.0, 15000.0)  # random missile altitude. # Similar altitude band
    
    miss_x = aircraft_start[0] + r * np.cos(bearing)
    miss_y = aircraft_start[1] + r * np.sin(bearing)
    miss_z = miss_alt
    missile_start = np.array([miss_x, miss_y, miss_z])
    
    return aircraft_start, missile_start

# ============================================================================
# TARGET TRAJECTORY FUNCTION
# ============================================================================

def target_location(t, prev_state, _index):
    """
    Compute the target aircraft position at time t.

    The trajectory is divided into segments:

      1. SEGMENT 1: Straight line in +X direction.
      2. SEGMENT 2: Circular arc (turn) in a tilted plane.
      3. SEGMENT 3: Straight line in the final heading direction after turn.
      4. SEGMENT 4: After path finishes, continue straight or hold final.

    On top of that base path, we overlay an evasive sinusoidal jink (after
    t_evasion_start).

    Parameters
    ----------
    t : float
        Current simulation time (s).
    prev_state : np.ndarray or None
        Previous target position [x, y, z], or None if this is the first point.
    index : int
        Current time index (unused, kept for potential future use).
        
    Returns
    -------
    pos : np.ndarray
        Current target position [x, y, z] at time t (m).
    """
    global curve_start_x, curve_start_y, curve_start_z, curve_initialized
    global straight2_start_x, straight2_start_y, straight2_start_z, straight2_initialized
    global center_x, center_y, center_z
    global aircraft_start_loc

    # --- SEGMENT 1: First straight flight in +X direction ---
    if 0 <= t <= Straight_time:
        # Straight motion along +X; Y,Z fixed at initial aircraft start.
        x = aircraft_start_loc[0] + targ_vel * t
        y = aircraft_start_loc[1]
        z = aircraft_start_loc[2]

    # --- SEGMENT 2: Curved turn (circular arc in a tilted plane) ---
    elif Straight_time < t <= Straight_time + curve_time:
        # Time elapsed since entering the curve.
        tc = t - Straight_time

        # Initialize curve center and starting point *once* on first entry.
        if not curve_initialized:
            if prev_state is not None:
                curve_start_x = prev_state[0]
                curve_start_y = prev_state[1]
                curve_start_z = prev_state[2]
            else:
                curve_start_x = aircraft_start_loc[0] + targ_vel * Straight_time
                curve_start_y = aircraft_start_loc[1]
                curve_start_z = aircraft_start_loc[2]

            # Define center of circular arc for the turn.
            # Center is offset in YZ plane by radius, with tilt yz_angle.
            center_x = curve_start_x
            center_y = curve_start_y + radius * np.cos(yz_angle)
            center_z = curve_start_z + radius * np.sin(yz_angle)

            curve_initialized = True

        # Fraction of turn angle completed so far.
        angle = tc * turn_angle / curve_time

        # Parameter angle for circle. Starting from -π/2 so that the circle
        # tangent initially points along +X.
        arc_angle = -np.pi / 2 + angle

        # Compute base circular path coordinates.
        x = center_x + radius * np.cos(arc_angle)

        # Vertical shaping term to give some extra climb/dive over the arc.
        shaping = targ_vel**2 * (1 - np.cos(np.pi * tc / curve_time)) * climb_rate_curve

        # Project the circular motion into YZ with tilt yz_angle.
        y = center_y + radius * np.sin(arc_angle) * np.cos(yz_angle) \
            + np.cos(yz_angle + np.pi / 2) * shaping
        z = center_z + radius * np.sin(arc_angle) * np.sin(yz_angle) \
            + np.sin(yz_angle + np.pi / 2) * shaping

    # --- SEGMENT 3: Second straight segment after the turn ---
    elif Straight_time + curve_time < t <= Straight_time + curve_time + Straight_time2:
        # Initialize second straight start point once (the end of the curve).
        if not straight2_initialized:
            if prev_state is not None:
                straight2_start_x = prev_state[0]
                straight2_start_y = prev_state[1]
                straight2_start_z = prev_state[2]
            else:
                straight2_start_x = curve_start_x
                straight2_start_y = curve_start_y
                straight2_start_z = curve_start_z
            straight2_initialized = True

        # Time since entering second straight.
        ts = t - (Straight_time + curve_time)

        # Unit direction vector of second straight: rotated by turn_angle + tilt.
        dx = np.cos(turn_angle)
        dy = np.sin(turn_angle) * np.cos(yz_angle)
        dz = np.sin(turn_angle) * np.sin(yz_angle)

        # Straight motion along that heading.
        x = straight2_start_x + targ_vel * ts * dx
        y = straight2_start_y + targ_vel * ts * dy
        z = straight2_start_z + targ_vel * ts * dz

    # --- SEGMENT 4: After defined path, just continue along or hold ---
    else:
        if straight2_initialized:
            # Continue along direction of second straight for its full duration.
            ts_max = Straight_time2
            dx = np.cos(turn_angle)
            dy = np.sin(turn_angle) * np.cos(yz_angle)
            dz = np.sin(turn_angle) * np.sin(yz_angle)
            x = straight2_start_x + targ_vel * ts_max * dx
            y = straight2_start_y + targ_vel * ts_max * dy
            z = straight2_start_z + targ_vel * ts_max * dz
        else:
            # Fallback: if second straight never initialized, hold end of first straight.
            x = aircraft_start_loc[0] + targ_vel * Straight_time
            y = aircraft_start_loc[1]
            z = aircraft_start_loc[2]

    # --- EVASIVE MANEUVER OVERLAY (JINK) ---
    # After t_evasion_start, we add a sinusoidal lateral offset in the
    # direction defined by yz_angle to simulate "jinking".
    if t > t_evasion_start:
        j = jink_amp * np.sin(2 * np.pi * jink_freq * (t - t_evasion_start))
        # Apply jink in the tilted plane direction.
        y += j * np.cos(yz_angle)
        z += j * np.sin(yz_angle)

    return np.array([x, y, z])


# ============================================================================
# CORE SIMULATION FUNCTION
# ============================================================================

def simulate_single_engagement(
    aircraft_start_loc_input,
    missile_start_loc_input,
    rng=None,
    make_plot=False,
    verbose=True,
    run_id=None,        # 1-based run index
    total_runs=None,    # total number of runs
):
    """
    Run a single missile–target engagement with given start positions.

    Steps:
    1. Set global aircraft/missile start positions.
    2. Generate full aircraft trajectory.
    3. Spawn decoys at decoy_deploy_time and simulate their drift.
    4. Simulate missile flight using a seeker and PN guidance:
       - Each timestep, missile seeker picks lock among (target + active decoys)
         based on signal strength ~ RCS / distance^2 + noise.
       - PN guidance steers missile based on line-of-sight rate to locked object.
       - Check for intercept vs real target or decoys (within kill_dist).
    5. Optionally create a 3D animation of the engagement.

    Parameters
    ----------
    aircraft_start_loc_input : np.ndarray, shape (3,)
        Starting position of the aircraft.
    missile_start_loc_input : np.ndarray, shape (3,)
        Starting position of the missile.
    rng : np.random.Generator or None
        Numpy random number generator instance. If None, creates a new one.
    make_plot : bool
        Whether to produce 3D animated visualization.
    verbose : bool
        Whether to print debug/intercept info.

    Returns
    -------
    result : dict
        A summary of the engagement outcome:
        {
          'intercept_real': bool,
          'intercept_decoy': bool,
          'intercept_type': 'real' | 'decoy' | None,
          'intercept_time': float or None,
          'closest_miss': float  # min distance from missile to REAL target
        }
    """
    global aircraft_start_loc, missile_start_loc
    global curve_start_x, curve_start_y, curve_start_z, curve_initialized
    global straight2_start_x, straight2_start_y, straight2_start_z, straight2_initialized
    global center_x, center_y, center_z

    # Override global start positions for this run.
    aircraft_start_loc = aircraft_start_loc_input.astype(float).copy()
    missile_start_loc = missile_start_loc_input.astype(float).copy()

    # Reset all segment/center state so target_location() recomputes for this run.
    curve_start_x = curve_start_y = curve_start_z = None
    straight2_start_x = straight2_start_y = straight2_start_z = None
    center_x = center_y = center_z = None
    curve_initialized = False
    straight2_initialized = False

    # Seed RNG to make decoy directions + seeker noise reproducible if desired.
    if rng is None:
        rng = np.random.default_rng()

    # ----------------------------------------------------------------------
    # Generate target trajectory for full simulation duration.
    # ----------------------------------------------------------------------
    times = np.arange(0, tmax, dt)
    n_points = len(times)

    target_states = np.zeros((n_points, 3))
    for i in range(n_points):
        t = times[i]
        prev_state = target_states[i-1] if i > 0 else None 
        target_states[i] = target_location(t, prev_state, i)

    if verbose:
        print(f"Generated {n_points} trajectory points over {tmax:.1f} seconds")
        print(f"Aircraft start: ({aircraft_start_loc[0]:.1f}, "
              f"{aircraft_start_loc[1]:.1f}, {aircraft_start_loc[2]:.1f})")
        print(f"Curve start: ({curve_start_x:.1f}, {curve_start_y:.1f}, {curve_start_z:.1f})")
        print(f"Straight2 start: ({straight2_start_x:.1f}, "
              f"{straight2_start_y:.1f}, {straight2_start_z:.1f})")

    # ----------------------------------------------------------------------
    # Generate decoy trajectories
    # ----------------------------------------------------------------------
    # decoy_states[d, i, :] = position of decoy d at time index i
    decoy_states = np.zeros((n_decoys, n_points, 3))
    # decoy_active[d, i] = True if decoy d exists and is active at time i
    decoy_active = np.zeros((n_decoys, n_points), dtype=bool)

    for d in range(n_decoys):
        deployed = False

        # Random direction for decoy drift (normalized) and random speed.
        drift_dir = rng.standard_normal(3)
        norm = np.linalg.norm(drift_dir)
        if norm < 1e-9:
            drift_dir = np.array([1.0, 0.0, 0.0])
        else:
            drift_dir /= norm
        drift_speed = max(0.0, decoy_base_speed + decoy_speed_sigma * rng.standard_normal())


        for i, t in enumerate(times):
            if t < decoy_deploy_time:
                # Before deployment: mark as inactive and set NaN positions for clarity.
                decoy_active[d, i] = False
                decoy_states[d, i] = np.nan
            else:
                if not deployed:
                    # At the deployment time, spawn decoy at current target position.
                    deployed = True
                    decoy_states[d, i] = target_states[i]
                    decoy_active[d, i] = True
                else:
                    # After deployment, decoy just drifts linearly along drift_dir.
                    decoy_states[d, i] = decoy_states[d, i - 1] + drift_dir * drift_speed * dt
                    decoy_active[d, i] = True

    if verbose:
        print(f"Decoys deployed at t = {decoy_deploy_time:.1f}s")

    # ----------------------------------------------------------------------
    # Missile simulation with seeker, decoy confusion & PN guidance
    # ----------------------------------------------------------------------

    # missile_states[i, :] = missile position at time index i
    missile_states = np.zeros((n_points, 3))
    missile_states[0] = missile_start_loc

    # missile_vel[i, :] = missile velocity at time index i
    missile_vel = np.zeros((n_points, 3))

    # Initial missile velocity: aim directly at target at t=0 (pure pursuit start).
    rel0 = target_states[0] - missile_start_loc
    d0 = np.linalg.norm(rel0)
    if d0 > 1e-6:
        missile_vel[0] = miss_vel * (rel0 / d0)
    else:
        # Degenerate case: if target and missile start at same point, just pick +X.
        missile_vel[0] = np.array([miss_vel, 0.0, 0.0])

    missile_launched = False  # becomes True once t >= missile_launch_time

    # Intercept bookkeeping
    intercept_time = None
    intercept_index = None
    intercept_type = None
    intercept_decoy_id = None

    intercepted = False       # True once missile has killed something.
    intercept_real = False    # True if real target was killed.
    intercept_decoy = False   # True if a decoy was killed.

    # lock_history[i] stores index of currently locked object:
    #   -1 = no lock
    #    0 = real target
    #   1..n_decoys = decoys
    lock_history = -np.ones(n_points, dtype=int)

    # Track closest approach of missile to REAL target (for miss distance).
    closest_miss = np.inf

    # Previous line-of-sight unit vector (for LOS rate in PN).
    prev_los = None

    # --- MAIN TIME STEP LOOP ---
    for i in range(1, n_points):
        t = times[i]

        # --- Handle missile launch time ---
        if t >= missile_launch_time and not missile_launched:
            missile_launched = True
            if verbose:
                print(f"Missile launched at t = {t:.2f}s")

        if not missile_launched:
            # Before launch, missile stays static at start location.
            missile_states[i] = missile_start_loc
            #missile_vel[i] = np.zeros(3) #old line
            missile_vel[i] = missile_vel[i-1] + GRAVITY * dt #new line accounting for gravity 
            lock_history[i] = -1
            continue

        if intercepted:
            # After intercept, missile is "dead": freeze position and lock state.
            missile_states[i] = missile_states[i - 1]
            missile_vel[i] = missile_vel[i - 1]
            lock_history[i] = lock_history[i - 1]
            continue

        # --- Build seeker candidate list: real target + any active decoys ---
        candidates_pos = [target_states[i]]       # position of real target at time step i
        candidates_rcs = [true_target_rcs]        # corresponding RCS (signal strength scale)

        # Append all active decoys to candidate list.
        for d in range(n_decoys):
            if decoy_active[d, i]:
                candidates_pos.append(decoy_states[d, i])
                candidates_rcs.append(decoy_rcs)

        # --- Seeker chooses lock based on noisy signal strength ---
        # We loop through each candidate and compute:
        #   signal = rcs / dist^2 + Gaussian noise
        # Then lock on to the highest signal.
        best_signal = -np.inf
        best_idx = 0
        best_distance = None
        best_rel = None

        for idx, (pos, rcs) in enumerate(zip(candidates_pos, candidates_rcs)):
            rel = pos - missile_states[i - 1]      # vector from missile to object
            dist = np.linalg.norm(rel)            # distance
            if dist < 1e-6:
                # Skip degenerate cases where positions coincide exactly.
                continue

            # Simple inverse-square law with additive Gaussian noise.\
            signal = rcs / (dist**2) + rng.normal(0.0, noise_std)

            # Pick whichever object generates maximal signal.
            if signal > best_signal:
                best_signal = signal
                best_idx = idx
                best_distance = dist
                best_rel = rel

        # Save lock index (0=real target, 1..n_decoys decoys, -1=none)
        lock_history[i] = best_idx

        # --- Track closest miss vs REAL target (for stats) ---
        dist_to_true = np.linalg.norm(target_states[i] - missile_states[i - 1])
        if dist_to_true < closest_miss:
            closest_miss = dist_to_true

        # --- Intercept checks ---
        # 1) Check kill vs REAL target using actual distance to real target.
        if dist_to_true < kill_dist and not intercepted:
            intercept_time = t
            intercept_index = i
            intercept_type = 'real'
            intercept_real = True
            intercepted = True
            missile_states[i] = missile_states[i - 1]
            missile_vel[i] = missile_vel[i - 1]
            if verbose:
                print(f"INTERCEPT REAL TARGET at t = {t:.2f}s, distance = {dist_to_true:.2f} m")
            continue

        # 2) Check kill vs DECOY, but only if currently locked onto a decoy
        #    (best_idx != 0) and within kill_dist of that locked object.
        if best_idx != 0 and best_distance is not None and best_distance < kill_dist and not intercepted:
            intercept_time = t
            intercept_index = i
            intercept_type = 'decoy'
            intercept_decoy_id = best_idx - 1  # adjust index: decoys start at 0
            intercept_decoy = True
            intercepted = True
            missile_states[i] = missile_states[i - 1]
            missile_vel[i] = missile_vel[i - 1]
            if verbose:
                print(f"MISSILE FOOLED BY DECOY {intercept_decoy_id + 1} at t = {t:.2f}s, "
                      f"distance = {best_distance:.2f} m")
            continue

        # --- PN GUIDANCE toward currently locked object (if any) ---
        if best_distance is not None and best_distance > 0:
            # LOS unit vector from missile to locked object.
            los = best_rel / best_distance

            if prev_los is None:
                # On first step with a lock, just set velocity along LOS
                # (pure pursuit initial condition).
                missile_vel[i] = miss_vel * los
            else:
                # Approximate line-of-sight rate (angular rate) as difference
                # in LOS unit vectors over dt.
                los_rate = (los - prev_los) / dt

                # Current missile velocity direction.
                v_prev = missile_vel[i - 1]
                v_prev_norm = np.linalg.norm(v_prev)
                if v_prev_norm < 1e-6:
                    # If velocity is almost zero, we define direction as LOS.
                    v_dir = los
                else:
                    v_dir = v_prev / v_prev_norm

                # Component of LOS rate perpendicular to velocity direction.
                # PN only uses this perpendicular component for lateral acceleration.
                los_rate_perp = los_rate - np.dot(los_rate, v_dir) * v_dir
                
                los_rate_perp_norm = np.linalg.norm(los_rate_perp)

                # PN acceleration magnitude ~ N * V * |LOS_rate_perp|
                max_accel = 40.0 * 9.81  # 40g limit in m/s²
                a_mag = N_pn * miss_vel * los_rate_perp_norm
                a_mag = min(a_mag, max_accel)
                
                # Direction of lateral acceleration.
                if los_rate_perp_norm > 1e-9:
                    a_dir = los_rate_perp / los_rate_perp_norm
                else:
                    a_dir = np.zeros(3)

                # Lateral acceleration vector.
                a = a_mag * a_dir

                # Velocity update: v_new = v_prev + a * dt
                missile_vel[i] = v_prev + a * dt

                # Enforce constant missile speed (normalize back to miss_vel).
                speed = np.linalg.norm(missile_vel[i])
                if speed > 1e-9:
                    missile_vel[i] = (missile_vel[i] / speed) * miss_vel
                else:
                    # If degenerate, just align with LOS again.
                    missile_vel[i] = miss_vel * los

            # Update missile position using updated velocity.
            missile_states[i] = missile_states[i - 1] + missile_vel[i] * dt
            # Store current LOS for next step's rate computation.
            prev_los = los
        else:
            # If seeker has no lock, coast straight ahead.
            missile_vel[i] = missile_vel[i - 1]
            missile_states[i] = missile_states[i - 1] + missile_vel[i] * dt
            prev_los = None  # Reset LOS since we lost lock

    # Final distance to real target at tmax (useful for "miss" scenario).
    final_distance = np.linalg.norm(target_states[-1] - missile_states[-1])
    if verbose:
        print(f"Final distance to REAL target: {final_distance:.2f} m")
        if intercept_real:
            print(f"RESULT: REAL target intercepted at t = {intercept_time:.2f}s "
                  f"(index {intercept_index})")
        elif intercept_decoy:
            print(f"RESULT: Missile wasted on decoy {intercept_decoy_id + 1} at "
                  f"t = {intercept_time:.2f}s (index {intercept_index})")
        else:
            print(f"RESULT: NO INTERCEPT. Closest miss distance = {closest_miss:.2f} m")

    # Package high-level results in a dict.
    result = {
        "intercept_real": intercept_real,
        "intercept_decoy": intercept_decoy,
        "intercept_type": intercept_type,
        "intercept_time": intercept_time,
        "closest_miss": closest_miss,
    }

    # ----------------------------------------------------------------------
    # Visualization (optional) – 3D animated plot of one engagement.
    # ----------------------------------------------------------------------
    if make_plot:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Collect all points (target, missile, decoys) to compute global bounds.
        all_points = np.vstack([target_states, missile_states])
        for d in range(n_decoys):
            valid = np.isfinite(decoy_states[d, :, 0])
            if np.any(valid):
                all_points = np.vstack([all_points, decoy_states[d, valid]])

        # Compute ranges and a symmetric bounding box around the scene.
        padding = 0.1
        x_range = np.ptp(all_points[:, 0])
        y_range = np.ptp(all_points[:, 1])
        z_range = np.ptp(all_points[:, 2])
        max_range = max(x_range, y_range, z_range)

        x_center = (np.max(all_points[:, 0]) + np.min(all_points[:, 0])) / 2
        y_center = (np.max(all_points[:, 1]) + np.min(all_points[:, 1])) / 2
        z_center = (np.max(all_points[:, 2]) + np.min(all_points[:, 2])) / 2
        plot_radius = max_range / 2 * (1 + padding)

        # Apply symmetric limits and equal aspect ratio.
        ax.set_xlim(x_center - plot_radius, x_center + plot_radius)
        ax.set_ylim(y_center - plot_radius, y_center + plot_radius)
        ax.set_zlim(z_center - plot_radius, z_center + plot_radius)
        ax.set_box_aspect([1, 1, 1])

        # Label axes and set title.
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Missile–Target Pursuit with Evasion, Decoys & PN Guidance')
        ax.grid(True)
        ax.view_init(elev=base_elev, azim=base_azim)

        # Draw a translucent ground plane at Z=0 for context.
        ground_size = max_range * 1.2
        gx = np.linspace(x_center - ground_size / 2, x_center + ground_size / 2, 10)
        gy = np.linspace(y_center - ground_size / 2, y_center + ground_size / 2, 10)
        GX, GY = np.meshgrid(gx, gy)
        GZ = np.zeros_like(GX)
        ax.plot_surface(GX, GY, GZ, alpha=0.15, linewidth=0, antialiased=True)

        # Draw a small kill sphere around missile start position (for visualizing kill_dist).
        phi = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, 2 * np.pi, 40)
        PHI, THETA = np.meshgrid(phi, theta)
        kx = missile_start_loc[0] + kill_dist * np.sin(PHI) * np.cos(THETA)
        ky = missile_start_loc[1] + kill_dist * np.sin(PHI) * np.sin(THETA)
        kz = missile_start_loc[2] + kill_dist * np.cos(PHI)
        ax.plot_wireframe(kx, ky, kz, alpha=0.15, linewidth=0.5)

        # Plot the full target path as a dashed reference line.
        ax.plot(
            target_states[:, 0], target_states[:, 1], target_states[:, 2],
            linestyle='--', linewidth=1, alpha=0.3, label='Target Path'
        )

        # Moving artists for target, missile, and decoys.
        target_point, = ax.plot([], [], [], 'bo', markersize=8, label='Aircraft')
        target_trail, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.7, label='Aircraft Trail')

        missile_point, = ax.plot([], [], [], 'ro', markersize=6, label='Missile')
        missile_trail, = ax.plot([], [], [], 'r-', linewidth=1.5, alpha=0.7, label='Missile Trail')

        decoy_points = []
        for d in range(n_decoys):
            label = 'Decoys' if d == 0 else None  # Only one legend entry for all decoys.
            p, = ax.plot([], [], [], 'yx', markersize=6, alpha=0.8, label=label)
            decoy_points.append(p)

        # Show start positions for aircraft and missile.
        ax.scatter(
            target_states[0, 0], target_states[0, 1], target_states[0, 2],
            c='green', s=80, marker='s', label='Aircraft Start'
        )
        ax.scatter(
            missile_start_loc[0], missile_start_loc[1], missile_start_loc[2],
            c='orange', s=80, marker='^', label='Missile Start'
        )

        # Mark intercept location if any.
        if intercept_index is not None and intercept_type == 'real':
            # Intercept at real target position.
            ax.scatter(
                target_states[intercept_index, 0],
                target_states[intercept_index, 1],
                target_states[intercept_index, 2],
                c='red', s=200, marker='*', label='Real Intercept'
            )
        elif intercept_index is not None and intercept_type == 'decoy':
            # Intercept at decoy position.
            d_id = intercept_decoy_id
            ax.scatter(
                decoy_states[d_id, intercept_index, 0],
                decoy_states[d_id, intercept_index, 1],
                decoy_states[d_id, intercept_index, 2],
                c='magenta', s=200, marker='*', label=f'Decoy Hit (#{d_id + 1})'
            )

        ax.legend()

        # HUD text overlays: show time, speeds, distances, and intercept/lock status.
        time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
        speed_text = ax.text2D(0.02, 0.90, '', transform=ax.transAxes, fontsize=10)
        distance_text = ax.text2D(0.02, 0.85, '', transform=ax.transAxes, fontsize=10)
        
        #Run counter in top-left
        if run_id is not None:
            if total_runs is not None:
                run_label = f'Run {run_id} / {total_runs}'
            else:
                run_label = f'Run {run_id}'
        else:
            run_label = ''

        run_text = ax.text2D(0.02, 0.98, run_label,
                            transform=ax.transAxes,
                            fontsize=12,
                            fontweight='bold')
        
        
        intercept_status_text = ax.text2D(
            0.70, 0.95, '', transform=ax.transAxes, fontsize=12, color='red'
        )
        lock_status_text = ax.text2D(
            0.70, 0.90, '', transform=ax.transAxes, fontsize=10, color='purple'
        )

        def init_anim():
            """
            Initialize all animated artists with empty data for frame 0.
            """
            target_point.set_data([], [])
            target_point.set_3d_properties([])

            target_trail.set_data([], [])
            target_trail.set_3d_properties([])

            missile_point.set_data([], [])
            missile_point.set_3d_properties([])

            missile_trail.set_data([], [])
            missile_trail.set_3d_properties([])

            for p in decoy_points:
                p.set_data([], [])
                p.set_3d_properties([])

            # Clear HUD text
            time_text.set_text('')
            speed_text.set_text('')
            distance_text.set_text('')
            intercept_status_text.set_text('')
            lock_status_text.set_text('')

            return (
                target_point, target_trail,
                missile_point, missile_trail,
                *decoy_points,
                time_text, speed_text, distance_text,
                intercept_status_text, lock_status_text,
                run_text
            )

        def update(frame):
            """
            Update callback for FuncAnimation. 'frame' is an index into time array.
            """
            # --- Target: point + trail up to current frame ---
            target_point.set_data([target_states[frame, 0]], [target_states[frame, 1]])
            target_point.set_3d_properties([target_states[frame, 2]])

            target_trail.set_data(target_states[:frame + 1, 0], target_states[:frame + 1, 1])
            target_trail.set_3d_properties(target_states[:frame + 1, 2])

            # --- Missile: point + trail ---
            missile_point.set_data([missile_states[frame, 0]], [missile_states[frame, 1]])
            missile_point.set_3d_properties([missile_states[frame, 2]])

            missile_trail.set_data(missile_states[:frame + 1, 0], missile_states[:frame + 1, 1])
            missile_trail.set_3d_properties(missile_states[:frame + 1, 2])

            # --- Color missile based on current seeker lock ---
            lock_idx = lock_history[frame]
            if lock_idx == 0:
                missile_point.set_color('r')   # locked on REAL target
            elif lock_idx > 0:
                missile_point.set_color('m')   # locked on decoy
            else:
                missile_point.set_color('k')   # no lock

            # --- Decoys: show active ones only ---
            for d in range(n_decoys):
                if decoy_active[d, frame]:
                    decoy_points[d].set_data(
                        [decoy_states[d, frame, 0]],
                        [decoy_states[d, frame, 1]]
                    )
                    decoy_points[d].set_3d_properties(
                        [decoy_states[d, frame, 2]]
                    )
                else:
                    decoy_points[d].set_data([], [])
                    decoy_points[d].set_3d_properties([])

            # --- HUD: target speed (approx), missile–target distance, time ---
            if frame > 0:
                # Approximate instantaneous speed with finite difference.
                dx = target_states[frame, 0] - target_states[frame - 1, 0]
                dy = target_states[frame, 1] - target_states[frame - 1, 1]
                dz = target_states[frame, 2] - target_states[frame - 1, 2]
                speed = np.sqrt(dx**2 + dy**2 + dz**2) / dt
            else:
                speed = targ_vel  # fallback nominal speed.

            distance = np.linalg.norm(target_states[frame] - missile_states[frame])

            time_text.set_text(f'Time = {times[frame]:.2f} s')
            speed_text.set_text(f'Target Speed ≈ {speed:.1f} m/s')
            distance_text.set_text(f'Distance to REAL Target = {distance:.1f} m')

            # Intercept message, only after intercept_index.
            if intercept_index is not None and frame >= intercept_index:
                if intercept_type == 'real':
                    intercept_status_text.set_text(
                        f'INTERCEPT REAL TARGET @ t = {intercept_time:.2f}s'
                    )
                    missile_point.set_marker('x')
                elif intercept_type == 'decoy':
                    intercept_status_text.set_text(
                        f'MISSILE FOOLED BY DECOY #{intercept_decoy_id + 1} '
                        f'@ t = {intercept_time:.2f}s'
                    )
                    missile_point.set_marker('x')
            else:
                intercept_status_text.set_text('')

            # Lock text
            if lock_idx == -1:
                lock_status_text.set_text('Lock: NONE')
            elif lock_idx == 0:
                lock_status_text.set_text('Lock: REAL TARGET')
            else:
                lock_status_text.set_text(f'Lock: DECOY #{lock_idx}')

            return (
                target_point, target_trail,
                missile_point, missile_trail,
                *decoy_points,
                time_text, speed_text, distance_text,
                intercept_status_text, lock_status_text,
                run_text
            )

        # Subsample frames for animation (500 frames max-ish).
        frame_skip = max(1, len(times) // 500)
        frames = range(0, len(times), frame_skip)

        num_frames = (len(times) - 1) // frame_skip + 1
        print(f"Animation will show {num_frames} frames")
        anim = FuncAnimation( # Keep reference to prevent GC
            fig, update,
            frames=frames,
            init_func=init_anim,
            blit=False,                    # Blitting off: simpler, more robust for 3D.
            interval=animation_interval,
            repeat=True
        )

        print("Showing animation...")
        plt.show()

    return result


# ============================================================================
# MAIN: single visual run + Monte Carlo
# ============================================================================

if __name__ == "__main__":
    # Create a numpy RNG for sampling randomized start conditions.
    #rng = np.random.default_rng(123) #fixed seed for consistent results
    rng = np.random.default_rng() #this will cause different results each time

    # ---- 1) Watch 5 visual randomized engagements ----
    # n_visual_runs = 5
    # for r in range(n_visual_runs):
    #     print(f"\n=== Visual run {r + 1} / {n_visual_runs} ===")
    #     a_vis, m_vis = sample_random_starts(rng)
    #     simulate_single_engagement(
    #         aircraft_start_loc_input=a_vis,
    #         missile_start_loc_input=m_vis,
    #         rng_seed=r,        # seed per run, so decoy/noise differ by r
    #         make_plot=True,    # show animation
    #         verbose=True,      # print events
    #     )

    # ---- 2) Monte Carlo over many randomized engagements ----
    n_trials = 10            # Number of Monte Carlo trials.
    n_real = 0              # Count of real target intercepts.
    n_decoy = 0             # Count of decoy intercepts.
    n_miss = 0              # Count of complete misses.
    closest_misses = []     # Closest miss distance per run (for stats).

    print(f"\nRunning Monte Carlo with {n_trials} randomized engagements...")
    for k in range(n_trials):
        # For each trial, sample new start geometry.
        a_mc, m_mc = sample_random_starts(rng)
        #print the start pos of both aircraft and missle 
        print(f"Trial {k+1}: AIRCRAFT_START = {a_mc}, MISSILE_START = {m_mc}")
        res = simulate_single_engagement(
            aircraft_start_loc_input=a_mc,
            missile_start_loc_input=m_mc,
            rng=rng,
            make_plot=True,
            verbose=True,
            run_id=k+1,         
            total_runs=n_trials
        )

        # Tally outcomes.
        if res["intercept_real"]:
            n_real += 1
        elif res["intercept_decoy"]:
            n_decoy += 1
        else:
            n_miss += 1

        closest_misses.append(res["closest_miss"])

    # Convert counts to probabilities.
    p_real = n_real / n_trials
    p_decoy = n_decoy / n_trials
    p_miss = n_miss / n_trials
    avg_closest = float(np.mean(closest_misses))
    
    # Print Monte Carlo summary.
    print("\nMonte Carlo results:")
    print(f"  Real target intercepted: {n_real} / {n_trials}  ({p_real*100:.1f}%)")
    print(f"  Decoy intercepted:       {n_decoy} / {n_trials} ({p_decoy*100:.1f}%)")
    print(f"  No intercept:            {n_miss} / {n_trials}   ({p_miss*100:.1f}%)")
    print(f"  Average closest miss (to REAL target): {avg_closest:.2f} m")