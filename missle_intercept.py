"""
================================================================================
REALISTIC 3D MISSILE-TARGET PURSUIT SIMULATION
================================================================================

This simulation models a complete air-to-air engagement including:

TARGET AIRCRAFT:
    - Piecewise trajectory: straight → banked turn → straight → coast
    - Evasive "jink" maneuvers with randomized amplitude/frequency/phase
    - Configurable trajectory timing and geometry

MISSILE SYSTEM:
    - Multi-phase propulsion: boost → sustain → coast
    - Mass depletion during powered flight
    - Mach-dependent aerodynamic drag with transonic rise
    - Gravity effects throughout flight
    - G-limited maneuvering capability

SEEKER MODEL:
    - Gimbal angle limits (configurable off-boresight angle)
    - SNR-based target detection using radar range equation
    - Aspect-angle dependent RCS modeling (head/beam/tail)
    - Doppler velocity gating for clutter rejection
    - Track memory to coast through brief signal loss
    - Lock hysteresis to prevent rapid target switching
    - Measurement noise on angle estimates

GUIDANCE SYSTEM:
    - Mid-course guidance: proportional pursuit toward predicted intercept
    - Terminal guidance: Augmented Proportional Navigation (APN)
    - Target acceleration estimation for APN
    - Realistic guidance loop latency
    - Actuator dynamics modeled as first-order lag

COUNTERMEASURES:
    - Multiple decoys with timed deployment
    - Decoy drag and gravitational deceleration
    - Seeker confusion logic based on SNR comparison

VISUALIZATION:
    - 3D animated plot with ground plane reference
    - Real-time HUD: time, range, Mach, lock status, guidance mode
    - Color-coded missile trajectory based on lock state
    - Monte Carlo statistics summary

ORIGINAL FEATURES PRESERVED:
    - All trajectory parameters (timing, turn angle, YZ tilt, climb shaping)
    - Evasive jink with randomization interval
    - Decoy drift with random direction/speed
    - Seeker gimbal, SNR, Doppler, hysteresis, track memory
    - Guidance delay buffer and actuator lag
    - Aspect-dependent RCS calculation
    - Mach-dependent drag coefficients
    - Mass depletion during burn
    - All visualization elements (trails, HUD, markers)

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - Required for 3D projection
from collections import deque


# ============================================================================
# SECTION 1: PHYSICAL CONSTANTS
# ============================================================================

# Gravitational acceleration vector (m/s²)
# Points in -Z direction (down)
GRAVITY = np.array([0.0, 0.0, -9.81])

# Specific gas constant for air (J/kg·K)
# Used in speed of sound calculation
R_GAS = 287.05

# Ratio of specific heats for air (dimensionless)
# γ = Cp/Cv for diatomic gas
GAMMA = 1.4


# ============================================================================
# SECTION 2: ATMOSPHERIC MODEL PARAMETERS
# ============================================================================

# Sea level air density (kg/m³)
# Standard atmosphere reference value
RHO_SEA_LEVEL = 1.225

# Atmospheric scale height (m)
# Characteristic height for exponential density model
# Density falls to 1/e at this altitude
SCALE_HEIGHT = 8500.0

# Sea level temperature (K)
# Standard atmosphere reference: 15°C = 288.15K
TEMP_SEA_LEVEL = 288.15

# Temperature lapse rate in troposphere (K/m)
# Temperature decreases ~6.5°C per 1000m up to tropopause
TEMP_LAPSE_RATE = 0.0065


# ============================================================================
# SECTION 3: TARGET AIRCRAFT PARAMETERS
# ============================================================================

# ----- Physical Properties -----
# Aircraft mass (kg) - typical fighter aircraft
TARG_MASS = 15000.0

# Wing reference area (m²) - used for lift/drag calculations
TARG_WING_AREA = 50.0

# Zero-lift drag coefficient - parasite drag
TARG_CD0 = 0.02

# Induced drag factor - drag due to lift
TARG_K = 0.05

# Maximum g-load capability - structural/physiological limit
TARG_MAX_G = 9.0

# ----- Trajectory Timing (seconds) -----
# Duration of first straight-line segment
STRAIGHT_TIME_1 = 5.0

# Duration of curved turning segment
CURVE_TIME = 25.0

# Duration of second straight-line segment
STRAIGHT_TIME_2 = 25.0

# ----- Trajectory Geometry -----
# Initial true airspeed (m/s)
# ~300 m/s ≈ Mach 0.88 at sea level, Mach 1.0 at altitude
TARG_INITIAL_SPEED = 300.0

# Total turn angle (radians)
# Negative = right turn, Positive = left turn
# -4π/3 ≈ -240° turn
TURN_ANGLE = -np.pi * 4 / 3

# Tilt angle of turn plane into YZ plane (radians)
# Makes the turn 3D instead of flat horizontal
# Negative = turn plane tilts down on one side
YZ_ANGLE = -np.pi / 12

# Vertical shaping factor during turn
# Adds climb/dive variation proportional to speed²
# Negative = net descent during turn
CLIMB_RATE_CURVE = -0.001

# ----- Evasive Maneuver Parameters -----
# Time when evasive maneuvers begin (s)
T_EVASION_START = 30.0

# Base amplitude of sinusoidal jink (m)
JINK_AMP_BASE = 500.0

# Random variation in jink amplitude (m)
# Actual amplitude = base ± random
JINK_AMP_RANDOM = 200.0

# Base frequency of jink oscillation (Hz)
# 0.2 Hz = one complete cycle every 5 seconds
JINK_FREQ_BASE = 0.2

# Random variation in jink frequency (Hz)
JINK_FREQ_RANDOM = 0.1

# How often to randomize jink parameters (s)
# Creates unpredictable evasion pattern
JINK_RANDOMIZE_INTERVAL = 3.0


# ============================================================================
# SECTION 4: MISSILE PARAMETERS
# ============================================================================

# Missile end-of-life logic
MISS_MAX_FLIGHT_TIME = 60.0    # s (from launch)
MISS_MIN_SPEED       = 200.0   # m/s (below this, treat as spent)
MISS_MIN_CLOSING_R   = 2000.0  # m (don't give up if we're still basically on top of target)

# ----- Mass Properties -----
# Launch mass including fuel (kg)
MISS_MASS_INITIAL = 150.0

# Mass after all fuel burned (kg)
# Fuel mass = 150 - 90 = 60 kg
MISS_MASS_BURNOUT = 90.0

# ----- Aerodynamics -----
# Reference area for drag calculation (m²)
# Approximately missile cross-sectional area
MISS_REF_AREA = 0.05

# Drag coefficient in subsonic regime (Mach < 0.8)
MISS_CD_SUBSONIC = 0.30

# Peak drag coefficient in transonic regime (Mach 0.8-1.2)
# Transonic drag rise due to shock wave formation
MISS_CD_TRANSONIC = 0.50

# Drag coefficient in supersonic regime (Mach > 1.2)
# Lower than transonic as flow becomes fully supersonic
MISS_CD_SUPERSONIC = 0.35

# ----- Propulsion Profile -----
# Boost phase: High thrust for rapid acceleration
MISS_BOOST_THRUST = 60000.0    # Thrust force (N)
MISS_BOOST_TIME = 4.0          # Duration (s)

# Sustain phase: Lower thrust to maintain speed
MISS_SUSTAIN_THRUST = 10000.0  # Thrust force (N)
MISS_SUSTAIN_TIME = 20.0       # Duration (s)

# Coast phase: Zero thrust, missile decelerates due to drag
# Begins after boost + sustain time

# ----- Performance Limits -----
# Maximum lateral acceleration (g's)
# Structural limit of airframe
MISS_MAX_G      = 60.0         # 60 g is still believable for a modern AAM

# Maximum angle of attack (radians)
# Beyond this, missile may stall or become unstable
MISS_MAX_AOA = np.radians(25)

# ----- Engagement Parameters -----
# Time when missile is released/launched (s)
MISSILE_LAUNCH_TIME = 0.0

# Proximity fuze lethal radius (m)
# If missile passes within this distance, target is killed
KILL_DIST       = 50.0         # slightly more generous fuze radius

# ============================================================================
# SECTION 5: SEEKER PARAMETERS
# ============================================================================

# ----- Field of Regard -----
# Maximum off-boresight angle seeker can look (radians)
# 75° allows wide-angle tracking
SEEKER_GIMBAL_LIMIT = np.radians(75)

# Instantaneous field of view (radians)
# Angular width of seeker beam
SEEKER_FOV = np.radians(3)

# ----- Detection Thresholds -----
# Minimum SNR to maintain an existing lock
# Lower threshold = easier to keep tracking
SEEKER_LOCK_SNR_THRESHOLD = 2.0

# Minimum SNR to acquire a new lock
# Higher threshold = need stronger signal to start tracking
SEEKER_ACQ_SNR_THRESHOLD = 4.0

# Maximum detection range (m)
# SNR normalized relative to this range
SEEKER_RANGE_MAX = 60000.0

# Angular measurement noise standard deviation (radians)
# Noise proportional to range: σ_position = σ_angle × range
SEEKER_NOISE_STD = 0.005

# Seeker update rate (Hz) - how often seeker refreshes
SEEKER_UPDATE_RATE = 50.0

# ----- Radar Cross Section (RCS) -----
# RCS varies with viewing angle relative to target heading
# Head-on RCS (m²) - typically smallest due to shaping
TARGET_RCS_HEAD = 3.0

# Beam aspect RCS (m²) - typically largest (side profile)
TARGET_RCS_BEAM = 10.0

# Tail aspect RCS (m²) - moderate (engine cavities visible)
TARGET_RCS_TAIL = 5.0

# Decoy RCS (m²) - designed to be attractive to seeker
DECOY_RCS_BASE = 8.0

# ----- Doppler Velocity Gating -----
# Minimum closing velocity to track (m/s)
# Filters out ground clutter and crossing targets
DOPPLER_MIN_VELOCITY = 20.0

# ----- Track Management -----
# Time to coast (maintain track without measurement) before dropping lock (s)
TRACK_MEMORY_TIME = 3.0

# Minimum time before switching to higher-SNR target (s)
# Prevents rapid switching between similar targets
LOCK_HYSTERESIS_TIME = 0.3


# ============================================================================
# SECTION 6: GUIDANCE PARAMETERS
# ============================================================================

# Navigation constant for Proportional Navigation
# Typical values: 3-5
# Higher = more aggressive correction, risk of overshoot
N_PN            = 5.0          # more aggressive PN

# Guidance loop latency (s)
# Time delay from measurement to command output
GUIDANCE_DELAY = 0.020

# Fin actuator time constant (s)
# First-order lag: τ(da/dt) + a = a_cmd
ACTUATOR_TIME_CONST = 0.03

# Guidance computer update rate (Hz)
GUIDANCE_UPDATE_RATE = 100.0

# Mid-course guidance proportional gain
# Higher = more aggressive steering before lock
MIDCOURSE_GAIN  = 4.0          # steer harder mid-course


# ============================================================================
# SECTION 7: DECOY/COUNTERMEASURE PARAMETERS
# ============================================================================

# Number of decoys aircraft deploys
N_DECOYS = 3

# Time when decoys are deployed (s)
DECOY_DEPLOY_TIME = 35.0

# Base drift speed of decoys after deployment (m/s)
DECOY_BASE_SPEED = 200.0

# Standard deviation of drift speed (m/s)
DECOY_SPEED_SIGMA = 50.0

# Decoy mass (kg) - affects deceleration due to drag
DECOY_MASS = 2.0

# Decoy drag area: Cd × Reference_Area (m²)
DECOY_DRAG_AREA = 0.1

# Decoy active emission duration (s)
# How long decoy produces strong radar/IR signature
DECOY_BURN_TIME = 5.0  # Decoy active emission duration (s)

# ============================================================================
# SECTION 8: SIMULATION PARAMETERS
# ============================================================================

# Total simulation duration (s)
TMAX = 75.0

# Integration time step (s)
# Smaller = more accurate but slower
DT = 0.001

# Animation frame interval (milliseconds)
ANIMATION_INTERVAL = 5

# Maximum number of frames in animation
# Frames are subsampled if total exceeds this
ANIMATION_MAX_FRAMES = 500

# ----- Camera/View Settings -----
# Initial view azimuth angle (degrees)
BASE_AZIM = 45

# Initial view elevation angle (degrees)
BASE_ELEV = 20

# Default starting positions (used if not randomizing)
AIRCRAFT_START_DEFAULT = np.array([0.0, 0.0, 12000.0])
MISSILE_START_DEFAULT = np.array([13000.0, 12000.0, 0.0])


# ============================================================================
# SECTION 9: ATMOSPHERIC FUNCTIONS
# ============================================================================

def get_air_density(altitude):
    """
    Calculate air density using exponential atmosphere model.
    
    The exponential model approximates the barometric formula:
    ρ(h) = ρ₀ × exp(-h/H)
    
    Parameters
    ----------
    altitude : float
        Altitude above sea level (m)
        
    Returns
    -------
    float
        Air density (kg/m³)
    """
    # Clamp altitude to non-negative
    alt = max(0.0, altitude)
    return RHO_SEA_LEVEL * np.exp(-alt / SCALE_HEIGHT)


def get_temperature(altitude):
    """
    Calculate atmospheric temperature using troposphere lapse rate.
    
    In the troposphere (0-11km), temperature decreases linearly.
    Above the tropopause, temperature is constant at ~216.65K.
    
    Parameters
    ----------
    altitude : float
        Altitude above sea level (m)
        
    Returns
    -------
    float
        Temperature (K)
    """
    alt = max(0.0, altitude)
    # Linear decrease with altitude, clamped at tropopause temperature
    return max(TEMP_SEA_LEVEL - TEMP_LAPSE_RATE * alt, 216.65)


def get_speed_of_sound(altitude):
    """
    Calculate speed of sound based on temperature.
    
    For an ideal gas: a = √(γRT)
    where γ = ratio of specific heats
          R = gas constant
          T = temperature
    
    Parameters
    ----------
    altitude : float
        Altitude above sea level (m)
        
    Returns
    -------
    float
        Speed of sound (m/s)
    """
    T = get_temperature(altitude)
    return np.sqrt(GAMMA * R_GAS * T)


def get_mach_number(velocity, altitude):
    """
    Calculate Mach number for given velocity and altitude.
    
    Mach number = velocity / speed_of_sound
    
    Parameters
    ----------
    velocity : np.ndarray
        Velocity vector (m/s)
    altitude : float
        Altitude above sea level (m)
        
    Returns
    -------
    float
        Mach number (dimensionless)
    """
    speed = np.linalg.norm(velocity)
    a = get_speed_of_sound(altitude)
    return speed / a if a > 0 else 0.0


# ============================================================================
# SECTION 10: AERODYNAMIC FUNCTIONS
# ============================================================================

def get_missile_cd(mach):
    """
    Calculate missile drag coefficient based on Mach number.
    
    Models the transonic drag rise phenomenon:
    - Subsonic (M < 0.8): Constant low drag
    - Transonic (0.8 < M < 1.2): Sharp drag rise due to shock waves
    - Supersonic (M > 1.2): Gradual decrease as flow stabilizes
    
    Parameters
    ----------
    mach : float
        Mach number
        
    Returns
    -------
    float
        Drag coefficient (dimensionless)
    """
    if mach < 0.8:
        # Subsonic: constant drag coefficient
        return MISS_CD_SUBSONIC
    elif mach < 1.2:
        # Transonic: linear interpolation through drag rise
        t = (mach - 0.8) / 0.4  # Normalized position in transonic range
        return MISS_CD_SUBSONIC + t * (MISS_CD_TRANSONIC - MISS_CD_SUBSONIC)
    else:
        # Supersonic: gradual decrease (wave drag diminishes)
        return MISS_CD_SUPERSONIC + 0.1 / mach


def calculate_drag(velocity, altitude, cd, ref_area):
    """
    Calculate aerodynamic drag force vector.
    
    Drag equation: D = 0.5 × ρ × Cd × A × V²
    Direction: opposite to velocity
    
    Parameters
    ----------
    velocity : np.ndarray
        Velocity vector (m/s)
    altitude : float
        Altitude for density calculation (m)
    cd : float
        Drag coefficient
    ref_area : float
        Reference area (m²)
        
    Returns
    -------
    np.ndarray
        Drag force vector (N), opposite to velocity direction
    """
    speed = np.linalg.norm(velocity)
    if speed < 1e-6:
        return np.zeros(3)
    
    rho = get_air_density(altitude)
    # D = 0.5 × ρ × Cd × A × V² × (-V_hat)
    # Simplified: -0.5 × ρ × Cd × A × V × V_vec
    return -0.5 * rho * cd * ref_area * speed * velocity


# ============================================================================
# SECTION 11: RADAR CROSS SECTION MODEL
# ============================================================================

def calculate_rcs(target_pos, target_vel, observer_pos):
    """
    Calculate aspect-angle-dependent Radar Cross Section (RCS).
    
    Real aircraft RCS varies significantly with viewing angle:
    - Head-on (0°): Low RCS due to small frontal area and stealth shaping
    - Beam (90°): High RCS due to large side profile
    - Tail (180°): Moderate RCS, engine cavities may increase signature
    
    Aspect angle is measured between:
    - Line of sight from target to observer
    - Target's velocity vector (heading)
    
    Parameters
    ----------
    target_pos : np.ndarray
        Target position vector (m)
    target_vel : np.ndarray
        Target velocity vector (m/s)
    observer_pos : np.ndarray
        Observer (missile) position vector (m)
        
    Returns
    -------
    float
        Effective RCS (m²)
    """
    # Line of sight from target to observer
    los = observer_pos - target_pos
    los_dist = np.linalg.norm(los)
    
    if los_dist < 1e-6:
        return TARGET_RCS_BEAM  # Default if coincident
    
    los_unit = los / los_dist
    
    # Target heading direction
    tgt_speed = np.linalg.norm(target_vel)
    if tgt_speed < 1e-6:
        return TARGET_RCS_BEAM  # Default if stationary
    
    tgt_heading = target_vel / tgt_speed
    
    # Aspect angle: 0 = head-on, π/2 = beam, π = tail
    cos_aspect = np.dot(los_unit, tgt_heading)
    aspect_angle = np.arccos(np.clip(cos_aspect, -1, 1))
    
    # Piecewise linear interpolation based on aspect angle
    if aspect_angle < np.pi / 4:
        # Forward quarter (0° to 45°): head → beam
        t = aspect_angle / (np.pi / 4)
        return TARGET_RCS_HEAD + t * (TARGET_RCS_BEAM - TARGET_RCS_HEAD)
    elif aspect_angle < 3 * np.pi / 4:
        # Beam aspect (45° to 135°): constant beam RCS
        return TARGET_RCS_BEAM
    else:
        # Rear quarter (135° to 180°): beam → tail
        t = (aspect_angle - 3 * np.pi / 4) / (np.pi / 4)
        return TARGET_RCS_BEAM + t * (TARGET_RCS_TAIL - TARGET_RCS_BEAM)


# ============================================================================
# SECTION 12: SEEKER CLASS
# ============================================================================

class Seeker:
    """
    Realistic missile seeker model.
    
    Simulates a radar seeker with physical and signal processing constraints:
    
    1. GIMBAL LIMITS: Seeker can only look within a cone around boresight
    2. SNR DETECTION: Target must have sufficient signal-to-noise ratio
    3. DOPPLER GATING: Filters out slow-moving/crossing targets
    4. TRACK MEMORY: Maintains track briefly through signal dropouts  
    5. LOCK HYSTERESIS: Prevents rapid switching between targets
    6. MEASUREMENT NOISE: Realistic angular measurement errors
    
    Attributes
    ----------
    rng : np.random.Generator
        Random number generator for noise
    current_lock : str or None
        Current lock: None, 'target', or 'decoy_N'
    lock_time : float
        Time since current lock was acquired (s)
    last_measurement_time : float
        Time of last valid measurement (s)
    track_coasting : bool
        True if coasting without measurement
    boresight : np.ndarray
        Seeker boresight direction (unit vector)
    """
    
    def __init__(self, rng, initial_boresight=None):
        """
        Initialize seeker.
        
        Parameters
        ----------
        rng : np.random.Generator
            Random number generator for noise
        initial_boresight : np.ndarray, optional
            Initial boresight direction. Defaults to +X axis.
        """
        self.rng = rng
        self.current_lock = None
        self.lock_time = 0.0
        self.last_measurement_time = 0.0
        self.track_coasting = False
        
        # Initialize boresight from provided direction or default
        if initial_boresight is not None:
            norm = np.linalg.norm(initial_boresight)
            if norm > 1e-6:
                self.boresight = initial_boresight / norm
            else:
                self.boresight = np.array([1.0, 0.0, 0.0])
        else:
            self.boresight = np.array([1.0, 0.0, 0.0])
    
    def update_boresight(self, missile_vel):
        """
        Update seeker boresight to align with missile velocity.
        
        Assumes seeker is body-fixed and missile flies along velocity vector.
        
        Parameters
        ----------
        missile_vel : np.ndarray
            Missile velocity vector (m/s)
        """
        speed = np.linalg.norm(missile_vel)
        if speed > 1e-6:
            self.boresight = missile_vel / speed
    
    def check_gimbal(self, missile_pos, target_pos):
        """
        Check if target is within seeker gimbal limits.
        
        Parameters
        ----------
        missile_pos : np.ndarray
            Missile position (m)
        target_pos : np.ndarray
            Target position (m)
            
        Returns
        -------
        tuple
            (within_limits: bool, angle: float in radians)
        """
        los = target_pos - missile_pos
        los_dist = np.linalg.norm(los)
        
        if los_dist < 1e-6:
            return True, 0.0
        
        los_unit = los / los_dist
        cos_angle = np.dot(los_unit, self.boresight)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        return angle <= SEEKER_GIMBAL_LIMIT, angle
    
    def calculate_snr(self, missile_pos, target_pos, rcs):
        """
        Calculate Signal-to-Noise Ratio for radar detection.
        
        Simplified radar range equation:
        SNR ∝ (RCS × G² × λ² × Pt) / (R⁴ × k × T × B × L)
        
        Normalized form: SNR = RCS × (R_max/R)⁴ × attenuation
        
        Parameters
        ----------
        missile_pos : np.ndarray
            Missile (radar) position (m)
        target_pos : np.ndarray
            Target position (m)
        rcs : float
            Target radar cross section (m²)
            
        Returns
        -------
        float
            Estimated SNR (linear scale, not dB)
        """
        los = target_pos - missile_pos
        dist = np.linalg.norm(los)
        
        # Check range limits
        if dist < 1e-6 or dist > SEEKER_RANGE_MAX:
            return 0.0
        
        # Radar equation: SNR ∝ RCS / R⁴
        # Normalized so SNR = RCS at max range
        range_factor = (SEEKER_RANGE_MAX / dist) ** 4
        snr_base = rcs * range_factor
        
        # Atmospheric attenuation (simplified exponential decay)
        attenuation = np.exp(-dist / 200000.0)
        
        # Apply attenuation
        snr = snr_base * attenuation
        
        # Add proportional noise (receiver noise + clutter)
        noise = self.rng.normal(0, 0.05 * snr + 0.1)
        
        return max(0.0, snr + noise)
    
    def check_doppler_gate(self, missile_pos, missile_vel, target_pos, target_vel):
        """
        Check if target passes Doppler velocity gate.
        
        Radar seekers filter returns by Doppler shift (closing velocity).
        This rejects:
        - Ground clutter (zero Doppler)
        - Slow-moving objects
        - Targets flying perpendicular (in the "notch")
        
        Parameters
        ----------
        missile_pos : np.ndarray
            Missile position (m)
        missile_vel : np.ndarray
            Missile velocity (m/s)
        target_pos : np.ndarray
            Target position (m)
        target_vel : np.ndarray
            Target velocity (m/s)
            
        Returns
        -------
        bool
            True if target passes Doppler gate
        """
        los = target_pos - missile_pos
        dist = np.linalg.norm(los)
        
        if dist < 1e-6:
            return True
        
        los_unit = los / dist
        
        # Closing velocity = rate of range decrease
        # Positive when missile approaching target
        rel_vel = missile_vel - target_vel
        closing_vel = np.dot(rel_vel, los_unit)
        
        # Must have sufficient closing velocity
        return closing_vel >= DOPPLER_MIN_VELOCITY
    
    def evaluate_target(self, missile_pos, missile_vel, target_pos, target_vel,
                        rcs, is_current_lock=False):
        """
        Evaluate whether a potential target can be tracked.
        
        Checks all seeker constraints:
        1. Within gimbal limits
        2. Sufficient SNR
        3. Passes Doppler gate (with penalty if fails)
        
        Parameters
        ----------
        missile_pos : np.ndarray
            Missile position (m)
        missile_vel : np.ndarray
            Missile velocity (m/s)
        target_pos : np.ndarray
            Target position (m)
        target_vel : np.ndarray
            Target velocity (m/s)
        rcs : float
            Target RCS (m²)
        is_current_lock : bool
            True if this is the currently locked target
            (uses lower SNR threshold)
            
        Returns
        -------
        tuple
            (trackable: bool, snr: float)
        """
        # Check gimbal limits - hard constraint
        in_gimbal, angle = self.check_gimbal(missile_pos, target_pos)
        if not in_gimbal:
            return False, 0.0
        
        # Calculate SNR
        snr = self.calculate_snr(missile_pos, target_pos, rcs)
        
        # Apply Doppler gate penalty (soft constraint)
        if not self.check_doppler_gate(missile_pos, missile_vel, target_pos, target_vel):
            snr *= 0.2  # 80% SNR penalty, but don't completely reject
        
        # Use appropriate threshold
        threshold = SEEKER_LOCK_SNR_THRESHOLD if is_current_lock else SEEKER_ACQ_SNR_THRESHOLD
        
        return snr >= threshold, snr
    
    def update(self, t, dt, missile_pos, missile_vel, target_pos, target_vel,
               decoy_positions, decoy_velocities, decoy_active):
        """
        Main seeker update - determine current lock among all candidates.
        
        Processing steps:
        1. Update boresight to follow missile velocity
        2. Evaluate real target and all active decoys
        3. Build list of trackable candidates with SNR
        4. Apply lock hysteresis logic
        5. Select best candidate or maintain coast track
        6. Add measurement noise to output
        
        Parameters
        ----------
        t : float
            Current simulation time (s)
        dt : float
            Time step (s)
        missile_pos : np.ndarray
            Missile position (m)
        missile_vel : np.ndarray
            Missile velocity (m/s)
        target_pos : np.ndarray
            True target position (m)
        target_vel : np.ndarray
            True target velocity (m/s)
        decoy_positions : list of np.ndarray
            Decoy positions (m)
        decoy_velocities : list of np.ndarray
            Decoy velocities (m/s)
        decoy_active : list of bool
            Decoy active flags
            
        Returns
        -------
        tuple
            (locked_pos, locked_vel, lock_type) or (None, None, None)
            lock_type is 'target', 'decoy_N', or None
        """
        # Update boresight to follow missile heading
        self.update_boresight(missile_vel)
        
        # Build list of trackable candidates
        candidates = []
        
        # ----- Evaluate real target -----
        rcs = calculate_rcs(target_pos, target_vel, missile_pos)
        is_current = (self.current_lock == 'target')
        trackable, snr = self.evaluate_target(
            missile_pos, missile_vel, target_pos, target_vel, rcs,
            is_current_lock=is_current
        )
        if trackable:
            candidates.append(('target', target_pos.copy(), target_vel.copy(), snr))
        
        # ----- Evaluate each active decoy -----
        for i, (d_pos, d_vel, active) in enumerate(zip(
                decoy_positions, decoy_velocities, decoy_active)):
            if not active:
                continue
            
            lock_name = f'decoy_{i}'
            is_current = (self.current_lock == lock_name)
            trackable, snr = self.evaluate_target(
                missile_pos, missile_vel, d_pos, d_vel, DECOY_RCS_BASE,
                is_current_lock=is_current
            )
            if trackable:
                candidates.append((lock_name, d_pos.copy(), d_vel.copy(), snr))
        
        # ----- Handle no valid candidates -----
        if not candidates:
            self.track_coasting = True
            # Drop lock after track memory expires
            if t - self.last_measurement_time > TRACK_MEMORY_TIME:
                self.current_lock = None
            return None, None, None
        
        # ----- Select best candidate -----
        # Sort by SNR (highest first)
        candidates.sort(key=lambda x: x[3], reverse=True)
        best = candidates[0]
        
        # Apply hysteresis - don't switch locks too quickly
        if self.current_lock is not None and self.current_lock != best[0]:
            if self.lock_time < LOCK_HYSTERESIS_TIME:
                # Try to keep current lock if still trackable
                for c in candidates:
                    if c[0] == self.current_lock:
                        best = c
                        break
        
        # ----- Update lock state -----
        if best[0] != self.current_lock:
            self.current_lock = best[0]
            self.lock_time = 0.0
        else:
            self.lock_time += dt
        
        self.last_measurement_time = t
        self.track_coasting = False
        
        # ----- Add measurement noise -----
        los = best[1] - missile_pos
        dist = np.linalg.norm(los)
        if dist > 1e-6:
            # Angular noise translates to position noise proportional to range
            noise_offset = self.rng.normal(0, SEEKER_NOISE_STD * dist, 3)
            return best[1] + noise_offset, best[2], best[0]
        
        return best[1], best[2], best[0]


# ============================================================================
# SECTION 13: GUIDANCE SYSTEM CLASS
# ============================================================================

class GuidanceSystem:
    """
    Missile guidance system with mid-course and terminal modes.
    
    MID-COURSE MODE (before seeker lock):
        Uses proportional pursuit toward predicted intercept point.
        Estimates time-to-go and leads the target.
    
    TERMINAL MODE (after seeker lock):
        Uses Augmented Proportional Navigation (APN):
        a_cmd = N × Vc × ω + (N/2) × a_t⊥
        
        where:
            N = navigation constant
            Vc = closing velocity
            ω = line-of-sight rate
            a_t⊥ = target acceleration normal to LOS
    
    Also models:
        - Guidance loop latency via delay buffer
        - Actuator dynamics via first-order lag
    
    Attributes
    ----------
    delay_buffer : deque
        Buffer storing (time, command) pairs for delay
    prev_los : np.ndarray or None
        Previous LOS unit vector for rate calculation
    prev_target_vel : np.ndarray or None
        Previous target velocity for acceleration estimation
    commanded_accel : np.ndarray
        Current commanded acceleration (before actuator)
    actual_accel : np.ndarray
        Actual acceleration (after actuator dynamics)
    mode : str
        Current guidance mode: 'midcourse' or 'terminal'
    """
    
    def __init__(self):
        """Initialize guidance system."""
        self.delay_buffer = deque()
        self.prev_los = None
        self.prev_target_vel = None
        self.commanded_accel = np.zeros(3)
        self.actual_accel = np.zeros(3)
        self.mode = 'midcourse'
    
    def compute_midcourse_command(self, missile_pos, missile_vel, 
                                target_pos, target_vel):
        """
        MID-COURSE GUIDANCE: classic 3D Proportional Navigation.

        a_cmd = N_mc * Vc * n_dot

        where:
            r    = target_pos - missile_pos         (relative position)
            v_rel= target_vel - missile_vel         (relative velocity)
            n    = r / |r|                          (LOS unit vector)
            Vc   = -v_rel · n                       (closing speed, >0 when closing)
            n_dot= v_rel_perp / |r|                 (LOS rate vector)
        """
        # Relative position and velocity
        r = target_pos - missile_pos
        v_rel = target_vel - missile_vel

        R = np.linalg.norm(r)
        if R < 1e-6:
            return np.zeros(3)

        n = r / R

        # Closing speed: positive when closing
        Vc = -np.dot(v_rel, n)
        if Vc <= 0.0:
            # Not closing – don't waste energy
            return np.zeros(3)

        # Component of relative velocity perpendicular to LOS
        v_rel_perp = v_rel - np.dot(v_rel, n) * n

        # LOS rate vector
        n_dot = v_rel_perp / R

        # PN command (use midcourse gain as Nav constant)
        a_cmd = MIDCOURSE_GAIN * Vc * n_dot

        # Project into plane perpendicular to missile velocity (pure lateral accel)
        speed = np.linalg.norm(missile_vel)
        if speed > 1e-6:
            u_v = missile_vel / speed
            a_cmd = a_cmd - np.dot(a_cmd, u_v) * u_v

        return a_cmd
    
    def estimate_target_accel(self, target_vel, dt):
        """
        Estimate target acceleration from velocity history.
        
        Simple finite difference: a = (v_new - v_old) / dt
        
        Parameters
        ----------
        target_vel : np.ndarray
            Current target velocity (m/s)
        dt : float
            Time step (s)
            
        Returns
        -------
        np.ndarray
            Estimated target acceleration (m/s²)
        """
        if self.prev_target_vel is None:
            self.prev_target_vel = target_vel.copy()
            return np.zeros(3)
        
        accel = (target_vel - self.prev_target_vel) / dt
        self.prev_target_vel = target_vel.copy()
        return accel
    
    def compute_apn_command(self, missile_pos, missile_vel, 
                        target_pos, target_vel, dt):
        """
        TERMINAL GUIDANCE: Augmented Proportional Navigation (APN).

        a_cmd = N * Vc * n_dot  +  (N/2) * a_t_perp
        """
        r = target_pos - missile_pos
        v_rel = target_vel - missile_vel

        R = np.linalg.norm(r)
        if R < 1e-6:
            return np.zeros(3)

        n = r / R

        # Closing speed: positive when closing
        Vc = -np.dot(v_rel, n)
        if Vc <= 0.0:
            return np.zeros(3)

        # LOS rate
        v_rel_perp = v_rel - np.dot(v_rel, n) * n
        n_dot = v_rel_perp / R

        # Base PN term
        a_pn = N_PN * Vc * n_dot

        # Target acceleration estimate + component perpendicular to LOS
        a_t = self.estimate_target_accel(target_vel, dt)
        a_t_perp = a_t - np.dot(a_t, n) * n

        # Augmented PN
        a_cmd = a_pn + 0.5 * N_PN * a_t_perp

        # Pure lateral wrt missile velocity
        speed = np.linalg.norm(missile_vel)
        if speed > 1e-6:
            u_v = missile_vel / speed
            a_cmd = a_cmd - np.dot(a_cmd, u_v) * u_v

        return a_cmd
        
    def update(self, t, dt, missile_pos, missile_vel, 
               target_pos, target_vel, has_lock):
        """
        Main guidance update with mode selection, delay, and actuator dynamics.
        
        Parameters
        ----------
        t : float
            Current time (s)
        dt : float
            Time step (s)
        missile_pos : np.ndarray
            Missile position (m)
        missile_vel : np.ndarray
            Missile velocity (m/s)
        target_pos : np.ndarray
            Target position (m)
        target_vel : np.ndarray
            Target velocity (m/s)
        has_lock : bool
            True if seeker has lock
            
        Returns
        -------
        np.ndarray
            Actual acceleration after delays and actuator dynamics (m/s²)
        """
        # Select guidance mode based on lock status
        if has_lock:
            self.mode = 'terminal'
            cmd = self.compute_apn_command(missile_pos, missile_vel,
                                           target_pos, target_vel, dt)
        else:
            self.mode = 'midcourse'
            cmd = self.compute_midcourse_command(missile_pos, missile_vel,
                                                  target_pos, target_vel)
            # Reset LOS history when transitioning from terminal to midcourse
            self.prev_los = None
        
        # ----- Apply guidance delay -----
        # Add current command to delay buffer
        self.delay_buffer.append((t, cmd))
        
        # Remove commands older than delay time
        while self.delay_buffer and self.delay_buffer[0][0] < t - GUIDANCE_DELAY:
            self.delay_buffer.popleft()
        
        # Use oldest command in buffer (delayed command)
        if self.delay_buffer:
            self.commanded_accel = self.delay_buffer[0][1]
        
        # ----- Apply actuator dynamics -----
        # First-order lag: τ(da/dt) + a = a_cmd
        # Discrete form: a_new = a_old + α(a_cmd - a_old)
        # where α = dt / (τ + dt)
        tau = ACTUATOR_TIME_CONST
        alpha = dt / (tau + dt)
        self.actual_accel = self.actual_accel + alpha * (self.commanded_accel - self.actual_accel)
        
        return self.actual_accel


# ============================================================================
# SECTION 14: MISSILE CLASS
# ============================================================================

class Missile:
    """
    Complete missile model with propulsion, aerodynamics, seeker, and guidance.
    
    Attributes
    ----------
    pos : np.ndarray
        Current position (m)
    vel : np.ndarray
        Current velocity (m/s)
    mass : float
        Current mass (kg)
    launched : bool
        Whether missile has been launched
    launch_time : float
        Time of launch (s)
    seeker : Seeker
        Seeker subsystem
    guidance : GuidanceSystem
        Guidance subsystem
    intercepted : bool
        Whether missile has intercepted something
    intercept_type : str or None
        Type of intercept: 'real', 'decoy_N', or None
    """
    
    def __init__(self, pos, vel, rng):
        """
        Initialize missile.
        
        Parameters
        ----------
        pos : np.ndarray
            Initial position (m)
        vel : np.ndarray
            Initial velocity (m/s)
        rng : np.random.Generator
            Random number generator
        """
        self.pos = pos.astype(float).copy()
        self.vel = vel.astype(float).copy()
        self.mass = MISS_MASS_INITIAL
        self.launched = False
        self.launch_time = 0.0
        
        # Initialize seeker with boresight along initial velocity
        self.seeker = Seeker(rng, initial_boresight=vel)
        self.guidance = GuidanceSystem()
        
        self.intercepted = False
        self.intercept_type = None
    
    def get_thrust(self, t_flight):
        """
        Get current thrust based on propulsion phase.
        
        Three phases:
        1. Boost: High thrust (0 to MISS_BOOST_TIME)
        2. Sustain: Lower thrust (MISS_BOOST_TIME to MISS_BOOST_TIME + MISS_SUSTAIN_TIME)
        3. Coast: Zero thrust (after sustain)
        
        Parameters
        ----------
        t_flight : float
            Time since launch (s)
            
        Returns
        -------
        float
            Thrust force (N)
        """
        if t_flight < MISS_BOOST_TIME:
            return MISS_BOOST_THRUST
        elif t_flight < MISS_BOOST_TIME + MISS_SUSTAIN_TIME:
            return MISS_SUSTAIN_THRUST
        else:
            return 0.0
    
    def get_mass(self, t_flight):
        """
        Get current mass accounting for fuel depletion.
        
        Assumes linear fuel consumption during boost and sustain phases.
        
        Parameters
        ----------
        t_flight : float
            Time since launch (s)
            
        Returns
        -------
        float
            Current mass (kg)
        """
        total_burn = MISS_BOOST_TIME + MISS_SUSTAIN_TIME
        
        if t_flight >= total_burn:
            return MISS_MASS_BURNOUT
        
        # Linear interpolation from initial to burnout mass
        fuel_mass = MISS_MASS_INITIAL - MISS_MASS_BURNOUT
        burn_fraction = t_flight / total_burn
        return MISS_MASS_INITIAL - fuel_mass * burn_fraction

    def update(self, t, dt, target_pos, target_vel,
            decoy_positions, decoy_velocities, decoy_active):
        """
        Main missile update step.

        Performs:
        1. Mass and thrust calculation based on flight time
        2. Seeker update to determine lock
        3. Guidance update to compute acceleration command
        4. G-limiting of commanded acceleration
        5. Propulsion (thrust along velocity)
        6. Aerodynamic drag calculation
        7. Integration of equations of motion

        Parameters
        ----------
        t : float
            Current time (s)
        dt : float
            Time step (s)
        target_pos : np.ndarray
            Target position (m)
        target_vel : np.ndarray
            Target velocity (m/s)
        decoy_positions : list of np.ndarray
            Decoy positions (m)
        decoy_velocities : list of np.ndarray
            Decoy velocities (m/s)
        decoy_active : list of bool
            Decoy active flags

        Returns
        -------
        str or None
            Current lock type: 'target', 'decoy_N', or None
        """
        # Don't update if already intercepted or not launched
        if self.intercepted:
            return self.seeker.current_lock
        if not self.launched:
            return None

        # Flight time and current altitude
        t_flight = t - self.launch_time
        altitude = max(0.0, self.pos[2])

        # ----- Update mass and get thrust -----
        self.mass = self.get_mass(t_flight)
        thrust_mag = self.get_thrust(t_flight)

        # ------------------------------------------------------------------
        # END-OF-LIFE / "SPENT" LOGIC
        # ------------------------------------------------------------------
        speed = np.linalg.norm(self.vel)

        # Geometry to check whether we've flown past the target
        r_vec = target_pos - self.pos
        R = np.linalg.norm(r_vec)

        closing_vel = 0.0
        if R > 1e-6 and speed > 1e-3:
            los_hat = r_vec / R
            rel_vel = target_vel - self.vel
            # positive when we are closing
            closing_vel = -np.dot(rel_vel, los_hat)

        # Conditions under which the missile is essentially done:
        #  - flight time exceeded
        #  - speed too low
        #  - we're opening (closing_vel < 0) and not in a knife-fight (R > MISS_MIN_CLOSING_R)
        spent = (
            t_flight > MISS_MAX_FLIGHT_TIME
            or speed < MISS_MIN_SPEED
            or (closing_vel < 0.0 and R > MISS_MIN_CLOSING_R)
        )

        # Default command & lock type
        accel_cmd = np.zeros(3)
        lock_type = self.seeker.current_lock

        if spent and not self.intercepted:
            # Drop lock and stop commanding guidance; coast ballistically
            self.seeker.current_lock = None
            has_lock = False
            lock_type = None
        else:
            # ----- Seeker + Guidance -----
            locked_pos, locked_vel, lock_type = self.seeker.update(
                t, dt, self.pos, self.vel, target_pos, target_vel,
                decoy_positions, decoy_velocities, decoy_active
            )
            has_lock = (locked_pos is not None)

            if has_lock:
                # TERMINAL / APN
                accel_cmd = self.guidance.update(
                    t, dt, self.pos, self.vel, locked_pos, locked_vel, True
                )
            else:
                # MIDCOURSE toward real target
                accel_cmd = self.guidance.update(
                    t, dt, self.pos, self.vel, target_pos, target_vel, False
                )

        # ----- G-limiting -----
        accel_mag = np.linalg.norm(accel_cmd)
        max_accel = MISS_MAX_G * 9.81
        if accel_mag > max_accel:
            accel_cmd = accel_cmd * (max_accel / accel_mag)

        # ----- Propulsion -----
        # Thrust acts along velocity vector
        speed = np.linalg.norm(self.vel)
        if speed > 1e-6:
            thrust_dir = self.vel / speed
        else:
            thrust_dir = np.array([1.0, 0.0, 0.0])

        thrust_accel = (thrust_mag / self.mass) * thrust_dir

        # ----- Aerodynamic drag -----
        mach = get_mach_number(self.vel, altitude)
        cd = get_missile_cd(mach)
        drag_force = calculate_drag(self.vel, altitude, cd, MISS_REF_AREA)
        drag_accel = drag_force / self.mass

        # ----- Total acceleration -----
        # Sum of all forces: thrust + drag + guidance + gravity
        total_accel = thrust_accel + drag_accel + accel_cmd + GRAVITY

        # ----- Integrate equations of motion -----
        # Simple Euler integration (adequate for small dt)
        self.vel = self.vel + total_accel * dt
        self.pos = self.pos + self.vel * dt

        # Prevent missile from going below ground
        if self.pos[2] <= 0.0:
            self.pos[2] = 0.0
            self.vel[:] = 0.0
            # Treat as a ground impact if we haven't intercepted anything
            if not self.intercepted:
                self.intercepted = True
                self.intercept_type = 'ground'

        return lock_type


# ============================================================================
# SECTION 15: TARGET AIRCRAFT CLASS
# ============================================================================

class TargetAircraft:
    """
    Target aircraft with piecewise trajectory and evasive maneuvers.
    
    Trajectory segments:
    1. STRAIGHT_1: Straight flight in +X direction
    2. CURVE: Circular arc turn in tilted plane
    3. STRAIGHT_2: Straight flight along new heading
    4. COAST: Continue on final heading
    
    Evasive jink maneuvers are overlaid on the base trajectory
    after T_EVASION_START, with periodically randomized parameters.
    
    Attributes
    ----------
    start_pos : np.ndarray
        Initial position (m)
    pos : np.ndarray
        Current position (m)
    vel : np.ndarray
        Current velocity (m/s)
    rng : np.random.Generator
        Random number generator for jink
    segment : int
        Current trajectory segment (0-3)
    turn_radius : float
        Radius of turn (m)
    turn_center : np.ndarray
        Center of turn circle (m)
    turn_start_angle : float
        Starting angle on turn circle (rad)
    straight2_start_pos : np.ndarray
        Position at start of second straight segment (m)
    straight2_start_vel : np.ndarray
        Velocity at start of second straight segment (m/s)
    jink_amp : float
        Current jink amplitude (m)
    jink_freq : float
        Current jink frequency (Hz)
    jink_phase : float
        Current jink phase (rad)
    last_jink_update : float
        Time of last jink parameter update (s)
    """
    
    def __init__(self, start_pos, rng):
        """
        Initialize target aircraft.
        
        Parameters
        ----------
        start_pos : np.ndarray
            Starting position (m)
        rng : np.random.Generator
            Random number generator for jink randomization
        """
        self.start_pos = start_pos.astype(float).copy()
        self.pos = start_pos.astype(float).copy()
        self.vel = np.array([TARG_INITIAL_SPEED, 0.0, 0.0])
        self.rng = rng
        
        # Trajectory segment tracking
        self.segment = 0
        
        # Turn geometry (computed on first entry to turn segment)
        self.turn_radius = None
        self.turn_center = None
        self.turn_start_angle = None
        
        # Second straight segment start state
        self.straight2_start_pos = None
        self.straight2_start_vel = None
        
        # Jink parameters (randomized periodically)
        self.jink_amp = JINK_AMP_BASE
        self.jink_freq = JINK_FREQ_BASE
        self.jink_phase = 0.0
        self.last_jink_update = -999.0  # Force update on first call
    
        self.evasion_start = T_EVASION_START  # default; can be overridden
        
    def _update_jink_params(self, t):
        """
        Randomize jink parameters periodically.
        
        Creates unpredictable evasive maneuver pattern that is
        difficult for missile guidance to compensate for.
        
        Parameters
        ----------
        t : float
            Current time (s)
        """
        if t - self.last_jink_update > JINK_RANDOMIZE_INTERVAL:
            self.jink_amp = JINK_AMP_BASE + self.rng.uniform(-JINK_AMP_RANDOM, JINK_AMP_RANDOM)
            self.jink_freq = JINK_FREQ_BASE + self.rng.uniform(-JINK_FREQ_RANDOM, JINK_FREQ_RANDOM)
            self.jink_phase += self.rng.uniform(-np.pi / 4, np.pi / 4)
            self.last_jink_update = t
    
    def _get_base_position(self, t):
        """
        Calculate base trajectory position (without jink).
        
        Parameters
        ----------
        t : float
            Current time (s)
            
        Returns
        -------
        np.ndarray
            Base position (m)
        """
        speed = TARG_INITIAL_SPEED
        
        # ----- SEGMENT 0: First straight segment -----
        if t <= STRAIGHT_TIME_1:
            self.segment = 0
            # Straight flight in +X direction from start
            return self.start_pos + np.array([speed * t, 0.0, 0.0])
        
        # ----- SEGMENT 1: Curved turn -----
        elif t <= STRAIGHT_TIME_1 + CURVE_TIME:
            # Initialize turn geometry on first entry
            if self.segment != 1:
                self.segment = 1
                
                # Position at start of turn
                turn_start_pos = self.start_pos + np.array([speed * STRAIGHT_TIME_1, 0.0, 0.0])
                
                # Turn radius from: arc_length = speed × time = radius × |angle|
                self.turn_radius = (speed * CURVE_TIME) / abs(TURN_ANGLE)
                
                # Turn center offset perpendicular to initial heading
                # Offset is in YZ plane, tilted by YZ_ANGLE
                self.turn_center = turn_start_pos + np.array([
                    0.0,
                    self.turn_radius * np.cos(YZ_ANGLE),
                    self.turn_radius * np.sin(YZ_ANGLE)
                ])
                
                # Starting angle on circle (tangent points in +X direction)
                self.turn_start_angle = -np.pi / 2
            
            # Time elapsed in turn segment
            tc = t - STRAIGHT_TIME_1
            
            # Current angle on turn circle
            angle = self.turn_start_angle + tc * TURN_ANGLE / CURVE_TIME
            
            # Vertical shaping term (adds climb/dive variation)
            shaping = speed**2 * (1 - np.cos(np.pi * tc / CURVE_TIME)) * CLIMB_RATE_CURVE
            
            # Position on tilted circular arc
            x = self.turn_center[0] + self.turn_radius * np.cos(angle)
            y = (self.turn_center[1] + 
                 self.turn_radius * np.sin(angle) * np.cos(YZ_ANGLE) +
                 np.cos(YZ_ANGLE + np.pi / 2) * shaping)
            z = (self.turn_center[2] + 
                 self.turn_radius * np.sin(angle) * np.sin(YZ_ANGLE) +
                 np.sin(YZ_ANGLE + np.pi / 2) * shaping)
            
            return np.array([x, y, z])
        
        # ----- SEGMENT 2: Second straight segment -----
        elif t <= STRAIGHT_TIME_1 + CURVE_TIME + STRAIGHT_TIME_2:
            # Initialize second straight on first entry
            if self.segment != 2:
                self.segment = 2
                
                # Position at end of turn
                final_angle = self.turn_start_angle + TURN_ANGLE
                self.straight2_start_pos = np.array([
                    self.turn_center[0] + self.turn_radius * np.cos(final_angle),
                    self.turn_center[1] + self.turn_radius * np.sin(final_angle) * np.cos(YZ_ANGLE),
                    self.turn_center[2] + self.turn_radius * np.sin(final_angle) * np.sin(YZ_ANGLE)
                ])
                
                # Velocity direction after turn (rotated by TURN_ANGLE)
                self.straight2_start_vel = speed * np.array([
                    np.cos(TURN_ANGLE),
                    np.sin(TURN_ANGLE) * np.cos(YZ_ANGLE),
                    np.sin(TURN_ANGLE) * np.sin(YZ_ANGLE)
                ])
            
            # Time in second straight segment
            ts = t - (STRAIGHT_TIME_1 + CURVE_TIME)
            return self.straight2_start_pos + self.straight2_start_vel * ts
        
        # ----- SEGMENT 3: Coast (continue on final heading) -----
        else:
            if self.segment != 3:
                self.segment = 3
                # Ensure segment 2 values are initialized
                if self.straight2_start_pos is None:
                    self.straight2_start_pos = self.start_pos.copy()
                    self.straight2_start_vel = np.array([speed, 0.0, 0.0])
            
            # Time past end of segment 2
            ts = t - (STRAIGHT_TIME_1 + CURVE_TIME + STRAIGHT_TIME_2)
            
            # Position at end of segment 2
            end_of_seg2 = self.straight2_start_pos + self.straight2_start_vel * STRAIGHT_TIME_2
            
            # Continue on same heading
            return end_of_seg2 + self.straight2_start_vel * ts
    
    def update(self, t, dt):
        """
        Update aircraft position with base trajectory + evasive jink.
        
        Parameters
        ----------
        t : float
            Current time (s)
        dt : float
            Time step (s)
            
        Returns
        -------
        np.ndarray
            Current position (m)
        """
        # Get base trajectory position
        base_pos = self._get_base_position(t)
        
        # Add evasive jink after evasion start time
        if t > self.evasion_start:
            self._update_jink_params(t)
            
            # Time since evasion started
            jink_t = t - T_EVASION_START
            
            # Sinusoidal jink offset
            jink_offset = self.jink_amp * np.sin(
                2 * np.pi * self.jink_freq * jink_t + self.jink_phase
            )
            
            # Apply jink perpendicular to flight path (in tilted plane)
            jink_dir = np.array([0.0, np.cos(YZ_ANGLE), np.sin(YZ_ANGLE)])
            base_pos = base_pos + jink_offset * jink_dir
        
        # Update velocity estimate from position change
        if dt > 0:
            self.vel = (base_pos - self.pos) / dt
        
        self.pos = base_pos.copy()
        return self.pos


# ============================================================================
# SECTION 16: DECOY CLASS
# ============================================================================

class Decoy:
    """
    Countermeasure decoy with aerodynamic drag.
    
    After deployment from aircraft:
    - Decoy drifts with random initial velocity offset
    - Experiences aerodynamic drag (decelerates)
    - Affected by gravity (falls)
    
    Attributes
    ----------
    pos : np.ndarray
        Current position (m)
    vel : np.ndarray
        Current velocity (m/s)
    deploy_time : float
        Time when decoy deploys (s)
    active : bool
        Whether decoy has been deployed
    mass : float
        Decoy mass (kg)
    drift_vel : np.ndarray
        Pre-computed random drift velocity (m/s)
    """
    
    def __init__(self, deploy_time, rng):
        """
        Initialize decoy.
        
        Parameters
        ----------
        deploy_time : float
            Time when decoy will be deployed (s)
        rng : np.random.Generator
            Random number generator
        """
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.deploy_time = deploy_time
        self.active = False
        self.mass = DECOY_MASS
        # Time window during which this decoy is "hot" (emitting strongly)
        self.emission_end_time = deploy_time + DECOY_BURN_TIME
        
        # Pre-compute random drift direction and speed
        drift_dir = rng.standard_normal(3)
        norm = np.linalg.norm(drift_dir)
        if norm > 1e-9:
            drift_dir = drift_dir / norm
        else:
            drift_dir = np.array([0.0, 1.0, 0.0])
        
        drift_speed = max(0.0, DECOY_BASE_SPEED + DECOY_SPEED_SIGMA * rng.standard_normal())
        self.drift_vel = drift_dir * drift_speed
    
    def deploy(self, aircraft_pos, aircraft_vel):
        """
        Deploy decoy from aircraft.
        
        Decoy starts at aircraft position with aircraft velocity
        plus random drift velocity.
        
        Parameters
        ----------
        aircraft_pos : np.ndarray
            Aircraft position at deployment (m)
        aircraft_vel : np.ndarray
            Aircraft velocity at deployment (m/s)
        """
        self.pos = aircraft_pos.copy()
        self.vel = aircraft_vel.copy() + self.drift_vel
        self.active = True
    
    def update(self, t, dt, aircraft_pos, aircraft_vel):
        """
        Update decoy state.
        
        Parameters
        ----------
        t : float
            Current time (s)
        dt : float
            Time step (s)
        aircraft_pos : np.ndarray
            Current aircraft position (for deployment reference)
        aircraft_vel : np.ndarray
            Current aircraft velocity (for deployment reference)
        """
        # Handle deployment / burn-time window
        # Check for decoy deployment 
        if t >= self.deploy_time and t <= self.emission_end_time:
            if not self.active:
                # First time we enter the emission window: deploy
                self.deploy(aircraft_pos, aircraft_vel)
        elif t > self.emission_end_time:
            # Decoy has burned out – no longer an attractive target
            self.active = False
        
        # Skip update if not active
        # Still integrate motion even if not radiating
        if not self.active and t < self.deploy_time:
            # Not yet released – just sit on the aircraft
            self.pos = aircraft_pos.copy()
            self.vel = aircraft_vel.copy()
            return
        
        # Get current altitude for drag calculation
        altitude = max(0.0, self.pos[2])
        
        # Calculate drag acceleration
        # Using Cd = 1.0 for high-drag decoy
        drag_force = calculate_drag(self.vel, altitude, 1.0, DECOY_DRAG_AREA)
        drag_accel = drag_force / self.mass
        
        # Total acceleration: gravity + drag
        total_accel = GRAVITY + drag_accel
        
        # Integrate equations of motion
        self.vel = self.vel + total_accel * dt
        self.pos = self.pos + self.vel * dt
        
        # Stop at the ground
        if self.pos[2] <= 0.0:
            self.pos[2] = 0.0
            self.vel[:] = 0.0


# ============================================================================
# SECTION 17: RANDOMIZED START POSITION SAMPLING
# ============================================================================

def sample_random_starts(rng):
    """
    Sample randomized starting positions for engagement geometry.
    
    Creates realistic head-on or crossing engagement scenarios where
    missile is positioned AHEAD of the aircraft.
    
    Aircraft:
        - XY position: within 3km of origin
        - Altitude: 9-13 km (typical combat altitude)
        - Initial heading: +X direction
    
    Missile:
        - Positioned in forward hemisphere of aircraft
        - Range: 15-30 km from aircraft
        - Altitude: 8-14 km
    
    Parameters
    ----------
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    tuple
        (aircraft_start, missile_start) as np.ndarray
    """
    # ----- Aircraft starting position -----
    aircraft_alt = rng.uniform(9000.0, 13000.0)
    r_ac = rng.uniform(0.0, 3000.0)
    bearing_ac = rng.uniform(0.0, 2 * np.pi)
    ax0 = r_ac * np.cos(bearing_ac)
    ay0 = r_ac * np.sin(bearing_ac)
    aircraft_start = np.array([ax0, ay0, aircraft_alt])
    
    # ----- Missile starting position -----
    # FIXED: Missile placed AHEAD of aircraft (forward hemisphere)
    # bearing = 0 is directly ahead, within ±60° cone
    r = rng.uniform(8000.0, 15000.0)  # instead of 15000–30000
    bearing = rng.uniform(-np.pi / 3, np.pi / 3)  # Forward hemisphere
    miss_alt = rng.uniform(8000.0, 14000.0)
    
    # Missile position relative to aircraft
    miss_x = aircraft_start[0] + r * np.cos(bearing)
    miss_y = aircraft_start[1] + r * np.sin(bearing)
    missile_start = np.array([miss_x, miss_y, miss_alt])
    
    return aircraft_start, missile_start


# ============================================================================
# SECTION 18: MAIN SIMULATION FUNCTION
# ============================================================================

def simulate_engagement(aircraft_start, missile_start, rng=None,
                        make_plot=False, verbose=True,
                        run_id=None, total_runs=None):
    """
    Run a single missile-target engagement simulation.
    
    Parameters
    ----------
    aircraft_start : np.ndarray
        Aircraft starting position (m)
    missile_start : np.ndarray
        Missile starting position (m)
    rng : np.random.Generator, optional
        Random number generator. Creates new one if None.
    make_plot : bool
        Whether to create animated visualization
    verbose : bool
        Whether to print status messages
    run_id : int, optional
        Run number for display
    total_runs : int, optional
        Total runs for display
        
    Returns
    -------
    dict
        Engagement results:
        - intercept_real: bool
        - intercept_decoy: bool
        - intercept_type: 'real', 'decoy', or None
        - intercept_time: float or None
        - closest_miss: float (minimum distance to real target)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Time array
    times = np.arange(0, TMAX, DT)
    n_points = len(times)
    
    # ----- Initialize target -----
    target = TargetAircraft(aircraft_start, rng)

    # ----- Initial missile velocity toward target -----
    # Line-of-sight from missile to aircraft at t=0
    rel = aircraft_start - missile_start
    R0 = np.linalg.norm(rel)
    if R0 > 1e-6:
        los_hat = rel / R0
    else:
        los_hat = np.array([1.0, 0.0, 0.0])

    # Start missile around ~Mach 1-ish (300 m/s) along LOS to target
    initial_speed = 300.0
    initial_vel = initial_speed * los_hat

    # ----- Create missile -----
    missile = Missile(missile_start, initial_vel, rng)
    missile.launched = True
    missile.launch_time = MISSILE_LAUNCH_TIME

    # ----- Rough initial time-to-go estimate -----
    # Relative velocity (target minus missile) at t=0
    v_rel0 = (target.vel - missile.vel)
    closing0 = -np.dot(v_rel0, los_hat)  # >0 when closing
    closing0 = max(closing0, 1.0)        # avoid divide-by-zero / non-closing

    t_go_est = R0 / closing0

    # Dynamic timing:
    # - Start jinking at ~40% of predicted TOF
    # - Drop decoys around ~60% of predicted TOF with small random jitter
    target.evasion_start = 0.4 * t_go_est

    decoys = []
    for _ in range(N_DECOYS):
        deploy_time = 0.6 * t_go_est + rng.uniform(-1.0, 1.0)
        decoys.append(Decoy(deploy_time, rng))

    # ----- Storage arrays for visualization -----
    target_states = np.zeros((n_points, 3))
    target_vels = np.zeros((n_points, 3))
    missile_states = np.zeros((n_points, 3))
    missile_vels = np.zeros((n_points, 3))
    missile_speeds = np.zeros(n_points)
    decoy_states = np.full((N_DECOYS, n_points, 3), np.nan)
    decoy_active = np.zeros((N_DECOYS, n_points), dtype=bool)
    lock_history = np.full(n_points, -1, dtype=int)  # -1=none, 0=target, 1+=decoy
    
    # ----- Result tracking -----
    intercept_time = None
    intercept_index = None
    intercept_type = None
    intercept_decoy_id = None
    closest_miss = np.inf
    
    # ----- Main simulation loop -----
    for i, t in enumerate(times):
        # Update target
        target.update(t, DT)
        target_states[i] = target.pos.copy()
        target_vels[i] = target.vel.copy()
        
        # Update decoys
        decoy_positions = []
        decoy_velocities = []
        decoy_active_list = []
        
        for d_idx, decoy in enumerate(decoys):
            decoy.update(t, DT, target.pos, target.vel)
            decoy_states[d_idx, i] = decoy.pos.copy()
            decoy_active[d_idx, i] = decoy.active
            decoy_positions.append(decoy.pos.copy())
            decoy_velocities.append(decoy.vel.copy())
            decoy_active_list.append(decoy.active)
        
        # Update missile
        if not missile.intercepted:
            lock_type = missile.update(
                t, DT, target.pos, target.vel,
                decoy_positions, decoy_velocities, decoy_active_list
            )
            
            # Record lock state
            if lock_type == 'target':
                lock_history[i] = 0
            elif lock_type and lock_type.startswith('decoy_'):
                lock_history[i] = int(lock_type.split('_')[1]) + 1
            else:
                lock_history[i] = -1
            
            # ----- Check for intercepts -----
            # Distance to real target
            dist_to_target = np.linalg.norm(target.pos - missile.pos)
            closest_miss = min(closest_miss, dist_to_target)
            
            # Check real target intercept
            if dist_to_target < KILL_DIST:
                missile.intercepted = True
                missile.intercept_type = 'real'
                intercept_time = t
                intercept_index = i
                intercept_type = 'real'
                if verbose:
                    print(f"  HIT TARGET at t={t:.2f}s, d={dist_to_target:.1f}m")
            else:
                # Check decoy intercepts
                for d_idx, decoy in enumerate(decoys):
                    if decoy.active:
                        dist_to_decoy = np.linalg.norm(decoy.pos - missile.pos)
                        if dist_to_decoy < KILL_DIST:
                            missile.intercepted = True
                            missile.intercept_type = f'decoy_{d_idx}'
                            intercept_time = t
                            intercept_index = i
                            intercept_type = 'decoy'
                            intercept_decoy_id = d_idx
                            if verbose:
                                print(f"  DECOY {d_idx + 1} HIT at t={t:.2f}s")
                            break
        
        # Store missile state
        missile_states[i] = missile.pos.copy()
        missile_vels[i] = missile.vel.copy()
        missile_speeds[i] = np.linalg.norm(missile.vel)
    
    # ----- Final results -----
    final_distance = np.linalg.norm(target_states[-1] - missile_states[-1])
    
    if verbose:
        print(f"  Final: {final_distance:.0f}m, Closest: {closest_miss:.0f}m")
        if intercept_type is None:
            print(f"  RESULT: MISS")
        elif intercept_type == 'real':
            print(f"  RESULT: TARGET HIT at t={intercept_time:.2f}s")
        elif intercept_type == 'decoy':
            print(f"  RESULT: DECOY {intercept_decoy_id + 1} HIT")
        elif intercept_type == 'ground':
            print(f"  RESULT: MISSILE IMPACTED GROUND")
    
    result = {
        'intercept_real': intercept_type == 'real',
        'intercept_decoy': intercept_type == 'decoy',
        'intercept_type': intercept_type,
        'intercept_time': intercept_time,
        'closest_miss': closest_miss,
    }
    
    # ----- Visualization -----
    if make_plot:
        create_animation(
            times, target_states, missile_states, missile_speeds,
            decoy_states, decoy_active, lock_history,
            intercept_index, intercept_type, intercept_decoy_id,
            aircraft_start, missile_start, run_id, total_runs
        )
    
    return result


# ============================================================================
# SECTION 19: ANIMATION/VISUALIZATION
# ============================================================================

# Global reference to prevent garbage collection of animation
_current_animation = None


def create_animation(times, target_states, missile_states, missile_speeds,
                     decoy_states, decoy_active, lock_history,
                     intercept_index, intercept_type, intercept_decoy_id,
                     aircraft_start, missile_start, run_id, total_runs):
    """
    Create 3D animated visualization of the engagement.
    
    Features:
    - 3D plot with ground plane reference
    - Target and missile trajectories with trails
    - Decoy visualization
    - Start position markers
    - Intercept marker (if applicable)
    - Real-time HUD showing: time, range, Mach, lock status, guidance mode
    - Color-coded missile based on lock state (red=target, magenta=decoy, gray=none)
    
    Parameters
    ----------
    times : np.ndarray
        Time array (s)
    target_states : np.ndarray
        Target positions over time (n_points, 3)
    missile_states : np.ndarray
        Missile positions over time (n_points, 3)
    missile_speeds : np.ndarray
        Missile speeds over time (n_points,)
    decoy_states : np.ndarray
        Decoy positions over time (n_decoys, n_points, 3)
    decoy_active : np.ndarray
        Decoy active flags (n_decoys, n_points)
    lock_history : np.ndarray
        Lock state over time (n_points,)
    intercept_index : int or None
        Time index of intercept
    intercept_type : str or None
        Type of intercept
    intercept_decoy_id : int or None
        ID of intercepted decoy
    aircraft_start : np.ndarray
        Aircraft starting position
    missile_start : np.ndarray
        Missile starting position
    run_id : int or None
        Run number for display
    total_runs : int or None
        Total runs for display
    """
    global _current_animation
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # ----- Compute plot bounds -----
    all_points = np.vstack([target_states, missile_states])
    for d in range(N_DECOYS):
        valid = decoy_active[d]
        if np.any(valid):
            valid_points = decoy_states[d, valid]
            # Filter out NaN values
            valid_points = valid_points[~np.isnan(valid_points).any(axis=1)]
            if len(valid_points) > 0:
                all_points = np.vstack([all_points, valid_points])
    
    # Calculate ranges and center
    max_range = max(np.ptp(all_points[:, 0]), np.ptp(all_points[:, 1]), 
                    np.ptp(all_points[:, 2]), 1000)
    center = (all_points.max(axis=0) + all_points.min(axis=0)) / 2
    radius = max_range / 2 * 1.1
    
    # Set axis limits (equal aspect ratio)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0, center[2] - radius), center[2] + radius)
    ax.set_box_aspect([1, 1, 1])
    
    # Labels and view
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.grid(True)
    ax.view_init(elev=BASE_ELEV, azim=BASE_AZIM)
    
    # ----- Ground plane -----
    ground_size = max_range * 1.2
    gx = np.linspace(center[0] - ground_size / 2, center[0] + ground_size / 2, 10)
    gy = np.linspace(center[1] - ground_size / 2, center[1] + ground_size / 2, 10)
    GX, GY = np.meshgrid(gx, gy)
    ax.plot_surface(GX, GY, np.zeros_like(GX), alpha=0.1, linewidth=0)
    
    # ----- Static elements -----
    # Full target path (reference)
    ax.plot(target_states[:, 0], target_states[:, 1], target_states[:, 2],
            'b--', linewidth=1, alpha=0.3)
    
    # Start markers
    ax.scatter(*aircraft_start, c='green', s=100, marker='s', zorder=5)
    ax.scatter(*missile_start, c='orange', s=100, marker='^', zorder=5)
    
    # Intercept marker
    if intercept_index is not None:
        if intercept_type == 'real':
            ax.scatter(*target_states[intercept_index], c='red', s=300, 
                      marker='*', zorder=10)
        elif intercept_type == 'decoy' and intercept_decoy_id is not None:
            ax.scatter(*decoy_states[intercept_decoy_id, intercept_index],
                      c='magenta', s=300, marker='*', zorder=10)
    
    # ----- Animated elements -----
    target_point, = ax.plot([], [], [], 'bo', markersize=10)
    target_trail, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.7)
    missile_point, = ax.plot([], [], [], 'ro', markersize=8)
    missile_trail, = ax.plot([], [], [], 'r-', linewidth=2, alpha=0.7)
    
    decoy_points = []
    for d in range(N_DECOYS):
        p, = ax.plot([], [], [], 'y*', markersize=12, alpha=0.9)
        decoy_points.append(p)
    
    # Legend (positioned to avoid HUD overlap)
    ax.legend(['Target Path', 'Aircraft Start', 'Missile Start'],
              loc='upper left', fontsize=8, framealpha=0.8)
    
    # ----- HUD text elements -----
    hud_x = 0.72  # Right side of plot
    time_txt = ax.text2D(hud_x, 0.95, '', transform=ax.transAxes, 
                         fontsize=10, family='monospace')
    range_txt = ax.text2D(hud_x, 0.91, '', transform=ax.transAxes, 
                          fontsize=10, family='monospace')
    speed_txt = ax.text2D(hud_x, 0.87, '', transform=ax.transAxes, 
                          fontsize=10, family='monospace')
    lock_txt = ax.text2D(hud_x, 0.83, '', transform=ax.transAxes, 
                         fontsize=10, family='monospace', color='purple')
    mode_txt = ax.text2D(hud_x, 0.79, '', transform=ax.transAxes, 
                         fontsize=10, family='monospace', color='darkgreen')
    status_txt = ax.text2D(0.35, 0.95, '', transform=ax.transAxes, 
                           fontsize=12, fontweight='bold', color='red')
    
    # Title with run number
    run_label = f'Run {run_id}/{total_runs}' if run_id and total_runs else ''
    ax.set_title(f'3D Missile-Target Engagement    {run_label}', fontsize=12)
    
    def init():
        """Initialize animation - clear all artists."""
        for artist in [target_point, target_trail, missile_point, missile_trail] + decoy_points:
            artist.set_data([], [])
            artist.set_3d_properties([])
        for txt in [time_txt, range_txt, speed_txt, lock_txt, mode_txt, status_txt]:
            txt.set_text('')
        return (target_point, target_trail, missile_point, missile_trail,
                *decoy_points, time_txt, range_txt, speed_txt, lock_txt, 
                mode_txt, status_txt)
    
    def update(frame):
        """Update animation for given frame."""
        # ----- Target -----
        target_point.set_data([target_states[frame, 0]], [target_states[frame, 1]])
        target_point.set_3d_properties([target_states[frame, 2]])
        target_trail.set_data(target_states[:frame + 1, 0], target_states[:frame + 1, 1])
        target_trail.set_3d_properties(target_states[:frame + 1, 2])
        
        # ----- Missile -----
        missile_point.set_data([missile_states[frame, 0]], [missile_states[frame, 1]])
        missile_point.set_3d_properties([missile_states[frame, 2]])
        missile_trail.set_data(missile_states[:frame + 1, 0], missile_states[:frame + 1, 1])
        missile_trail.set_3d_properties(missile_states[:frame + 1, 2])
        
        # Color missile by lock state
        lock = lock_history[frame]
        if lock == 0:
            color = 'red'       # Locked on target
        elif lock > 0:
            color = 'magenta'   # Locked on decoy
        else:
            color = 'gray'      # No lock
        missile_point.set_color(color)
        missile_trail.set_color(color)
        
        # ----- Decoys -----
        for d in range(N_DECOYS):
            if decoy_active[d, frame]:
                pos = decoy_states[d, frame]
                if not np.any(np.isnan(pos)):
                    decoy_points[d].set_data([pos[0]], [pos[1]])
                    decoy_points[d].set_3d_properties([pos[2]])
                else:
                    decoy_points[d].set_data([], [])
                    decoy_points[d].set_3d_properties([])
            else:
                decoy_points[d].set_data([], [])
                decoy_points[d].set_3d_properties([])
        
        # ----- HUD updates -----
        dist = np.linalg.norm(target_states[frame] - missile_states[frame])
        mach = missile_speeds[frame] / get_speed_of_sound(missile_states[frame, 2])
        
        time_txt.set_text(f'T: {times[frame]:.1f}s')
        range_txt.set_text(f'R: {dist / 1000:.1f}km')
        speed_txt.set_text(f'M: {mach:.1f}')
        lock_txt.set_text(f'Lock: {"TGT" if lock == 0 else f"D{lock}" if lock > 0 else "---"}')
        mode_txt.set_text(f'LockMode: {"TRM" if lock >= 0 else "MID"}')
        
        # Intercept status
        if intercept_index is not None and frame >= intercept_index:
            if intercept_type == 'real':
                status_txt.set_text('HIT')
            else:
                status_txt.set_text('DECOY HIT')
            missile_point.set_marker('x')
            missile_point.set_markersize(12)
        else:
            status_txt.set_text('')
            missile_point.set_marker('o')
            missile_point.set_markersize(8)
        
        return (target_point, target_trail, missile_point, missile_trail,
                *decoy_points, time_txt, range_txt, speed_txt, lock_txt,
                mode_txt, status_txt)
    
    # ----- Create animation -----
    frame_skip = max(1, len(times) // ANIMATION_MAX_FRAMES)
    frames = list(range(0, len(times), frame_skip))
    
    _current_animation = FuncAnimation(
        fig, update,
        frames=frames,
        init_func=init,
        blit=False,
        interval=ANIMATION_INTERVAL,
        repeat=True
    )
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# SECTION 20: MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Create random number generator
    # Use fixed seed for reproducibility: rng = np.random.default_rng(42)
    rng = np.random.default_rng()
    
    # ----- Monte Carlo simulation -----
    n_trials = 10
    n_real = 0
    n_decoy = 0
    n_miss = 0
    closest_misses = []
    
    print("=" * 60)
    print("REALISTIC 3D MISSILE-TARGET ENGAGEMENT SIMULATION")
    print("=" * 60)
    print(f"\nRunning {n_trials} randomized engagements...\n")
    
    for k in range(n_trials):
        # Sample random start positions
        a_start, m_start = sample_random_starts(rng)
        
        print(f"Trial {k + 1}/{n_trials}:")
        print(f"  Aircraft: [{a_start[0]:.0f}, {a_start[1]:.0f}, {a_start[2]:.0f}]")
        print(f"  Missile:  [{m_start[0]:.0f}, {m_start[1]:.0f}, {m_start[2]:.0f}]")
        
        # Run simulation
        result = simulate_engagement(
            aircraft_start=a_start,
            missile_start=m_start,
            rng=rng,
            make_plot=True,
            verbose=True,
            run_id=k + 1,
            total_runs=n_trials
        )
        
        # Tally results
        if result['intercept_real']:
            n_real += 1
        elif result['intercept_decoy']:
            n_decoy += 1
        else:
            n_miss += 1
        
        closest_misses.append(result['closest_miss'])
        print()
    
    # ----- Print summary -----
    print("=" * 60)
    print("MONTE CARLO RESULTS")
    print("=" * 60)
    print(f"  Target hit:   {n_real:3d}/{n_trials} ({100 * n_real / n_trials:.0f}%)")
    print(f"  Decoy hit:    {n_decoy:3d}/{n_trials} ({100 * n_decoy / n_trials:.0f}%)")
    print(f"  Miss:         {n_miss:3d}/{n_trials} ({100 * n_miss / n_trials:.0f}%)")
    print(f"  Avg closest:  {np.mean(closest_misses):.0f}m")
    print("=" * 60)