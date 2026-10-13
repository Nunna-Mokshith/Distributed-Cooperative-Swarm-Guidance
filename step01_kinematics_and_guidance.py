"""
Distributed Cooperative Swarm Guidance (DCSG)
Module 1: Vehicle Kinematics & Classical Proportional Navigation (PN) Baseline
File: 01_applied_ml/distributed_swarm_guidance/step01_kinematics_and_guidance.py

Candidate: Nunna Mokshith (Robotics & AI)
Target: Implement the fundamental physical equations of motion and classical GNC guidance.

OPERATING RULES:
- Do NOT alter typed function signatures.
- Implement all methods marked with `raise NotImplementedError`.
- Ensure numerical stability (avoid division by zero when R -> 0).
- Run the self-audit assertions at the bottom to verify intercept dynamics.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import math
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("KinematicsEngine")

# Standard gravity (m/s^2) for G-load calculations
G_ACCEL: float = 9.80665


@dataclass(frozen=True)
class MissileState:
    """Represents the 2D kinematic state of a guided missile."""
    x: float                     # X position in inertial frame (meters)
    y: float                     # Y position in inertial frame (meters)
    heading: float               # Heading angle theta from X-axis (radians)
    velocity: float              # Scalar forward velocity v (m/s)
    max_lateral_accel: float     # Maximum normal acceleration limit (m/s^2, e.g. 20G)

    @property
    def vx(self) -> float:
        """Velocity component along inertial X-axis."""
        return self.velocity * math.cos(self.heading)

    @property
    def vy(self) -> float:
        """Velocity component along inertial Y-axis."""
        return self.velocity * math.sin(self.heading)


@dataclass(frozen=True)
class TargetState:
    """Represents the kinematic state of an aerial/ground target."""
    target_id: int
    x: float                     # Target X position (meters)
    y: float                     # Target Y position (meters)
    vx: float                    # Target velocity along X (m/s)
    vy: float                    # Target velocity along Y (m/s)
    priority: float              # Tactical priority score (e.g. 1.0 to 10.0)


def wrap_to_pi(angle: float) -> float:
    """Wraps an angle in radians to the interval [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def calculate_los_geometry(
    missile: MissileState,
    target: TargetState,
    epsilon: float = 1e-5,
) -> Tuple[float, float, float]:
    """
    Computes the instantaneous Line-of-Sight (LOS) engagement geometry.

    Args:
        missile: Current kinematic state of the missile.
        target: Current kinematic state of the target.
        epsilon: Small numerical guard to prevent zero-division near intercept.

    Returns:
        Tuple containing:
            1. range_m (float): Euclidean distance R between missile and target (meters).
            2. closing_velocity (float): Closing speed V_c = -dR/dt (m/s). Must be positive when closing.
            3. los_rate (float): Time derivative of line-of-sight angle (d_lambda/dt) in rad/s.

    Formulation:
        delta_x = x_t - x_m
        delta_y = y_t - y_m
        R = sqrt(delta_x^2 + delta_y^2)
        delta_vx = vx_t - vx_m
        delta_vy = vy_t - vy_m
        V_c = - (delta_x * delta_vx + delta_y * delta_vy) / max(R, epsilon)
        d_lambda/dt = (delta_x * delta_vy - delta_y * delta_vx) / max(R^2, epsilon^2)
    """
    delta_x = target.x - missile.x
    delta_y = target.y - missile.y
    range_m = math.hypot(delta_x, delta_y)

    delta_vx = target.vx - missile.vx
    delta_vy = target.vy - missile.vy

    safe_r = max(range_m, epsilon)
    safe_r_sq = max(range_m * range_m, epsilon * epsilon)

    closing_velocity = -(delta_x * delta_vx + delta_y * delta_vy) / safe_r
    los_rate = (delta_x * delta_vy - delta_y * delta_vx) / safe_r_sq

    return range_m, closing_velocity, los_rate


def proportional_navigation_law(
    nav_constant: float,
    closing_velocity: float,
    los_rate: float,
    max_lateral_accel: float,
) -> float:
    """
    Computes commanded lateral acceleration using Pure Proportional Navigation (PPN).

    The classical PN guidance command is:
        a_cmd = N' * V_c * (d_lambda / dt)

    Args:
        nav_constant: Navigation ratio N' (dimensionless, typically between 3.0 and 5.0).
        closing_velocity: V_c in m/s.
        los_rate: d_lambda/dt in rad/s.
        max_lateral_accel: Max allowable fin aerodynamic acceleration limit (m/s^2).

    Returns:
        float: Clamped commanded normal acceleration a_cmd within [-max_accel, +max_accel].
    """
    a_raw = nav_constant * closing_velocity * los_rate
    return max(-max_lateral_accel, min(max_lateral_accel, a_raw))


def step_kinematics(
    state: MissileState,
    commanded_lateral_accel: float,
    dt: float,
) -> MissileState:
    """
    Integrates the 2D Dubins unicycle equations of motion over timestep dt.

    Equations of motion:
        x_{k+1} = x_k + v * cos(theta_k) * dt
        y_{k+1} = y_k + v * sin(theta_k) * dt
        theta_{k+1} = wrap_to_pi(theta_k + (a_cmd / v) * dt)

    Args:
        state: Current missile state at time t.
        commanded_lateral_accel: Commanded normal acceleration a_n (m/s^2).
        dt: Integration step size in seconds (e.g., 0.05s = 20 Hz).

    Returns:
        MissileState: New immutable state at time t + dt.
    """
    x_dot = state.velocity * math.cos(state.heading)
    y_dot = state.velocity * math.sin(state.heading)
    theta_dot = commanded_lateral_accel / state.velocity

    new_x = state.x + x_dot * dt
    new_y = state.y + y_dot * dt
    new_heading = wrap_to_pi(state.heading + theta_dot * dt)

    return MissileState(
        x=new_x,
        y=new_y,
        heading=new_heading,
        velocity=state.velocity,
        max_lateral_accel=state.max_lateral_accel,
    )


def simulate_single_engagement(
    initial_missile: MissileState,
    target: TargetState,
    nav_constant: float = 3.5,
    dt: float = 0.05,
    max_time_s: float = 30.0,
    lethal_radius_m: float = 10.0,
) -> Dict[str, Any]:
    """
    Runs a deterministic simulation loop of a single missile intercepting a target.

    Returns telemetry summary dict containing:
        - intercepted (bool)
        - flight_time_s (float)
        - final_miss_distance_m (float)
        - peak_g_load (float)
    """
    time_elapsed = 0.0
    current_missile = initial_missile
    current_target = target
    peak_accel = 0.0
    prev_range = float("inf")

    while time_elapsed < max_time_s:
        # 1. Update target position (constant velocity model)
        current_target = TargetState(
            target_id=current_target.target_id,
            x=current_target.x + current_target.vx * dt,
            y=current_target.y + current_target.vy * dt,
            vx=current_target.vx,
            vy=current_target.vy,
            priority=current_target.priority,
        )

        # 2. Compute LOS engagement geometry
        range_m, v_c, los_rate = calculate_los_geometry(current_missile, current_target)

        # 3. Intercept termination check (direct hit or closest point of approach within lethal radius)
        if range_m <= lethal_radius_m or (v_c <= 0.0 and range_m < prev_range and range_m <= lethal_radius_m * 1.5):
            logger.info(f"Target intercepted at t={time_elapsed:.2f}s! Final miss distance: {range_m:.3f}m")
            return {
                "intercepted": True,
                "flight_time_s": time_elapsed,
                "final_miss_distance_m": range_m,
                "peak_g_load": peak_accel / G_ACCEL,
            }

        prev_range = range_m

        # 4. Guidance calculation
        a_cmd = proportional_navigation_law(
            nav_constant=nav_constant,
            closing_velocity=v_c,
            los_rate=los_rate,
            max_lateral_accel=current_missile.max_lateral_accel,
        )
        if abs(a_cmd) > peak_accel:
            peak_accel = abs(a_cmd)

        # 5. Physics state update
        current_missile = step_kinematics(current_missile, a_cmd, dt)
        time_elapsed += dt

    # If timeout reached without intercept
    final_range, _, _ = calculate_los_geometry(current_missile, current_target)
    logger.warning(f"Engagement timed out. Final range: {final_range:.2f}m")
    return {
        "intercepted": False,
        "flight_time_s": time_elapsed,
        "final_miss_distance_m": final_range,
        "peak_g_load": peak_accel / G_ACCEL,
    }


if __name__ == "__main__":
    logger.info("Running baseline engagement sanity test...")

    # Scenario: Missile launched at origin, heading due East (0 rad) at 500 m/s (Mach 1.5)
    # Target located 8 km East, 3 km North, moving North at 50 m/s
    m0 = MissileState(
        x=0.0,
        y=0.0,
        heading=math.radians(10.0),
        velocity=500.0,
        max_lateral_accel=25.0 * G_ACCEL,  # 25G fin capability
    )

    t0 = TargetState(
        target_id=1,
        x=8000.0,
        y=3000.0,
        vx=0.0,
        vy=50.0,
        priority=10.0,
    )

    metrics = simulate_single_engagement(m0, t0, nav_constant=4.0)
    print("\n--- TEST RESULTS ---")
    print(f"Intercepted: {metrics['intercepted']}")
    print(f"Flight Time: {metrics['flight_time_s']:.2f} s")
    print(f"Miss Distance: {metrics['final_miss_distance_m']:.3f} m")
    print(f"Peak G-Load: {metrics['peak_g_load']:.1f} G")
