# Velocity Limit Controller

A ROS 2 controller plugin that computes velocity limits based on braking distance from trajectory target positions. Designed for Dynamixel Y servos where `goal_velocity` acts as a velocity limit for the internal cascade controller.

## Overview

This controller:
- Subscribes to the same `trajectory_msgs/JointTrajectory` topic as `joint_trajectory_controller`
- Extracts instantaneous target positions from `points[0].positions`
- Computes smooth, braking-distance-based velocity limits
- Writes only `goal_velocity` to the hardware interface
- Runs at the controller update rate (typically 100 Hz) and is robust to jitter

## How It Works

### Control Law

For each joint, the controller:

1. **Computes position error**: `e = q_target - q_current`

2. **Calculates braking-distance-based velocity limit**:
   ```
   v_stop = sqrt(2 * a_brake * |e|)
   // Apply minimum velocity threshold to overcome friction
   if (v_stop > 0 && v_stop < v_min):
       v_stop = v_min
   v_target = clamp(v_stop, 0.0, v_max)
   ```

3. **Applies low-pass filter** (optional smoothing):
   ```
   v_target = lpf(v_target)
   ```

4. **Applies rate limiting** (prevents sudden changes):
   ```
   v_out += clamp(v_target - v_out, -ramp_down * dt, +ramp_up * dt)
   ```

5. **Optional locking near target** (if `use_lock=true`):
   - If `|e| < e_dead_in`: set `v_out = 0` (lock)
   - If `|e| > e_dead_out`: unlock
   - Hysteresis prevents chattering

6. **Safety timeout**:
   - If no trajectory message received for `timeout_ms`, velocity ramps down to 0

### Key Features

- **Smooth start**: Rate limiting prevents jerk at motion start
- **Smooth braking**: Velocity automatically reduces as target is approached
- **Robust to jitter**: Low-pass filter and rate limiting handle non-RT PC timing variations
- **Compatible with Dynamixel Y**: 
  - `goal_velocity` is always written as **absolute value (positive)**
  - `goal_velocity = 0` means hard hold (no motion)
  - Direction is determined by the position error, but the velocity limit itself is always positive

## Interfaces

### Command Interfaces (claimed)
- `<joint_name>/velocity` - The `goal_velocity` command interface (always written as absolute value, positive)

### State Interfaces (read)
- `<joint_name>/position` - Current joint position

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `joints` | `string[]` | List of joint names |
| `trajectory_topic` | `string` | Topic to subscribe to (e.g., `/joint_trajectory_controller/joint_trajectory`) |
| `a_brake` | `double[]` | Braking acceleration per joint (rad/s²) |
| `v_max` | `double[]` | Maximum velocity limit per joint (rad/s) |
| `v_min` | `double[]` | Minimum velocity threshold per joint (rad/s). When computed velocity is > 0 but < v_min, use v_min to overcome friction/deadband. Set to 0.0 to disable. |
| `ramp_up` | `double[]` | Velocity ramp-up rate per joint (rad/s²) |
| `ramp_down` | `double[]` | Velocity ramp-down rate per joint (rad/s²) |
| `e_dead_in` | `double[]` | Inner deadband threshold for locking (rad, optional) |
| `e_dead_out` | `double[]` | Outer deadband threshold for locking (rad, optional) |
| `use_lock` | `bool` | Enable locking near target (default: false) |
| `lpf_cutoff_hz` | `double` | Low-pass filter cutoff frequency (Hz, default: 10.0) |
| `timeout_ms` | `double` | Trajectory timeout in milliseconds (default: 100.0) |

## Usage

### 1. Build the Controller

```bash
cd ~/ros2_ws
colcon build --packages-select velocity_limit_controller
source install/setup.bash
```

### 2. Configure in Controller Manager

Add to your `controllers.yaml`:

```yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    velocity_limit_controller:
      type: velocity_limit_controller/VelocityLimitController
      joints:
        - joint1
        - joint2
        - joint3
      trajectory_topic: "/joint_trajectory_controller/joint_trajectory"
      a_brake: [2.0, 2.0, 2.0]
      v_max: [3.0, 3.0, 3.0]
      v_min: [-3.0, -3.0, -3.0]
      ramp_up: [5.0, 5.0, 5.0]
      ramp_down: [10.0, 10.0, 10.0]
      use_lock: false
      lpf_cutoff_hz: 10.0
      timeout_ms: 100.0
```

### 3. Load and Start

```bash
ros2 control load_controller velocity_limit_controller
ros2 control set_controller_state velocity_limit_controller active
```

### 4. Verify

The controller will automatically subscribe to the trajectory topic and start computing velocity limits. Monitor the command interfaces to see the computed `goal_velocity` values.

## Tuning Guide

### Smooth Motion
- **Increase `lpf_cutoff_hz`**: Faster response, less smoothing
- **Decrease `lpf_cutoff_hz`**: Slower response, more smoothing
- **Increase `ramp_up`**: Faster acceleration
- **Increase `ramp_down`**: Faster deceleration

### Small Movements / Friction Compensation
- **Set `v_min`**: Minimum velocity threshold to overcome friction/deadband
  - If motor doesn't move for small position errors, increase `v_min` (e.g., 0.1-0.5 rad/s)
  - If motor overshoots or oscillates near target, decrease `v_min` or set to 0.0
  - Typical values: 0.1-0.3 rad/s for small motors, 0.2-0.5 rad/s for larger motors

### Braking Behavior
- **Increase `a_brake`**: More aggressive braking (velocity reduces faster near target)
- **Decrease `a_brake`**: Gentler braking (velocity reduces slower near target)

### Locking Near Target
- **Enable `use_lock`**: Sets velocity to 0 when very close to target
- **Tune `e_dead_in` and `e_dead_out`**: Adjust deadband thresholds
  - Smaller values: Lock closer to target
  - Larger values: Lock further from target

### Safety
- **Adjust `timeout_ms`**: How long to wait before ramping down if no trajectory received
- **Set `v_max`/`v_min`**: Hard limits on velocity

## Example Configuration

See `config/velocity_limit_controller_example.yaml` for a complete example configuration.

## Notes

- This controller is designed to work **alongside** `joint_trajectory_controller`, not replace it
- The `joint_trajectory_controller` still commands positions
- This controller only sets velocity limits via `goal_velocity`
- Both controllers can run simultaneously on the same robot

## Architecture

The controller follows the standard ROS 2 control pattern:
- Inherits from `controller_interface::ControllerInterface`
- Uses `realtime_tools::RealtimeBuffer` for thread-safe message passing
- Subscribes to trajectory topic in non-RT thread
- Processes in RT `update()` function
- Claims command/state interfaces through controller manager
