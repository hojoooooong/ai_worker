## Cursor Prompt (English)

> **Context**
>
> I am working on a ROS 2 (Humble) system using `ros2_control`.
> I already use `joint_trajectory_controller` (JTC) to command **position** only, with `time_from_start = 0` (bypass style).
>
> My hardware is Dynamixel Y, where:
>
> * `Goal Position` is the position command
> * `Goal Velocity` acts as a **velocity limit** for the internal cascade controller
> * `Goal Velocity = 0` means hard hold (no motion)
> * I want to keep the JTC interface unchanged
>
> I want to implement a **separate ros2_control controller** that:
>
> * Subscribes to the same `trajectory_msgs/JointTrajectory` topic as JTC
> * Extracts the instantaneous target position (`points[0].positions`)
> * Computes a **smooth, braking-distance-based velocity limit**
> * Writes only `goal_velocity` to the hardware interface
> * Runs at ~100 Hz and is robust to jitter (non-RT PC)

---

> **Task**
>
> Create a new ros2_control controller plugin called
> **`VelocityLimitController`** using `controller_interface::ControllerInterface`.
>
> The controller must:
>
> ### Interfaces
>
> * **Command interfaces (claim only):**
>
>   * `<joint_name>/goal_velocity`
> * **State interfaces (read):**
>
>   * `<joint_name>/position`
>
> ### Input
>
> * Subscribe to a `trajectory_msgs/msg/JointTrajectory` topic
> * Use only `points[0].positions` as the instantaneous target
> * Support joint name remapping
>
> ### Core Control Logic (per joint)
>
> ```
> e = q_target - q_current
>
> v_stop = sqrt(2 * a_brake * abs(e))
> v_target = clamp(v_stop, v_min, v_max)
>
> // Optional smoothing
> v_target = low_pass_filter(v_target)
>
> // Rate limit (important)
> v_out += clamp(v_target - v_out,
>                -ramp_down * dt,
>                +ramp_up   * dt)
>
> // Safety
> if trajectory input times out:
>     v_out -> ramp down to 0
>
> command goal_velocity = v_out
> ```
>
> Notes:
>
> * `goal_velocity` is a **limit**, not a velocity command
> * No acceleration or torque commands are used
> * No internal trajectory generation
> * Locking (goal_velocity = 0 near target) must be **optional and parameterized**
>
> ### Parameters (YAML)
>
> * `joints[]`
> * `trajectory_topic`
> * `a_brake[]`
> * `v_max[]`
> * `v_min[]`
> * `ramp_up[]`
> * `ramp_down[]`
> * `e_dead_in[]`, `e_dead_out[]` (optional lock hysteresis)
> * `use_lock` (bool)
> * `lpf_cutoff_hz`
> * `timeout_ms`
>
> ### Behavior Goals
>
> * Smooth start (no jerk at motion start)
> * Smooth braking when target stops abruptly
> * Stable behavior with 100 Hz update and moderate jitter
> * Compatible with strong velocity PI loop (Dynamixel Y)
>
> ### Output
>
> Please generate:
>
> 1. Controller class header + source (`.hpp` / `.cpp`)
> 2. `plugin.xml`
> 3. Example `controller.yaml`
> 4. Short explanation of how the controller works
>
> The code should be clean, minimal, and suitable for real hardware.
>
> Assume the hardware interface already supports the `goal_velocity` command interface.
>
> Do **not** modify the joint trajectory controller.

---

## Why this prompt works well in Cursor

* Very clear separation of **what already exists** vs **what to build**
* Explicit control law (Cursor won’t hallucinate the logic)
* Mentions **Dynamixel Y behavior** (velocity limit, hard hold at 0)
* Tells Cursor **not to touch HW interface or JTC**
* Asks for **plugin.xml + yaml**, which Cursor often forgets unless asked

---

If you want, next I can:

* Reduce this to a **shorter “follow-up refinement prompt”**
* Or write a **second prompt** to tune parameters after first test
* Or give you a **debug checklist** (“if it jerks → change this param”)

Just tell me how you plan to iterate.
