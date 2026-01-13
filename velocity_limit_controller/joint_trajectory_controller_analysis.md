# Joint Trajectory Controller - Structure & Topic Subscription Analysis

## Overview
This document analyzes how `joint_trajectory_controller` is structured and how it subscribes to trajectory topics, to serve as a reference for implementing `VelocityLimitController`.

## Key Patterns

### 1. Class Structure
- **Inherits from**: `controller_interface::ControllerInterface`
- **Namespace**: `joint_trajectory_controller`
- **Location**: 
  - Header: `include/joint_trajectory_controller/joint_trajectory_controller.hpp`
  - Source: `src/joint_trajectory_controller.cpp`

### 2. Topic Subscription Pattern

#### Subscription Creation (in `on_configure()`)
```cpp
// Line 933-936 in joint_trajectory_controller.cpp
joint_command_subscriber_ =
  get_node()->create_subscription<trajectory_msgs::msg::JointTrajectory>(
    "~/joint_trajectory",  // Topic name (expands to controller_name/joint_trajectory)
    rclcpp::SystemDefaultsQoS(),
    std::bind(&JointTrajectoryController::topic_callback, this, std::placeholders::_1));
```

**Key points:**
- Topic name uses `~/` prefix, which expands to `<controller_name>/joint_trajectory`
- Uses `rclcpp::SystemDefaultsQoS()` for quality of service
- Callback is bound using `std::bind`

#### Realtime-Safe Message Passing
```cpp
// Header (line 159-160)
realtime_tools::RealtimeBuffer<std::shared_ptr<trajectory_msgs::msg::JointTrajectory>>
  new_trajectory_msg_;
```

**Why RealtimeBuffer?**
- Subscriber callback runs in **non-RT thread** (ROS 2 callback thread)
- `update()` runs in **RT thread** (controller update loop)
- `RealtimeBuffer` provides lock-free, thread-safe communication between them

#### Callback Implementation
```cpp
// Line 1250-1264
void JointTrajectoryController::topic_callback(
  const std::shared_ptr<trajectory_msgs::msg::JointTrajectory> msg)
{
  if (!validate_trajectory_msg(*msg))
  {
    return;
  }
  // Always replace old msg with new one
  if (subscriber_is_active_)
  {
    add_new_trajectory_msg(msg);
    rt_is_holding_ = false;
  }
}
```

**Key points:**
- Validates message before processing
- Checks `subscriber_is_active_` flag (set in `on_activate()`, cleared in `on_deactivate()`)
- Writes to RealtimeBuffer using `add_new_trajectory_msg()`

#### Writing to RealtimeBuffer
```cpp
// Line 1672-1676
void JointTrajectoryController::add_new_trajectory_msg(
  const std::shared_ptr<trajectory_msgs::msg::JointTrajectory> & traj_msg)
{
  new_trajectory_msg_.writeFromNonRT(traj_msg);
}
```

**Note:** Uses `writeFromNonRT()` because callback is in non-RT thread.

#### Reading from RealtimeBuffer (in `update()`)
```cpp
// Line 157
auto new_external_msg = new_trajectory_msg_.readFromRT();
```

**Note:** Uses `readFromRT()` because `update()` is in RT thread.

### 3. Interface Configuration

#### Command Interfaces
```cpp
// Line 103-117
controller_interface::InterfaceConfiguration
JointTrajectoryController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration conf;
  conf.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  conf.names.reserve(num_cmd_joints_ * params_.command_interfaces.size());
  for (const auto & joint_name : command_joint_names_)
  {
    for (const auto & interface_type : params_.command_interfaces)
    {
      conf.names.push_back(joint_name + "/" + interface_type);
    }
  }
  return conf;
}
```

**Pattern:**
- Returns `INDIVIDUAL` type
- Format: `<joint_name>/<interface_type>` (e.g., `joint1/position`, `joint1/velocity`)

#### State Interfaces
```cpp
// Line 119-133
controller_interface::InterfaceConfiguration
JointTrajectoryController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration conf;
  conf.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  conf.names.reserve(dof_ * params_.state_interfaces.size());
  for (const auto & joint_name : params_.joints)
  {
    for (const auto & interface_type : params_.state_interfaces)
    {
      conf.names.push_back(joint_name + "/" + interface_type);
    }
  }
  return conf;
}
```

### 4. Lifecycle Methods

#### `on_init()`
- Initialize parameter listener
- Parse URDF (optional)
- **No subscriber creation here**

#### `on_configure()`
- Create subscriber (line 933-936)
- Create publisher
- Create action server
- **This is where ROS 2 communication is set up**

#### `on_activate()`
- Claim command/state interfaces
- Initialize trajectory
- Set `subscriber_is_active_ = true` (line 1063)
- **Controller becomes active and ready to process messages**

#### `on_deactivate()`
- Release interfaces
- Set `subscriber_is_active_ = false` (line 1182)
- **Controller stops processing new messages**

#### `update()`
- Called at controller update rate (e.g., 100 Hz)
- Reads from RealtimeBuffer: `new_trajectory_msg_.readFromRT()`
- Processes trajectory
- Writes to command interfaces

### 5. Plugin Registration

#### Plugin XML (`joint_trajectory_plugin.xml`)
```xml
<library path="joint_trajectory_controller">
  <class name="joint_trajectory_controller/JointTrajectoryController" 
         type="joint_trajectory_controller::JointTrajectoryController" 
         base_class_type="controller_interface::ControllerInterface">
    <description>
      The joint trajectory controller executes joint-space trajectories on a set of joints
    </description>
  </class>
</library>
```

#### Plugin Export (end of .cpp file)
```cpp
// Line 1821-1824
#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(
  joint_trajectory_controller::JointTrajectoryController, 
  controller_interface::ControllerInterface)
```

### 6. CMakeLists.txt Pattern

```cmake
# Create shared library
add_library(joint_trajectory_controller SHARED
  src/joint_trajectory_controller.cpp
  src/trajectory.cpp
)

# Link dependencies
target_link_libraries(joint_trajectory_controller PUBLIC
  controller_interface::controller_interface
  rclcpp::rclcpp
  realtime_tools::realtime_tools
  # ... other deps
)

# Export plugin description
pluginlib_export_plugin_description_file(controller_interface joint_trajectory_plugin.xml)
```

## Key Takeaways for VelocityLimitController

1. **Subscription**: Create in `on_configure()`, use `~/trajectory_topic` pattern
2. **Realtime Safety**: Use `RealtimeBuffer` to pass messages from callback to `update()`
3. **Lifecycle**: Use `subscriber_is_active_` flag to gate message processing
4. **Interfaces**: Return `INDIVIDUAL` type with format `<joint>/<interface>`
5. **Plugin**: Export class using `PLUGINLIB_EXPORT_CLASS` macro
6. **Validation**: Always validate incoming messages before processing

## Differences for VelocityLimitController

Since `VelocityLimitController` should:
- Subscribe to the **same topic** as JTC (not `~/joint_trajectory`)
- Use only `points[0].positions` (first point, instantaneous target)
- Write only `goal_velocity` command interface

**Topic subscription should be:**
```cpp
// In on_configure(), use parameter for topic name
std::string topic_name = params_.trajectory_topic;  // e.g., "/joint_trajectory_controller/joint_trajectory"
trajectory_subscriber_ = get_node()->create_subscription<trajectory_msgs::msg::JointTrajectory>(
  topic_name,
  rclcpp::SystemDefaultsQoS(),
  std::bind(&VelocityLimitController::trajectory_callback, this, std::placeholders::_1));
```

**Simpler than JTC:**
- No action server needed
- No trajectory interpolation (just use `points[0]`)
- No state publisher needed (unless you want debugging)
- Simpler validation (just check `points[0].positions` exists)
