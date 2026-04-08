cmake_minimum_required(VERSION 3.5)
project(v_marker_estimation)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(example_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(nav_msgs REQUIRED)


rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LaserSample.msg"
  "msg/LaserSampleSet.msg"
  "msg/LaserSampleSetList.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

install(DIRECTORY launch param msg
  DESTINATION share/${PROJECT_NAME}
)

include_directories(include)

add_executable(lidar_cluster
  src/cluster_node.cpp
  src/laser_processor.cpp
)

add_executable(v_marker_localization
  src/v_marker_localization_node.cpp
)

add_executable(holonomic_drive_controller
  src/holonomic_drive_controller.cpp
)

add_executable(marker_ab_goal_router
  src/marker_ab_goal_router.cpp
)

ament_target_dependencies(lidar_cluster
  ament_index_cpp
  rclcpp
  std_msgs
  geometry_msgs
  tf2
  tf2_ros
  visualization_msgs
  std_srvs
  sensor_msgs
  tf2_geometry_msgs
  nav_msgs
)

ament_target_dependencies(v_marker_localization
  ament_index_cpp
  rclcpp
  std_msgs
  geometry_msgs
  tf2
  tf2_ros
  visualization_msgs
  std_srvs
  sensor_msgs
  tf2_geometry_msgs
  nav_msgs
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

target_link_libraries(lidar_cluster "${cpp_typesupport_target}")
target_link_libraries(v_marker_localization "${cpp_typesupport_target}")
target_link_libraries(holonomic_drive_controller "${cpp_typesupport_target}")
target_link_libraries(marker_ab_goal_router "${cpp_typesupport_target}")

ament_target_dependencies(holonomic_drive_controller
  ament_index_cpp
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
  tf2
  tf2_ros
  std_srvs
  nav_msgs
)

ament_target_dependencies(marker_ab_goal_router
  ament_index_cpp
  rclcpp
  std_msgs
  geometry_msgs
  std_srvs
  example_interfaces
  nav_msgs
  tf2
  tf2_ros
)

install(TARGETS
  lidar_cluster
  v_marker_localization
  holonomic_drive_controller
  marker_ab_goal_router
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()

