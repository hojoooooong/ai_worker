cmake_minimum_required(VERSION 3.8)
project(ffw_kinematics)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(urdf REQUIRED)
find_package(orocos_kdl REQUIRED)

# C++ executables
add_executable(kdl_ik_solver_cpp src/kdl_ik_solver.cpp)

target_include_directories(kdl_ik_solver_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(kdl_ik_solver_cpp PUBLIC c_std_99 cxx_std_17)

ament_target_dependencies(kdl_ik_solver_cpp
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  trajectory_msgs
  kdl_parser
  urdf
  orocos_kdl
)

# Realtime IK Solver executable
add_executable(realtime_ik_solver src/realtime_ik_solver.cpp)

target_include_directories(realtime_ik_solver PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(realtime_ik_solver PUBLIC c_std_99 cxx_std_17)

ament_target_dependencies(realtime_ik_solver
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  trajectory_msgs
  kdl_parser
  urdf
  orocos_kdl
)

# Target Pose Publisher executable
add_executable(target_pose_publisher src/target_pose_publisher.cpp)

target_include_directories(target_pose_publisher PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(target_pose_publisher PUBLIC c_std_99 cxx_std_17)

ament_target_dependencies(target_pose_publisher
  rclcpp
  geometry_msgs
)

# Install C++ executables
install(TARGETS kdl_ik_solver_cpp realtime_ik_solver target_pose_publisher
  DESTINATION lib/${PROJECT_NAME})


# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
