cmake_minimum_required(VERSION 3.16)
project(ffw_safety_qp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(controller_manager_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(urdfdom REQUIRED)
find_package(urdfdom_headers REQUIRED)
find_package(PCL REQUIRED COMPONENTS common kdtree)
find_package(osqp_eigen_vendor REQUIRED)
find_package(cyclo_motion_controller_core REQUIRED)

set(LIB_DEPS
  rclcpp
  sensor_msgs
  std_msgs
  std_srvs
  trajectory_msgs
  visualization_msgs
  controller_manager_msgs
  tf2_ros
)

add_library(${PROJECT_NAME}_lib SHARED
  src/capsule_extractor.cpp
  src/capsule_cloud_distance.cpp
  src/robot_kinematics.cpp
  src/shield_qp.cpp
  src/leader_shield_node.cpp
)
target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_lib PUBLIC
  Eigen3::Eigen
  pinocchio::pinocchio
  OsqpEigen::OsqpEigen
  cyclo_motion_controller_core::cyclo_motion_controller_core
  ${urdfdom_LIBRARIES}
  ${PCL_LIBRARIES}
)
ament_target_dependencies(${PROJECT_NAME}_lib PUBLIC ${LIB_DEPS})

add_executable(leader_shield_node src/leader_shield_main.cpp)
target_link_libraries(leader_shield_node PRIVATE ${PROJECT_NAME}_lib)

install(
  TARGETS ${PROJECT_NAME}_lib
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(
  TARGETS leader_shield_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/${PROJECT_NAME} DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_capsule_cloud_distance test/test_capsule_cloud_distance.cpp)
  target_link_libraries(test_capsule_cloud_distance ${PROJECT_NAME}_lib)

  ament_add_gtest(test_capsule_extractor test/test_capsule_extractor.cpp)
  target_link_libraries(test_capsule_extractor ${PROJECT_NAME}_lib)

  ament_add_gtest(test_shield_qp test/test_shield_qp.cpp)
  target_link_libraries(test_shield_qp ${PROJECT_NAME}_lib)
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_lib)
ament_export_dependencies(${LIB_DEPS} Eigen3 pinocchio osqp_eigen_vendor cyclo_motion_controller_core PCL urdfdom)
ament_package()
