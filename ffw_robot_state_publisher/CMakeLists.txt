################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 3.16)
project(ffw_robot_state_publisher LANGUAGES CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang)")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror=conversion -Werror=unused-but-set-variable
                      -Werror=return-type -Werror=shadow -Werror=format -Werror=range-loop-construct
                      -Werror=missing-braces)
endif()

set(THIS_PACKAGE_INCLUDE_DEPENDS
  builtin_interfaces
  geometry_msgs
  kdl_parser
  rclcpp
  rclcpp_components
  robot_state_publisher
  sensor_msgs
  std_msgs
  urdf
)

################################################################################
# Find and load build settings from external packages
################################################################################
find_package(ament_cmake REQUIRED)
find_package(orocos_kdl REQUIRED)
foreach(Dependency IN ITEMS ${THIS_PACKAGE_INCLUDE_DEPENDS})
  find_package(${Dependency} REQUIRED)
endforeach()

################################################################################
# Build
################################################################################
add_library(${PROJECT_NAME}_node SHARED
  src/ffw_robot_state_publisher.cpp
)
target_compile_features(${PROJECT_NAME}_node PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME}_node PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
target_link_libraries(${PROJECT_NAME}_node PUBLIC
  ${builtin_interfaces_TARGETS}
  ${geometry_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${std_msgs_TARGETS}
  orocos-kdl
  rclcpp::rclcpp
  robot_state_publisher::robot_state_publisher_node
  urdf::urdf
)
target_link_libraries(${PROJECT_NAME}_node PRIVATE
  kdl_parser::kdl_parser
  rclcpp_components::component
)

# Register as component node
rclcpp_components_register_node(${PROJECT_NAME}_node
  PLUGIN "ffw_robot_state_publisher::FFWRobotStatePublisher"
  EXECUTABLE ffw_robot_state_publisher
)

################################################################################
# Install
################################################################################
install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(
  TARGETS ${PROJECT_NAME}_node
  EXPORT export_${PROJECT_NAME}
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

################################################################################
# Macro for ament package
################################################################################
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(${THIS_PACKAGE_INCLUDE_DEPENDS} orocos_kdl)
ament_package()
