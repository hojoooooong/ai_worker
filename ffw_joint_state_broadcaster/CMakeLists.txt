################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 3.16)
project(ffw_joint_state_broadcaster LANGUAGES CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang)")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror=conversion -Werror=unused-but-set-variable
                      -Werror=return-type -Werror=shadow -Werror=format -Werror=range-loop-construct
                      -Werror=missing-braces)
endif()

set(THIS_PACKAGE_INCLUDE_DEPENDS
  builtin_interfaces
  controller_interface
  generate_parameter_library
  pluginlib
  rclcpp_lifecycle
  realtime_tools
  sensor_msgs
)

################################################################################
# Find and load build settings from external packages
################################################################################
find_package(ament_cmake REQUIRED)
find_package(backward_ros REQUIRED)
foreach(Dependency IN ITEMS ${THIS_PACKAGE_INCLUDE_DEPENDS})
  find_package(${Dependency} REQUIRED)
endforeach()

generate_parameter_library(ffw_joint_state_broadcaster_parameters
  src/ffw_joint_state_broadcaster_parameters.yaml
)

add_library(ffw_joint_state_broadcaster SHARED
  src/ffw_joint_state_broadcaster.cpp
)
target_compile_features(ffw_joint_state_broadcaster PUBLIC cxx_std_17)
target_include_directories(ffw_joint_state_broadcaster PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/ffw_joint_state_broadcaster>
)
target_link_libraries(ffw_joint_state_broadcaster PUBLIC
  ${builtin_interfaces_TARGETS}
  ${sensor_msgs_TARGETS}
  controller_interface::controller_interface
  ffw_joint_state_broadcaster_parameters
  pluginlib::pluginlib
  rclcpp_lifecycle::rclcpp_lifecycle
  realtime_tools::realtime_tools
)

# Causes the visibility macros to use dllexport rather than dllimport,
# which is appropriate when building the dll but not consuming it.
target_compile_definitions(ffw_joint_state_broadcaster PRIVATE "FFW_JOINT_STATE_BROADCASTER_BUILDING_DLL")
pluginlib_export_plugin_description_file(controller_interface ffw_joint_state_broadcaster_plugin.xml)

################################################################################
# Install
################################################################################
install(
  DIRECTORY include/
  DESTINATION include/ffw_joint_state_broadcaster
)

install(
  TARGETS
    ffw_joint_state_broadcaster
    ffw_joint_state_broadcaster_parameters
  EXPORT export_ffw_joint_state_broadcaster
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

################################################################################
# Macro for ament package
################################################################################
ament_export_targets(export_ffw_joint_state_broadcaster HAS_LIBRARY_TARGET)
ament_export_dependencies(${THIS_PACKAGE_INCLUDE_DEPENDS})
ament_package()
