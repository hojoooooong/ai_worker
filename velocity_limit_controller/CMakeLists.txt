cmake_minimum_required(VERSION 3.16)
project(velocity_limit_controller)

find_package(ros2_control_cmake REQUIRED)
set_compiler_options()
export_windows_symbols()

set(THIS_PACKAGE_INCLUDE_DEPENDS
  controller_interface
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  realtime_tools
  trajectory_msgs
)

find_package(ament_cmake REQUIRED)
foreach(Dependency IN ITEMS ${THIS_PACKAGE_INCLUDE_DEPENDS})
  find_package(${Dependency} REQUIRED)
endforeach()

add_library(velocity_limit_controller SHARED
  src/velocity_limit_controller.cpp
)

target_compile_features(velocity_limit_controller PUBLIC cxx_std_17)

if(WIN32)
  target_compile_definitions(velocity_limit_controller PRIVATE _USE_MATH_DEFINES)
endif()

target_include_directories(velocity_limit_controller PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/velocity_limit_controller>
)

target_link_libraries(velocity_limit_controller PUBLIC
  controller_interface::controller_interface
  hardware_interface::hardware_interface
  pluginlib::pluginlib
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  realtime_tools::realtime_tools
  ${trajectory_msgs_TARGETS}
)

pluginlib_export_plugin_description_file(controller_interface velocity_limit_plugin.xml)

install(
  DIRECTORY include/
  DESTINATION include/velocity_limit_controller
)

install(TARGETS
    velocity_limit_controller
  EXPORT export_velocity_limit_controller
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

ament_export_targets(export_velocity_limit_controller HAS_LIBRARY_TARGET)
ament_export_dependencies(${THIS_PACKAGE_INCLUDE_DEPENDS})
ament_package()
