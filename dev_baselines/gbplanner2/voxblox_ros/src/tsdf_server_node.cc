#include <gflags/gflags.h>
#include <glog/logging.h>
#include <rclcpp/rclcpp.hpp>

#include "voxblox_ros/tsdf_server.h"

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();

  rclcpp::init(argc, argv);

  // Use ~/ namespace under the node name to match the ROS 1 nh_private layout.
  auto node = std::make_shared<rclcpp::Node>("voxblox");
  voxblox::TsdfServer server(node.get());

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
