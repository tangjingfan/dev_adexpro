#include <gflags/gflags.h>
#include <glog/logging.h>
#include <rclcpp/rclcpp.hpp>

#include "voxblox_ros/esdf_server.h"

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();

  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("voxblox");
  voxblox::EsdfServer server(node.get());

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
