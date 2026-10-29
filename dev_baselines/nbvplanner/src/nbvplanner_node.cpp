#include "nbvplanner/nbvplanner.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<NBVPlanner>());
  rclcpp::shutdown();
  return 0;
}
