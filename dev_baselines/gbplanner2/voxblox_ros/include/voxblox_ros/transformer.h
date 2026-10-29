#ifndef VOXBLOX_ROS_TRANSFORMER_H_
#define VOXBLOX_ROS_TRANSFORMER_H_

#include <deque>
#include <memory>
#include <string>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <voxblox/core/common.h>

namespace voxblox {

/**
 * Class that binds to either the TF tree or resolves transformations from the
 * ROS parameter server, depending on settings loaded from ROS params.
 *
 * ROS 2 port: original held two ros::NodeHandles (one global, one private).
 * Here we take a single rclcpp::Node* so the caller controls lifetime.
 */
class Transformer {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit Transformer(rclcpp::Node* node);

  bool lookupTransform(const std::string& from_frame,
                       const std::string& to_frame,
                       const rclcpp::Time& timestamp,
                       Transformation* transform);

  void transformCallback(
      const geometry_msgs::msg::TransformStamped::SharedPtr transform_msg);

 private:
  bool lookupTransformTf(const std::string& from_frame,
                         const std::string& to_frame,
                         const rclcpp::Time& timestamp,
                         Transformation* transform);

  bool lookupTransformQueue(const rclcpp::Time& timestamp,
                            Transformation* transform);

  rclcpp::Node* node_;

  std::string world_frame_;
  std::string sensor_frame_;
  bool use_tf_transforms_;
  int64_t timestamp_tolerance_ns_;

  /**
   * B is the body frame of the robot, C is the camera/sensor frame creating
   * the pointclouds, and D is the 'dynamic' frame; i.e. incoming messages
   * are assumed to be T_G_D.
   */
  Transformation T_B_C_;
  Transformation T_B_D_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

  // Only used if use_tf_transforms_ set to false.
  rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr
      transform_sub_;

  // Transform queue, used only when use_tf_transforms is false.
  AlignedDeque<geometry_msgs::msg::TransformStamped> transform_queue_;
};

}  // namespace voxblox

#endif  // VOXBLOX_ROS_TRANSFORMER_H_
