#include "voxblox_ros/transformer.h"

#include <vector>

#include <minkindr_conversions/kindr_msg.h>
#include <minkindr_conversions/kindr_tf.h>
#include <minkindr_conversions/kindr_xml.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include "voxblox_ros/ros_params.h"

namespace voxblox {

Transformer::Transformer(rclcpp::Node* node)
    : node_(node),
      world_frame_("world"),
      sensor_frame_(""),
      use_tf_transforms_(true),
      timestamp_tolerance_ns_(1000000) {
  world_frame_ = declareOrGet<std::string>(node_, "world_frame", world_frame_);
  sensor_frame_ =
      declareOrGet<std::string>(node_, "sensor_frame", sensor_frame_);

  const double kNanoSecondsInSecond = 1.0e9;
  double timestamp_tolerance_sec =
      timestamp_tolerance_ns_ / kNanoSecondsInSecond;
  timestamp_tolerance_sec = declareOrGet<double>(
      node_, "timestamp_tolerance_sec", timestamp_tolerance_sec);
  timestamp_tolerance_ns_ =
      static_cast<int64_t>(timestamp_tolerance_sec * kNanoSecondsInSecond);

  use_tf_transforms_ =
      declareOrGet<bool>(node_, "use_tf_transforms", use_tf_transforms_);

  if (use_tf_transforms_) {
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(node_->get_clock());
    tf_listener_ =
        std::make_unique<tf2_ros::TransformListener>(*tf_buffer_, node_);
  } else {
    transform_sub_ =
        node_->create_subscription<geometry_msgs::msg::TransformStamped>(
            "transform", 40,
            std::bind(&Transformer::transformCallback, this,
                      std::placeholders::_1));

    // Retrieve T_B_D and T_B_C from params. The ROS 1 version read XmlRpcValue;
    // here we expect a flat row-major 16-element vector. See
    // minkindr_conversions/kindr_xml.h.
    const std::vector<double> kIdentity4x4 = {
        1, 0, 0, 0,  // row 0
        0, 1, 0, 0,  // row 1
        0, 0, 1, 0,  // row 2
        0, 0, 0, 1   // row 3
    };

    std::vector<double> T_B_D_vec =
        declareOrGet<std::vector<double>>(node_, "T_B_D", kIdentity4x4);
    if (!T_B_D_vec.empty()) {
      kindr::minimal::xmlRpcToKindr(T_B_D_vec, &T_B_D_);
      bool invert_T_B_D = declareOrGet<bool>(node_, "invert_T_B_D", false);
      if (invert_T_B_D) {
        T_B_D_ = T_B_D_.inverse();
      }
    }

    std::vector<double> T_B_C_vec =
        declareOrGet<std::vector<double>>(node_, "T_B_C", kIdentity4x4);
    if (!T_B_C_vec.empty()) {
      kindr::minimal::xmlRpcToKindr(T_B_C_vec, &T_B_C_);
      bool invert_T_B_C = declareOrGet<bool>(node_, "invert_T_B_C", false);
      if (invert_T_B_C) {
        T_B_C_ = T_B_C_.inverse();
      }
    }
  }
}

void Transformer::transformCallback(
    const geometry_msgs::msg::TransformStamped::SharedPtr transform_msg) {
  transform_queue_.push_back(*transform_msg);
}

bool Transformer::lookupTransform(const std::string& from_frame,
                                  const std::string& to_frame,
                                  const rclcpp::Time& timestamp,
                                  Transformation* transform) {
  CHECK_NOTNULL(transform);
  if (use_tf_transforms_) {
    return lookupTransformTf(from_frame, to_frame, timestamp, transform);
  } else {
    return lookupTransformQueue(timestamp, transform);
  }
}

bool Transformer::lookupTransformTf(const std::string& from_frame,
                                    const std::string& to_frame,
                                    const rclcpp::Time& timestamp,
                                    Transformation* transform) {
  CHECK_NOTNULL(transform);

  std::string from_frame_modified = from_frame;
  if (!sensor_frame_.empty()) {
    from_frame_modified = sensor_frame_;
  }

  geometry_msgs::msg::TransformStamped tf_msg;
  try {
    tf_msg = tf_buffer_->lookupTransform(to_frame, from_frame_modified,
                                         timestamp);
  } catch (const tf2::TransformException& ex) {
    RCLCPP_ERROR_STREAM(node_->get_logger(),
                        "Error getting TF transform from sensor data: "
                            << ex.what());
    return false;
  }

  tf2::transformMsgToKindr(tf_msg.transform, transform);
  return true;
}

bool Transformer::lookupTransformQueue(const rclcpp::Time& timestamp,
                                       Transformation* transform) {
  CHECK_NOTNULL(transform);
  if (transform_queue_.empty()) {
    RCLCPP_WARN_STREAM_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 30000,
        "No match found for transform timestamp: "
            << timestamp.nanoseconds()
            << " as transform queue is empty.");
    return false;
  }

  // Try to match the transforms in the queue.
  bool match_found = false;
  std::deque<geometry_msgs::msg::TransformStamped>::iterator it =
      transform_queue_.begin();
  for (; it != transform_queue_.end(); ++it) {
    const rclcpp::Time it_stamp(it->header.stamp);
    if (it_stamp > timestamp) {
      if ((it_stamp - timestamp).nanoseconds() < timestamp_tolerance_ns_) {
        match_found = true;
      }
      break;
    }
    if ((timestamp - it_stamp).nanoseconds() < timestamp_tolerance_ns_) {
      match_found = true;
      break;
    }
  }

  Transformation T_G_D;
  if (match_found) {
    tf2::transformMsgToKindr(it->transform, &T_G_D);
  } else {
    if (it == transform_queue_.begin() || it == transform_queue_.end()) {
      RCLCPP_WARN_STREAM_THROTTLE(
          node_->get_logger(), *node_->get_clock(), 30000,
          "No match found for transform timestamp: "
              << timestamp.nanoseconds() << " Queue front: "
              << rclcpp::Time(transform_queue_.front().header.stamp)
                     .nanoseconds()
              << " back: "
              << rclcpp::Time(transform_queue_.back().header.stamp)
                     .nanoseconds());
      return false;
    }
    Transformation T_G_D_newest;
    tf2::transformMsgToKindr(it->transform, &T_G_D_newest);
    int64_t offset_newest_ns =
        (rclcpp::Time(it->header.stamp) - timestamp).nanoseconds();
    --it;
    Transformation T_G_D_oldest;
    tf2::transformMsgToKindr(it->transform, &T_G_D_oldest);
    int64_t offset_oldest_ns =
        (timestamp - rclcpp::Time(it->header.stamp)).nanoseconds();

    FloatingPoint t_diff_ratio =
        static_cast<FloatingPoint>(offset_oldest_ns) /
        static_cast<FloatingPoint>(offset_newest_ns + offset_oldest_ns);

    Transformation::Vector6 diff_vector =
        (T_G_D_oldest.inverse() * T_G_D_newest).log();
    T_G_D = T_G_D_oldest * Transformation::exp(t_diff_ratio * diff_vector);
  }

  *transform = T_G_D * T_B_D_.inverse() * T_B_C_;

  // Clear queue up to this point but leave the current message in place.
  transform_queue_.erase(transform_queue_.begin(), it);
  return true;
}

}  // namespace voxblox
