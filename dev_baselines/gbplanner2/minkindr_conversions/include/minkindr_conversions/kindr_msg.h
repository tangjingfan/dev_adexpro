#ifndef MINKINDR_CONVERSIONS_KINDR_MSG_H
#define MINKINDR_CONVERSIONS_KINDR_MSG_H

#include <cmath>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <glog/logging.h>
#include <kindr/minimal/quat-transformation.h>
#include <kindr/minimal/transform-2d.h>
#include <rclcpp/time.hpp>

// ROS 2 port: original lived in namespace `tf` (ROS 1). We use `tf2` here
// to match ROS 2 conventions, since these helpers fundamentally bridge
// minkindr <-> geometry_msgs / tf2 types.
namespace tf2 {

// --- Inline replacements for eigen_conversions (ROS 1 only). -----------------
namespace minkindr_internal {

template <typename Scalar>
inline void quaternionEigenToMsg(const Eigen::Quaternion<Scalar>& q,
                                 geometry_msgs::msg::Quaternion& msg) {
  msg.x = static_cast<double>(q.x());
  msg.y = static_cast<double>(q.y());
  msg.z = static_cast<double>(q.z());
  msg.w = static_cast<double>(q.w());
}

template <typename Scalar>
inline void quaternionMsgToEigen(const geometry_msgs::msg::Quaternion& msg,
                                 Eigen::Quaternion<Scalar>& q) {
  q = Eigen::Quaternion<Scalar>(
      static_cast<Scalar>(msg.w), static_cast<Scalar>(msg.x),
      static_cast<Scalar>(msg.y), static_cast<Scalar>(msg.z));
}

template <typename Scalar>
inline void pointEigenToMsg(const Eigen::Matrix<Scalar, 3, 1>& v,
                            geometry_msgs::msg::Point& msg) {
  msg.x = static_cast<double>(v.x());
  msg.y = static_cast<double>(v.y());
  msg.z = static_cast<double>(v.z());
}

template <typename Scalar>
inline void pointMsgToEigen(const geometry_msgs::msg::Point& msg,
                            Eigen::Matrix<Scalar, 3, 1>& v) {
  v.x() = static_cast<Scalar>(msg.x);
  v.y() = static_cast<Scalar>(msg.y);
  v.z() = static_cast<Scalar>(msg.z);
}

template <typename Scalar>
inline void vectorEigenToMsg(const Eigen::Matrix<Scalar, 3, 1>& v,
                             geometry_msgs::msg::Vector3& msg) {
  msg.x = static_cast<double>(v.x());
  msg.y = static_cast<double>(v.y());
  msg.z = static_cast<double>(v.z());
}

template <typename Scalar>
inline void vectorMsgToEigen(const geometry_msgs::msg::Vector3& msg,
                             Eigen::Matrix<Scalar, 3, 1>& v) {
  v.x() = static_cast<Scalar>(msg.x);
  v.y() = static_cast<Scalar>(msg.y);
  v.z() = static_cast<Scalar>(msg.z);
}

}  // namespace minkindr_internal

// --- Quaternion --------------------------------------------------------------

template <typename Scalar>
void quaternionKindrToMsg(
    const kindr::minimal::RotationQuaternionTemplate<Scalar>& kindr,
    geometry_msgs::msg::Quaternion* msg) {
  CHECK_NOTNULL(msg);
  minkindr_internal::quaternionEigenToMsg(kindr.toImplementation(), *msg);
}

template <typename Scalar>
void quaternionMsgToKindr(
    const geometry_msgs::msg::Quaternion& msg,
    kindr::minimal::RotationQuaternionTemplate<Scalar>* kindr) {
  CHECK_NOTNULL(kindr);
  Eigen::Quaternion<Scalar> quat;
  minkindr_internal::quaternionMsgToEigen(msg, quat);
  *kindr = kindr::minimal::RotationQuaternionTemplate<Scalar>(quat);
}

template <typename Scalar>
void quaternionKindrToMsg(const Eigen::Quaternion<Scalar>& kindr,
                          geometry_msgs::msg::Quaternion* msg) {
  CHECK_NOTNULL(msg);
  minkindr_internal::quaternionEigenToMsg(kindr, *msg);
}

template <typename Scalar>
void quaternionMsgToKindr(const geometry_msgs::msg::Quaternion& msg,
                          Eigen::Quaternion<Scalar>* kindr) {
  CHECK_NOTNULL(kindr);
  Eigen::Quaternion<double> kindr_double;
  minkindr_internal::quaternionMsgToEigen(msg, kindr_double);
  *kindr = kindr_double.cast<Scalar>();
}

template <typename Scalar>
void rotationKindr2DToMsg(const Eigen::Rotation2D<Scalar>& kindr,
                          geometry_msgs::msg::Quaternion* msg) {
  CHECK_NOTNULL(msg);
  const kindr::minimal::RotationQuaternionTemplate<Scalar> quat(
      kindr::minimal::AngleAxisTemplate<Scalar>(
          kindr.angle(), static_cast<Scalar>(0.0), static_cast<Scalar>(0.0),
          static_cast<Scalar>(1.0)));
  minkindr_internal::quaternionEigenToMsg(quat.toImplementation(), *msg);
}

template <typename Scalar>
void quaternionMsgToKindr2D(const geometry_msgs::msg::Quaternion& msg,
                            Eigen::Rotation2D<Scalar>* kindr) {
  CHECK_NOTNULL(kindr);
  CHECK_LT(std::abs(msg.x), std::numeric_limits<double>::epsilon())
      << "No proper 2D rotation.";
  CHECK_LT(std::abs(msg.y), std::numeric_limits<double>::epsilon())
      << "No proper 2D rotation.";
  kindr::minimal::RotationQuaternionTemplate<Scalar> quat;
  minkindr_internal::quaternionMsgToEigen(msg, quat.toImplementation());
  const kindr::minimal::AngleAxisTemplate<Scalar> angle_axis(quat);
  kindr->angle() = std::copysign(angle_axis.angle(), angle_axis.axis()(2));
}

// --- Point / Vector ----------------------------------------------------------

template <typename Scalar>
void pointKindrToMsg(const Eigen::Matrix<Scalar, 3, 1>& kindr,
                     geometry_msgs::msg::Point* msg) {
  CHECK_NOTNULL(msg);
  minkindr_internal::pointEigenToMsg(kindr, *msg);
}

template <typename Scalar>
void pointMsgToKindr(const geometry_msgs::msg::Point& msg,
                     Eigen::Matrix<Scalar, 3, 1>* kindr) {
  CHECK_NOTNULL(kindr);
  minkindr_internal::pointMsgToEigen(msg, *kindr);
}

template <typename Scalar>
void pointKindr2DToMsg(const Eigen::Matrix<Scalar, 2, 1>& kindr,
                       geometry_msgs::msg::Point* msg) {
  CHECK_NOTNULL(msg);
  msg->x = static_cast<double>(kindr.x());
  msg->y = static_cast<double>(kindr.y());
  msg->z = 0.0;
}

template <typename Scalar>
void pointMsgToKindr2D(const geometry_msgs::msg::Point& msg,
                       Eigen::Matrix<Scalar, 2, 1>* kindr) {
  CHECK_NOTNULL(kindr);
  CHECK_LT(std::abs(msg.z), std::numeric_limits<Scalar>::epsilon())
      << "No proper 2D position.";
  kindr->x() = static_cast<Scalar>(msg.x);
  kindr->y() = static_cast<Scalar>(msg.y);
}

template <typename Scalar>
void vectorKindrToMsg(const Eigen::Matrix<Scalar, 3, 1>& kindr,
                      geometry_msgs::msg::Vector3* msg) {
  CHECK_NOTNULL(msg);
  minkindr_internal::vectorEigenToMsg(kindr, *msg);
}

template <typename Scalar>
void vectorMsgToKindr(const geometry_msgs::msg::Vector3& msg,
                      Eigen::Matrix<Scalar, 3, 1>* kindr) {
  CHECK_NOTNULL(kindr);
  Eigen::Matrix<double, 3, 1> kindr_double;
  minkindr_internal::vectorMsgToEigen(msg, kindr_double);
  *kindr = kindr_double.cast<Scalar>();
}

// --- Pose / Transform --------------------------------------------------------

template <typename Scalar>
void poseKindrToMsg(
    const kindr::minimal::QuatTransformationTemplate<Scalar>& kindr,
    geometry_msgs::msg::Pose* msg) {
  CHECK_NOTNULL(msg);
  pointKindrToMsg(kindr.getPosition(), &msg->position);
  quaternionKindrToMsg(kindr.getRotation(), &msg->orientation);
}

template <typename Scalar>
void poseMsgToKindr(
    const geometry_msgs::msg::Pose& msg,
    kindr::minimal::QuatTransformationTemplate<Scalar>* kindr) {
  CHECK_NOTNULL(kindr);
  Eigen::Matrix<Scalar, 3, 1> position;
  Eigen::Quaternion<Scalar> rotation;

  quaternionMsgToKindr(msg.orientation, &rotation);
  pointMsgToKindr(msg.position, &position);

  *kindr =
      kindr::minimal::QuatTransformationTemplate<Scalar>(rotation, position);
}

template <typename Scalar>
void poseStampedKindrToMsg(
    const kindr::minimal::QuatTransformationTemplate<Scalar>& kindr,
    const rclcpp::Time& time, const std::string& reference_frame,
    geometry_msgs::msg::PoseStamped* msg) {
  CHECK_NOTNULL(msg);
  msg->header.frame_id = reference_frame;
  msg->header.stamp = time;
  poseKindrToMsg(kindr, &msg->pose);
}

// Convert a kindr::minimal::QuatTransformation to a geometry_msgs::Transform.
template <typename Scalar>
void transformKindrToMsg(
    const kindr::minimal::QuatTransformationTemplate<Scalar>& kindr,
    geometry_msgs::msg::Transform* msg) {
  CHECK_NOTNULL(msg);
  vectorKindrToMsg(kindr.getPosition(), &msg->translation);
  quaternionKindrToMsg(kindr.getRotation(), &msg->rotation);
}

template <typename Scalar>
void transformMsgToKindr(
    const geometry_msgs::msg::Transform& msg,
    kindr::minimal::QuatTransformationTemplate<Scalar>* kindr) {
  CHECK_NOTNULL(kindr);
  Eigen::Matrix<Scalar, 3, 1> position;
  Eigen::Quaternion<Scalar> rotation;

  quaternionMsgToKindr(msg.rotation, &rotation);
  vectorMsgToKindr(msg.translation, &position);

  *kindr =
      kindr::minimal::QuatTransformationTemplate<Scalar>(rotation, position);
}

// --- 2D pose -----------------------------------------------------------------

template <typename Scalar>
void poseKindr2DToMsg(
    const kindr::minimal::Transformation2DTemplate<Scalar>& kindr,
    geometry_msgs::msg::Pose* msg) {
  CHECK_NOTNULL(msg);
  pointKindr2DToMsg(kindr.getPosition(), &msg->position);
  rotationKindr2DToMsg(kindr.getRotation(), &msg->orientation);
}

template <typename Scalar>
void poseMsgToKindr2D(
    const geometry_msgs::msg::Pose& msg,
    kindr::minimal::Transformation2DTemplate<Scalar>* kindr) {
  CHECK_NOTNULL(kindr);
  Eigen::Matrix<Scalar, 2, 1> position;
  Eigen::Rotation2D<Scalar> rotation;

  quaternionMsgToKindr2D(msg.orientation, &rotation);
  pointMsgToKindr2D(msg.position, &position);

  *kindr = kindr::minimal::Transformation2DTemplate<Scalar>(rotation, position);
}

template <typename Scalar>
void poseStampedKindr2DToMsg(
    const kindr::minimal::Transformation2DTemplate<Scalar>& kindr,
    const rclcpp::Time& time, const std::string& reference_frame,
    geometry_msgs::msg::PoseStamped* msg) {
  CHECK_NOTNULL(msg);
  msg->header.frame_id = reference_frame;
  msg->header.stamp = time;
  poseKindr2DToMsg(kindr, &msg->pose);
}

}  // namespace tf2

#endif  // MINKINDR_CONVERSIONS_KINDR_MSG_H
