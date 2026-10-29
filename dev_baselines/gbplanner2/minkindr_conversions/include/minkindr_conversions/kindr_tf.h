#ifndef MINKINDR_CONVERSIONS_KINDR_TF_H
#define MINKINDR_CONVERSIONS_KINDR_TF_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <kindr/minimal/quat-transformation.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>

// ROS 2 port: original lived in namespace `tf` (ROS 1). We use `tf2` here
// since tf2 is the ROS 2 successor and tf2 types (tf2::Transform / Quaternion
// / Vector3) live in that namespace.
namespace tf2 {

// --- Inline replacements for tf_conversions/tf_eigen.h (ROS 1 only). ---------
namespace minkindr_internal {

inline void quaternionTF2ToEigen(const tf2::Quaternion& q,
                                 Eigen::Quaterniond& e) {
  e = Eigen::Quaterniond(q.w(), q.x(), q.y(), q.z());
}

inline void quaternionEigenToTF2(const Eigen::Quaterniond& e,
                                 tf2::Quaternion& q) {
  q.setValue(e.x(), e.y(), e.z(), e.w());
}

inline void vectorTF2ToEigen(const tf2::Vector3& t, Eigen::Vector3d& e) {
  e = Eigen::Vector3d(t.x(), t.y(), t.z());
}

inline void vectorEigenToTF2(const Eigen::Vector3d& e, tf2::Vector3& t) {
  t = tf2::Vector3(e.x(), e.y(), e.z());
}

}  // namespace minkindr_internal

// --- Quaternion --------------------------------------------------------------

template <typename Scalar>
void quaternionKindrToTF(
    const kindr::minimal::RotationQuaternionTemplate<Scalar>& kindr,
    tf2::Quaternion* tf_type) {
  CHECK_NOTNULL(tf_type);
  Eigen::Quaterniond q = kindr.toImplementation().template cast<double>();
  minkindr_internal::quaternionEigenToTF2(q, *tf_type);
}

template <typename Scalar>
void quaternionTFToKindr(
    const tf2::Quaternion& tf_type,
    kindr::minimal::RotationQuaternionTemplate<Scalar>* kindr) {
  CHECK_NOTNULL(kindr);
  Eigen::Quaterniond q_double;
  minkindr_internal::quaternionTF2ToEigen(tf_type, q_double);
  *kindr =
      kindr::minimal::RotationQuaternionTemplate<Scalar>(q_double.cast<Scalar>());
}

template <typename Scalar>
void quaternionKindrToTF(const Eigen::Quaternion<Scalar>& kindr,
                         tf2::Quaternion* tf_type) {
  CHECK_NOTNULL(tf_type);
  Eigen::Quaterniond q = kindr.template cast<double>();
  minkindr_internal::quaternionEigenToTF2(q, *tf_type);
}

template <typename Scalar>
void quaternionTFToKindr(const tf2::Quaternion& tf_type,
                         Eigen::Quaternion<Scalar>* kindr) {
  CHECK_NOTNULL(kindr);
  Eigen::Quaterniond kindr_double;
  minkindr_internal::quaternionTF2ToEigen(tf_type, kindr_double);
  *kindr = kindr_double.cast<Scalar>();
}

// --- Vector ------------------------------------------------------------------

template <typename Scalar>
void vectorKindrToTF(const Eigen::Matrix<Scalar, 3, 1>& kindr,
                     tf2::Vector3* tf_type) {
  CHECK_NOTNULL(tf_type);
  Eigen::Vector3d v = kindr.template cast<double>();
  minkindr_internal::vectorEigenToTF2(v, *tf_type);
}

template <typename Scalar>
void vectorTFToKindr(const tf2::Vector3& tf_type,
                     Eigen::Matrix<Scalar, 3, 1>* kindr) {
  CHECK_NOTNULL(kindr);
  Eigen::Vector3d kindr_double;
  minkindr_internal::vectorTF2ToEigen(tf_type, kindr_double);
  *kindr = kindr_double.cast<Scalar>();
}

// --- Transform / Pose --------------------------------------------------------

// Convert a kindr::minimal::QuatTransformation to a tf2::Transform.
template <typename Scalar>
void transformKindrToTF(
    const kindr::minimal::QuatTransformationTemplate<Scalar>& kindr,
    tf2::Transform* tf_type) {
  CHECK_NOTNULL(tf_type);
  tf2::Vector3 origin;
  tf2::Quaternion rotation;
  vectorKindrToTF(kindr.getPosition(), &origin);
  quaternionKindrToTF(kindr.getRotation(), &rotation);
  tf_type->setOrigin(origin);
  tf_type->setRotation(rotation);
}

template <typename Scalar>
void transformTFToKindr(
    const tf2::Transform& tf_type,
    kindr::minimal::QuatTransformationTemplate<Scalar>* kindr) {
  CHECK_NOTNULL(kindr);
  Eigen::Matrix<Scalar, 3, 1> position;
  Eigen::Quaternion<Scalar> rotation;

  quaternionTFToKindr(tf_type.getRotation(), &rotation);
  vectorTFToKindr(tf_type.getOrigin(), &position);

  // Enforce positive w.
  if (rotation.w() < 0) {
    rotation.coeffs() = -rotation.coeffs();
  }

  *kindr =
      kindr::minimal::QuatTransformationTemplate<Scalar>(rotation, position);
}

// tf2 has no separate `tf::Pose` type; it is aliased to tf2::Transform.
// These pose overloads forward to the transform versions to preserve the
// original API surface.
template <typename Scalar>
void poseKindrToTF(
    const kindr::minimal::QuatTransformationTemplate<Scalar>& kindr,
    tf2::Transform* tf_type) {
  transformKindrToTF(kindr, tf_type);
}

template <typename Scalar>
void poseTFToKindr(const tf2::Transform& tf_type,
                   kindr::minimal::QuatTransformationTemplate<Scalar>* kindr) {
  transformTFToKindr(tf_type, kindr);
}

}  // namespace tf2

#endif  // MINKINDR_CONVERSIONS_KINDR_TF_H
