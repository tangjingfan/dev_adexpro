#ifndef MINKINDR_CONVERSIONS_KINDR_XML_H
#define MINKINDR_CONVERSIONS_KINDR_XML_H

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <kindr/minimal/quat-transformation.h>

namespace kindr {
namespace minimal {

// ROS 2 port note:
// The ROS 1 version of this header consumed `XmlRpc::XmlRpcValue` (a 4x4
// nested array loaded from the ROS 1 param server). ROS 2 does not expose
// XmlRpc; the natural replacement is to declare a `std::vector<double>`
// parameter (16 elements, row-major) and pass it here.
//
// Callers (e.g. voxblox_ros::Transformer) should do:
//   std::vector<double> T_B_C =
//       node->declare_parameter<std::vector<double>>(
//           "T_B_C", std::vector<double>{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1});
//   kindr::minimal::xmlRpcToKindr(T_B_C, &T_B_C_);
template <typename Scalar>
void xmlRpcToKindr(const std::vector<double>& mat,
                   kindr::minimal::QuatTransformationTemplate<Scalar>* kindr) {
  if (kindr == nullptr) {
    LOG(ERROR) << "Null pointer given";
    return;
  }
  if (mat.size() != 16) {
    LOG(ERROR) << "Transform vector has " << mat.size()
               << " elements, expected 16 (row-major 4x4).";
    return;
  }

  typename kindr::minimal::QuatTransformationTemplate<Scalar>::RotationMatrix
      temp_rot_matrix;
  typename kindr::minimal::QuatTransformationTemplate<Scalar>::Position
      temp_translation;

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      temp_rot_matrix(i, j) = mat[i * 4 + j];
    }
    temp_translation(i) = mat[i * 4 + 3];
  }

  // Renormalize rotation to correct for rounding error when yaml was written.
  kindr::minimal::RotationQuaternionTemplate<Scalar> temp_rot_quat =
      kindr::minimal::RotationQuaternionTemplate<
          Scalar>::constructAndRenormalize(temp_rot_matrix);

  *kindr = kindr::minimal::QuatTransformationTemplate<Scalar>(temp_rot_quat,
                                                              temp_translation);
}

// Overload taking a nested array (4 rows × 4 cols). Slightly closer in shape
// to the original XmlRpc 4x4 structure for callers that prefer it.
template <typename Scalar>
void xmlRpcToKindr(const std::vector<std::vector<double>>& mat,
                   kindr::minimal::QuatTransformationTemplate<Scalar>* kindr) {
  if (kindr == nullptr) {
    LOG(ERROR) << "Null pointer given";
    return;
  }
  if (mat.size() != 4) {
    LOG(ERROR) << "Transform matrix has " << mat.size()
               << " rows, expected 4.";
    return;
  }

  std::vector<double> flat;
  flat.reserve(16);
  for (size_t i = 0; i < 4; ++i) {
    if (mat[i].size() != 4) {
      LOG(ERROR) << "Transform matrix row " << i << " has " << mat[i].size()
                 << " columns, expected 4.";
      return;
    }
    for (size_t j = 0; j < 4; ++j) flat.push_back(mat[i][j]);
  }
  xmlRpcToKindr(flat, kindr);
}

}  // namespace minimal
}  // namespace kindr

#endif  // MINKINDR_CONVERSIONS_KINDR_XML_H
