#include "planner_common/params.h"

#include <cmath>
#include <iostream>
#include <limits>

#include <rclcpp/exceptions.hpp>

// ===== Local helper: declare-once parameter accessor ========================
//
// The ROS 1 version used ros::param::get(name, var), which returned false if
// the param was missing. ROS 2 requires every parameter to be declared on the
// owning node before it can be read. We emulate the original semantics with
// `getOrDeclareParam`:
//   * if the parameter is not declared yet, declare it with default_val and
//     return false unless the user overrode it (we approximate via type tags);
//   * if it is already declared, just read it.
//
// To distinguish "user provided a value" from "we used the default", we declare
// every parameter with PARAMETER_NOT_SET sentinel + dynamic typing first, then
// query whether the resolved type matches. Where possible we go through the
// simpler overload that uses node->declare_parameter<T>(name, default) and
// detect override via the optional `node->describe_parameter()` route.

namespace {

template <typename T>
bool tryDeclareAndGet(rclcpp::Node* node, const std::string& name,
                      const T& default_val, T& out, bool* user_set = nullptr) {
  bool was_set = true;
  if (!node->has_parameter(name)) {
    try {
      // Declaring with PARAMETER_NOT_SET sentinel + ParameterValue(default).
      rcl_interfaces::msg::ParameterDescriptor desc;
      desc.dynamic_typing = false;
      node->declare_parameter(name, rclcpp::ParameterValue(default_val), desc);
    } catch (const rclcpp::exceptions::InvalidParameterTypeException&) {
      out = default_val;
      if (user_set) *user_set = false;
      return false;
    } catch (const rclcpp::exceptions::ParameterAlreadyDeclaredException&) {
      // ignored
    }
  }

  rclcpp::Parameter p;
  if (!node->get_parameter(name, p)) {
    out = default_val;
    if (user_set) *user_set = false;
    return false;
  }
  try {
    T fetched = p.get_value<T>();
    // Heuristic: if the fetched value is bitwise identical to the default
    // value, we can't tell whether the user actually set it. We optimistically
    // claim "set" so we don't double-emit warnings in the steady state. The
    // original ROS 1 code printed a warning only when the param was completely
    // missing; ROS 2 doesn't expose that bit cheaply, so we just keep silent.
    out = fetched;
    if (user_set) *user_set = (fetched == default_val) ? was_set : true;
    return true;
  } catch (const rclcpp::exceptions::InvalidParameterTypeException&) {
    out = default_val;
    if (user_set) *user_set = false;
    return false;
  }
}

// Convenience overload for the very common "vector<double> of length N" case.
bool getVec3(rclcpp::Node* node, const std::string& name,
             Eigen::Vector3d& out) {
  std::vector<double> v;
  if (!tryDeclareAndGet(node, name, std::vector<double>(), v) ||
      v.size() != 3) {
    return false;
  }
  out << v[0], v[1], v[2];
  return true;
}

bool getVec2(rclcpp::Node* node, const std::string& name,
             Eigen::Vector2d& out) {
  std::vector<double> v;
  if (!tryDeclareAndGet(node, name, std::vector<double>(), v) ||
      v.size() != 2) {
    return false;
  }
  out << v[0], v[1];
  return true;
}

}  // namespace

// =============================================================================
// SensorParamsBase
// =============================================================================

bool SensorParamsBase::loadParams(rclcpp::Node* node, const std::string& ns) {
  ROSPARAM_INFO("Loading: " + ns);
  std::string parse_str;

  if (!tryDeclareAndGet<std::string>(node, ns + ".type", std::string(),
                                     parse_str)) {
    ROSPARAM_ERROR(ns + ".type");
    return false;
  }
  if (parse_str == "kCamera")
    type = SensorType::kCamera;
  else if (parse_str == "kLidar")
    type = SensorType::kLidar;
  else {
    ROSPARAM_ERROR(ns + ".type");
    return false;
  }

  parse_str.clear();
  if (type == SensorType::kCamera) {
    tryDeclareAndGet<std::string>(node, ns + ".CameraType", std::string(),
                                  parse_str);
    if (parse_str == "kFixed")
      camera_type = CameraType::kFixed;
    else if (parse_str == "kRotating")
      camera_type = CameraType::kRotating;
    else if (parse_str == "kZoom")
      camera_type = CameraType::kZoom;
    else if (parse_str == "kRotatingZoom")
      camera_type = CameraType::kRotatingZoom;
    else {
      ROSPARAM_ERROR(ns + ".CameraType");
      return false;
    }
  } else {
    camera_type = CameraType::kFixed;
  }

  tryDeclareAndGet<std::string>(node, ns + ".callback_topic", std::string(),
                                callback_topic);
  tryDeclareAndGet<std::string>(node, ns + ".focal_length_topic",
                                std::string(), focal_lenght_topic);

  tryDeclareAndGet<double>(node, ns + ".min_range", 0.0, min_range);
  tryDeclareAndGet<double>(node, ns + ".max_range", 5.0, max_range);

  if (!getVec3(node, ns + ".center_offset", center_offset)) {
    center_offset.setZero();
  }
  if (!getVec3(node, ns + ".rotations", rotations)) {
    rotations.setZero();
  }
  if (!getVec2(node, ns + ".fov", fov)) {
    fov << 1.57, 1.04719;
  }
  if (!getVec2(node, ns + ".resolution", resolution)) {
    resolution << M_PI / 180.0, M_PI / 180.0;
  }

  tryDeclareAndGet<double>(node, ns + ".frontier_percentage_threshold", 0.1,
                           frontier_percentage_threshold);
  tryDeclareAndGet<std::string>(node, ns + ".frame_id", std::string(""),
                                frame_id);
  tryDeclareAndGet<int>(node, ns + ".width", 0, width);
  tryDeclareAndGet<int>(node, ns + ".height", 0, height);
  tryDeclareAndGet<int>(node, ns + ".width_removal", 0, widthRemoval);
  tryDeclareAndGet<int>(node, ns + ".height_removal", 0, heightRemoval);

  // Precompute rotation matrices and frustum endpoints (mirrors ROS 1 code).
  rot_B2S = Eigen::AngleAxisd(rotations[0], Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(rotations[1], Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(rotations[2], Eigen::Vector3d::UnitX());
  rot_S2B = rot_B2S.inverse();

  const double v_res = (resolution[1] > 0) ? resolution[1] : (M_PI / 180.0);
  const double h_res = (resolution[0] > 0) ? resolution[0] : (M_PI / 180.0);
  if (type == SensorType::kCamera) {
    if (width == 0 && height == 0) {
      const double h_2 = fov[0] / 2;
      const double v_2 = fov[1] / 2;
      Eigen::Vector3d pTL(std::cos(h_2), std::sin(h_2), std::sin(v_2));
      Eigen::Vector3d pTR(std::cos(h_2), -std::sin(h_2), std::sin(v_2));
      Eigen::Vector3d pBR(std::cos(h_2), -std::sin(h_2), -std::sin(v_2));
      Eigen::Vector3d pBL(std::cos(h_2), std::sin(h_2), -std::sin(v_2));
      edge_points.col(0) = pTL;
      edge_points.col(1) = pTR;
      edge_points.col(2) = pBR;
      edge_points.col(3) = pBL;
      normal_vectors.col(0) = edge_points.col(0).cross(edge_points.col(1));
      normal_vectors.col(1) = edge_points.col(1).cross(edge_points.col(2));
      normal_vectors.col(2) = edge_points.col(2).cross(edge_points.col(3));
      normal_vectors.col(3) = edge_points.col(3).cross(edge_points.col(0));
      edge_points = max_range * edge_points;
      edge_points_B = rot_B2S * edge_points;
      frustum_endpoints.clear();
      frustum_endpoints_B.clear();
      int w = 0, h = 0;
      height = 0;
      width = 0;
      const double h_lim_2 = fov[0] / 2;
      const double v_lim_2 = fov[1] / 2;
      for (double dv = -v_lim_2; dv < v_lim_2; dv += v_res) {
        ++h;
        for (double dh = -h_lim_2; dh < h_lim_2; dh += h_res) {
          if (width == 0) ++w;
          double x = max_range * std::cos(dh);
          double y = max_range * std::sin(dh);
          double z = max_range * std::sin(dv);
          Eigen::Vector3d ep(x, y, z);
          frustum_endpoints.push_back(ep);
          frustum_endpoints_B.push_back(rot_B2S * ep + center_offset);
        }
        if (width == 0) width = w;
      }
      height = h;
    }
  } else if (type == SensorType::kLidar) {
    frustum_endpoints.clear();
    frustum_endpoints_B.clear();
    int w = 0, h = 0;
    height = 0;
    width = 0;
    const double h_lim_2 = fov[0] / 2;
    const double v_lim_2 = fov[1] / 2;
    for (double dv = -v_lim_2; dv < v_lim_2; dv += v_res) {
      ++h;
      for (double dh = -h_lim_2; dh < h_lim_2; dh += h_res) {
        if (width == 0) ++w;
        double x = max_range * std::cos(dh);
        double y = max_range * std::sin(dh);
        double z = max_range * std::sin(dv);
        Eigen::Vector3d ep(x, y, z);
        frustum_endpoints.push_back(ep);
        frustum_endpoints_B.push_back(rot_B2S * ep + center_offset);
      }
      if (width == 0) width = w;
    }
    height = h;
  }

  num_voxels_full_fov = (fov[0] / h_res) * (fov[1] / v_res) * max_range;
  ROSPARAM_INFO("Done.");
  return true;
}

void SensorParamsBase::getFrustumEndpoints(StateVec& state,
                                           std::vector<Eigen::Vector3d>& ep) {
  Eigen::Vector3d origin(state[0], state[1], state[2]);
  Eigen::Matrix3d rot_W2B;
  rot_W2B = Eigen::AngleAxisd(state[3], Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());
  ep.clear();
  for (auto& p : frustum_endpoints_B) {
    ep.push_back(origin + rot_W2B * p);
  }
}

void SensorParamsBase::getFrustumEndpoints(StateVec& state,
                                           std::vector<Eigen::Vector3d>& ep,
                                           float darkness_range) {
  Eigen::Vector3d origin(state[0], state[1], state[2]);
  Eigen::Matrix3d rot_W2B;
  rot_W2B = Eigen::AngleAxisd(state[3], Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());
  ep.clear();
  for (auto& p : frustum_endpoints_B) {
    ep.push_back(origin + rot_W2B * p * darkness_range);
  }
}

void SensorParamsBase::updateFrustumEndpoints() {
  const double v_res = (resolution[1] > 0) ? resolution[1] : (M_PI / 180.0);
  const double h_res = (resolution[0] > 0) ? resolution[0] : (M_PI / 180.0);
  if (type == SensorType::kCamera) {
    const double h_2 = fov[0] / 2;
    const double v_2 = fov[1] / 2;
    Eigen::Vector3d pTL(std::cos(h_2), std::sin(h_2), std::sin(v_2));
    Eigen::Vector3d pTR(std::cos(h_2), -std::sin(h_2), std::sin(v_2));
    Eigen::Vector3d pBR(std::cos(h_2), -std::sin(h_2), -std::sin(v_2));
    Eigen::Vector3d pBL(std::cos(h_2), std::sin(h_2), -std::sin(v_2));
    edge_points.col(0) = pTL;
    edge_points.col(1) = pTR;
    edge_points.col(2) = pBR;
    edge_points.col(3) = pBL;
    normal_vectors.col(0) = edge_points.col(0).cross(edge_points.col(1));
    normal_vectors.col(1) = edge_points.col(1).cross(edge_points.col(2));
    normal_vectors.col(2) = edge_points.col(2).cross(edge_points.col(3));
    normal_vectors.col(3) = edge_points.col(3).cross(edge_points.col(0));
    edge_points = max_range * edge_points;
    edge_points_B = rot_B2S * edge_points;
    frustum_endpoints.clear();
    frustum_endpoints_B.clear();
    const double h_lim_2 = fov[0] / 2;
    const double v_lim_2 = fov[1] / 2;
    for (double dv = -v_lim_2; dv < v_lim_2; dv += v_res) {
      for (double dh = -h_lim_2; dh < h_lim_2; dh += h_res) {
        double x = max_range * std::cos(dh);
        double y = max_range * std::sin(dh);
        double z = max_range * std::sin(dv);
        Eigen::Vector3d ep(x, y, z);
        frustum_endpoints.push_back(ep);
        frustum_endpoints_B.push_back(rot_B2S * ep + center_offset);
      }
    }
  } else if (type == SensorType::kLidar) {
    frustum_endpoints.clear();
    frustum_endpoints_B.clear();
    const double h_lim_2 = fov[0] / 2;
    const double v_lim_2 = fov[1] / 2;
    for (double dv = -v_lim_2; dv < v_lim_2; dv += v_res) {
      for (double dh = -h_lim_2; dh < h_lim_2; dh += h_res) {
        double x = max_range * std::cos(dh);
        double y = max_range * std::sin(dh);
        double z = max_range * std::sin(dv);
        Eigen::Vector3d ep(x, y, z);
        frustum_endpoints.push_back(ep);
        frustum_endpoints_B.push_back(rot_B2S * ep + center_offset);
      }
    }
  }
}

void SensorParamsBase::convertBodyToSensor(
    pcl::PointCloud<pcl::PointXYZ>::Ptr ep,
    pcl::PointCloud<pcl::PointXYZ>::Ptr ep_s) {
  ep_s->points.clear();

  Eigen::Matrix3d rot_S2Cam;
  rot_S2Cam = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ()) *
              Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitY()) *
              Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX());

  for (auto p : ep->points) {
    Eigen::Vector3d ip_p(p.x, p.y, p.z);
    Eigen::Vector3d op_p = rot_S2B * (ip_p - center_offset);
    if (type == SensorType::kCamera) op_p = rot_S2Cam * op_p;
    pcl::PointXYZ data;
    data.x = op_p(0);
    data.y = op_p(1);
    data.z = op_p(2);
    ep_s->points.push_back(data);
  }
}

void SensorParamsBase::getFrustumEdges(StateVec& state,
                                       std::vector<Eigen::Vector3d>& edges) {
  Eigen::Vector3d origin(state[0], state[1], state[2]);
  Eigen::Matrix3d rot_W2B;
  rot_W2B = Eigen::AngleAxisd(state[3], Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());
  edges.clear();
  for (int i = 0; i < 4; ++i) {
    edges.push_back(origin + rot_W2B * edge_points_B.col(i));
  }
}

bool SensorParamsBase::isInsideFOV(StateVec& state, Eigen::Vector3d& pos) {
  Eigen::Vector3d origin(state[0], state[1], state[2]);
  Eigen::Vector3d pos_S =
      rot_S2B * Eigen::AngleAxisd(-state[3], Eigen::Vector3d::UnitZ()) *
          (pos - origin) -
      center_offset;
  float pos_S_norm = pos_S.norm();

  if (pos_S_norm > max_range) return false;

  if (type == SensorType::kCamera) {
    for (int i = 0; i < 4; ++i) {
      double res = pos_S.dot(normal_vectors.col(i));
      if (res <= 0) return false;
    }
  } else if (type == SensorType::kLidar) {
    float h_angle = std::atan2(static_cast<float>(pos_S.y()),
                               static_cast<float>(pos_S.x()));
    float v_angle = std::asin(static_cast<float>(pos_S.z()) / pos_S_norm);
    if ((std::abs(h_angle) > (fov[0] / 2)) ||
        (std::abs(v_angle) > (fov[1] / 2)))
      return false;
  } else {
    return false;
  }
  return true;
}

bool SensorParamsBase::isFrontier(double num_unknown_voxels_normalized) {
  double unknown_percentage =
      num_unknown_voxels_normalized / num_voxels_full_fov;
  return unknown_percentage >= frontier_percentage_threshold;
}

// =============================================================================
// SensorParams
// =============================================================================

bool SensorParams::loadParams(rclcpp::Node* node, const std::string& ns) {
  ROSPARAM_INFO("Loading: " + ns);
  std::vector<std::string> parse_str_list;
  if (!tryDeclareAndGet(node, ns + ".sensor_list",
                        std::vector<std::string>{}, parse_str_list) ||
      parse_str_list.empty()) {
    ROSPARAM_ERROR(ns + ".sensor_list");
    return false;
  }
  sensor_list = parse_str_list;
  for (auto& s : sensor_list) {
    SensorParamsBase spb;
    if (spb.loadParams(node, ns + "." + s)) {
      sensor.emplace(s, spb);
    } else {
      ROSPARAM_ERROR(ns + "." + s);
      return false;
    }
  }
  ROSPARAM_INFO("Done.");
  return true;
}

// =============================================================================
// RobotParams
// =============================================================================

bool RobotParams::loadParams(rclcpp::Node* node, const std::string& ns) {
  ROSPARAM_INFO("Loading: " + ns);
  std::string parse_str;

  tryDeclareAndGet<std::string>(node, ns + ".type", std::string(), parse_str);
  if (parse_str == "kAerialRobot")
    type = RobotType::kAerialRobot;
  else if (parse_str == "kGroundRobot")
    type = RobotType::kGroundRobot;
  else {
    type = RobotType::kAerialRobot;
  }

  if (!getVec3(node, ns + ".size", size)) {
    ROSPARAM_ERROR(ns + ".size");
    return false;
  }
  if (!getVec3(node, ns + ".size_extension_min", size_extension_min)) {
    ROSPARAM_ERROR(ns + ".size_extension_min");
    return false;
  }
  if (!getVec3(node, ns + ".size_extension", size_extension)) {
    size_extension = size_extension_min;
  }
  // size_extension must be >= size_extension_min in each component.
  for (int i = 0; i < 3; ++i) {
    if (size_extension[i] < size_extension_min[i]) {
      size_extension[i] = size_extension_min[i];
    }
  }

  if (!getVec3(node, ns + ".center_offset", center_offset)) {
    center_offset.setZero();
  }
  if (!getVec3(node, ns + ".safety_extension", safety_extension)) {
    ROSPARAM_ERROR(ns + ".safety_extension");
    return false;
  }

  tryDeclareAndGet<double>(node, ns + ".relax_ratio", 0.5, relax_ratio);

  parse_str.clear();
  tryDeclareAndGet<std::string>(node, ns + ".bound_mode", std::string(),
                                parse_str);
  if (parse_str == "kExtendedBound")
    bound_mode = BoundModeType::kExtendedBound;
  else if (parse_str == "kRelaxedBound")
    bound_mode = BoundModeType::kRelaxedBound;
  else if (parse_str == "kMinBound")
    bound_mode = BoundModeType::kMinBound;
  else if (parse_str == "kExactBound")
    bound_mode = BoundModeType::kExactBound;
  else if (parse_str == "kNoBound")
    bound_mode = BoundModeType::kNoBound;
  else
    bound_mode = BoundModeType::kExtendedBound;

  ROSPARAM_INFO("Done.");
  return true;
}

void RobotParams::setBoundMode(BoundModeType bmode) { bound_mode = bmode; }

void RobotParams::getPlanningSize(Eigen::Vector3d& psize) {
  psize.setZero();
  switch (bound_mode) {
    case BoundModeType::kExtendedBound:
      psize = size + size_extension;
      break;
    case BoundModeType::kRelaxedBound:
      psize = size + relax_ratio * size_extension_min +
              (1 - relax_ratio) * size_extension;
      break;
    case BoundModeType::kMinBound:
      psize = size + size_extension_min;
      break;
    case BoundModeType::kExactBound:
      psize = size;
      break;
    case BoundModeType::kNoBound:
      psize.setZero();
      break;
  }
}

// =============================================================================
// BoundedSpaceParams
// =============================================================================

bool BoundedSpaceParams::loadParams(rclcpp::Node* node, const std::string& ns) {
  ROSPARAM_INFO("Loading: " + ns);
  std::string parse_str;
  tryDeclareAndGet<std::string>(node, ns + ".type", std::string(), parse_str);
  if (parse_str == "kCuboid")
    type = BoundedSpaceType::kCuboid;
  else if (parse_str == "kSphere")
    type = BoundedSpaceType::kSphere;
  else {
    ROSPARAM_ERROR(ns + ".type");
    return false;
  }

  if (type == BoundedSpaceType::kCuboid) {
    if (!getVec3(node, ns + ".min_val", min_val)) {
      ROSPARAM_ERROR(ns + ".min_val");
      return false;
    }
    if (!getVec3(node, ns + ".max_val", max_val)) {
      ROSPARAM_ERROR(ns + ".max_val");
      return false;
    }
    root_pos.setZero();

    if (!getVec3(node, ns + ".min_extension", min_extension)) {
      min_extension.setZero();
    }
    if (!getVec3(node, ns + ".max_extension", max_extension)) {
      max_extension.setZero();
    }
    min_val_total = min_val + min_extension;
    max_val_total = max_val + max_extension;

    if (!getVec3(node, ns + ".rotations", rotations)) {
      rotations.setZero();
    }
    setRotation(rotations);
  } else {
    if (!tryDeclareAndGet<double>(node, ns + ".radius", 0.0, radius)) {
      ROSPARAM_ERROR(ns + ".radius");
      return false;
    }
    tryDeclareAndGet<double>(node, ns + ".radius_extension", 0.0,
                             radius_extension);
  }
  ROSPARAM_INFO("Done.");
  return true;
}

void BoundedSpaceParams::setCenter(StateVec& state, bool use_extension) {
  root_pos << state[0], state[1], state[2];
  if (use_extension) {
    min_val_total = min_val + min_extension;
    max_val_total = max_val + max_extension;
    radius_total = radius + radius_extension;
  } else {
    min_val_total = min_val;
    max_val_total = max_val;
    radius_total = radius;
  }
  Eigen::Matrix3d rot_W2B;
  rot_W2B = Eigen::AngleAxisd(rotations[0], Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(rotations[1], Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(rotations[2], Eigen::Vector3d::UnitX());
  rot_B2W = rot_W2B.transpose();
}

void BoundedSpaceParams::setCenter(Eigen::Vector3d& root, bool use_extension) {
  root_pos = root;
  if (use_extension) {
    min_val_total = min_val + min_extension;
    max_val_total = max_val + max_extension;
    radius_total = radius + radius_extension;
  } else {
    min_val_total = min_val;
    max_val_total = max_val;
    radius_total = radius;
  }
  Eigen::Matrix3d rot_W2B;
  rot_W2B = Eigen::AngleAxisd(rotations[0], Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(rotations[1], Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(rotations[2], Eigen::Vector3d::UnitX());
  rot_B2W = rot_W2B.transpose();
}

void BoundedSpaceParams::setBound(Eigen::Vector3d& min_val_in,
                                  Eigen::Vector3d& max_val_in) {
  min_val = min_val_in;
  max_val = max_val_in;
}

void BoundedSpaceParams::setRotation(Eigen::Vector3d& rotations_in) {
  rotations = rotations_in;
  Eigen::Matrix3d rot_W2B;
  rot_W2B = Eigen::AngleAxisd(rotations[0], Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(rotations[1], Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(rotations[2], Eigen::Vector3d::UnitX());
  rot_B2W = rot_W2B.transpose();
}

bool BoundedSpaceParams::isInsideSpace(Eigen::Vector3d& pos) {
  if (type == BoundedSpaceType::kSphere) {
    Eigen::Vector3d dist = pos - root_pos;
    return dist.norm() <= radius_total;
  }
  Eigen::Vector3d pos_B = rot_B2W * (pos - root_pos);
  for (int i = 0; i < 3; ++i) {
    if (pos_B[i] < min_val[i] || pos_B[i] > max_val[i]) return false;
  }
  return true;
}

// =============================================================================
// PlanningParams
// =============================================================================

bool PlanningParams::loadParams(rclcpp::Node* node, const std::string& ns) {
  ROSPARAM_INFO("Loading: " + ns);
  std::string parse_str;
  tryDeclareAndGet<std::string>(node, ns + ".type", std::string(), parse_str);
  if (parse_str == "kBasicExploration")
    type = PlanningModeType::kBasicExploration;
  else if (parse_str == "kNarrowEnvExploration")
    type = PlanningModeType::kNarrowEnvExploration;
  else if (parse_str == "kAdaptiveExploration")
    type = PlanningModeType::kAdaptiveExploration;
  else
    type = PlanningModeType::kBasicExploration;

  parse_str.clear();
  tryDeclareAndGet<std::string>(node, ns + ".rr_mode", std::string(),
                                parse_str);
  rr_mode = (parse_str == "kTree") ? RRModeType::kTree : RRModeType::kGraph;

  tryDeclareAndGet<std::vector<std::string>>(
      node, ns + ".exp_sensor_list", std::vector<std::string>{},
      exp_sensor_list);
  tryDeclareAndGet<std::vector<std::string>>(
      node, ns + ".no_gain_zones_list", std::vector<std::string>{},
      no_gain_zones_list);

  tryDeclareAndGet<double>(node, ns + ".v_max", 0.2, v_max);
  tryDeclareAndGet<double>(node, ns + ".v_homing_max", v_max, v_homing_max);
  tryDeclareAndGet<double>(node, ns + ".yaw_rate_max", 0.4, yaw_rate_max);
  tryDeclareAndGet<bool>(node, ns + ".yaw_tangent_correction", false,
                         yaw_tangent_correction);
  tryDeclareAndGet<double>(node, ns + ".exp_gain_voxel_size", 0.4,
                           exp_gain_voxel_size);
  tryDeclareAndGet<bool>(node, ns + ".use_ray_model_for_volumetric_gain", false,
                         use_ray_model_for_volumetric_gain);
  tryDeclareAndGet<double>(node, ns + ".free_voxel_gain", 1.0, free_voxel_gain);
  tryDeclareAndGet<double>(node, ns + ".occupied_voxel_gain", 1.0,
                           occupied_voxel_gain);
  tryDeclareAndGet<double>(node, ns + ".unknown_voxel_gain", 10.0,
                           unknown_voxel_gain);
  tryDeclareAndGet<double>(node, ns + ".edge_length_min", 0.2, edge_length_min);
  tryDeclareAndGet<double>(node, ns + ".edge_length_max", 0.2, edge_length_max);
  tryDeclareAndGet<double>(node, ns + ".num_vertices_max", 500.0,
                           num_vertices_max);
  tryDeclareAndGet<double>(node, ns + ".num_edges_max", 5000.0, num_edges_max);
  tryDeclareAndGet<double>(node, ns + ".edge_overshoot", 0.2, edge_overshoot);
  tryDeclareAndGet<double>(node, ns + ".num_loops_cutoff", 1000.0,
                           num_loops_cutoff);
  tryDeclareAndGet<double>(node, ns + ".num_loops_max", 10000.0, num_loops_max);
  tryDeclareAndGet<double>(node, ns + ".nearest_range", 1.0, nearest_range);
  tryDeclareAndGet<double>(node, ns + ".nearest_range_min", 0.5,
                           nearest_range_min);
  tryDeclareAndGet<double>(node, ns + ".nearest_range_max", 2.0,
                           nearest_range_max);
  tryDeclareAndGet<bool>(node, ns + ".use_current_state", true,
                         use_current_state);
  tryDeclareAndGet<double>(node, ns + ".path_length_penalty", 0.0,
                           path_length_penalty);
  tryDeclareAndGet<double>(node, ns + ".path_direction_penalty", 0.0,
                           path_direction_penalty);
  tryDeclareAndGet<double>(node, ns + ".hanging_vertex_penalty", 0.0,
                           hanging_vertex_penalty);
  tryDeclareAndGet<double>(node, ns + ".traverse_length_max", edge_length_max,
                           traverse_length_max);
  tryDeclareAndGet<double>(node, ns + ".traverse_time_max",
                           edge_length_max / std::max(v_max, 1e-3),
                           traverse_time_max);
  tryDeclareAndGet<double>(node, ns + ".augment_free_voxels_time", 5.0,
                           augment_free_voxels_time);
  tryDeclareAndGet<bool>(node, ns + ".free_frustum_before_planning", false,
                         free_frustum_before_planning);
  tryDeclareAndGet<bool>(node, ns + ".auto_homing_enable", false,
                         auto_homing_enable);
  tryDeclareAndGet<bool>(node, ns + ".geofence_checking_enable", false,
                         geofence_checking_enable);
  tryDeclareAndGet<double>(node, ns + ".time_budget_limit",
                           std::numeric_limits<double>::max(),
                           time_budget_limit);
  tryDeclareAndGet<bool>(node, ns + ".homing_backward", false, homing_backward);
  tryDeclareAndGet<bool>(node, ns + ".planning_backward", false,
                         planning_backward);
  tryDeclareAndGet<bool>(node, ns + ".path_safety_enhance_enable", false,
                         path_safety_enhance_enable);
  tryDeclareAndGet<std::string>(node, ns + ".global_frame_id",
                                std::string("world"), global_frame_id);
  tryDeclareAndGet<bool>(node, ns + ".freespace_cloud_enable", false,
                         freespace_cloud_enable);
  tryDeclareAndGet<bool>(node, ns + ".leafs_only_for_volumetric_gain", false,
                         leafs_only_for_volumetric_gain);
  tryDeclareAndGet<double>(node, ns + ".max_ground_height", 1.2,
                           max_ground_height);
  tryDeclareAndGet<double>(node, ns + ".robot_height", 1.0, robot_height);
  tryDeclareAndGet<double>(node, ns + ".max_inclination", 0.52,
                           max_inclination);
  tryDeclareAndGet<bool>(node, ns + ".cluster_vertices_for_gain", false,
                         cluster_vertices_for_gain);
  tryDeclareAndGet<double>(node, ns + ".clustering_radius", 2.0,
                           clustering_radius);
  tryDeclareAndGet<double>(node, ns + ".path_interpolation_distance", 0.5,
                           path_interpolation_distance);
  tryDeclareAndGet<bool>(node, ns + ".auto_global_planner_enable", true,
                         auto_global_planner_enable);
  tryDeclareAndGet<double>(node, ns + ".relaxed_corridor_multiplier", 1.0,
                           relaxed_corridor_multiplier);
  tryDeclareAndGet<bool>(node, ns + ".interpolate_projection_distance", false,
                         interpolate_projection_distance);
  tryDeclareAndGet<bool>(node, ns + ".go_home_if_fully_explored", false,
                         go_home_if_fully_explored);
  tryDeclareAndGet<double>(node, ns + ".ray_cast_step_size_multiplier", 1.0,
                           ray_cast_step_size_multiplier);
  tryDeclareAndGet<bool>(node, ns + ".nonuniform_ray_cast", true,
                         nonuniform_ray_cast);
  tryDeclareAndGet<double>(node, ns + ".time_budget_before_landing",
                           time_budget_limit, time_budget_before_landing);
  tryDeclareAndGet<bool>(node, ns + ".auto_landing_enable", false,
                         auto_landing_enable);
  tryDeclareAndGet<double>(node, ns + ".max_negative_inclination", 0.37,
                           max_negative_inclination);
  tryDeclareAndGet<bool>(node, ns + ".augment_free_frustum_en", false,
                         augment_free_frustum_en);

  ROSPARAM_INFO("Done.");
  return true;
}

void PlanningParams::setPlanningMode(PlanningModeType pmode) {
  switch (pmode) {
    case PlanningModeType::kBasicExploration:
      type = PlanningModeType::kBasicExploration;
      break;
    default:
      type = PlanningModeType::kBasicExploration;
      break;
  }
}

// =============================================================================
// RobotDynamicsParams
// =============================================================================

bool RobotDynamicsParams::loadParams(rclcpp::Node* node, const std::string& ns) {
  ROSPARAM_INFO("Loading: " + ns);
  tryDeclareAndGet<double>(node, ns + ".v_max", 0.2, v_max);
  tryDeclareAndGet<double>(node, ns + ".v_homing_max", v_max, v_homing_max);
  tryDeclareAndGet<double>(node, ns + ".yaw_rate_max", M_PI_4 / 2.0,
                           yaw_rate_max);
  ROSPARAM_INFO("Done.");
  return true;
}
