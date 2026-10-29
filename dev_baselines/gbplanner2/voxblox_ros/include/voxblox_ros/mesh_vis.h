// The MIT License (MIT)
// Copyright (c) 2014 Matthew Klingensmith and Ivan Dryanovski
// Mesh output taken from open_chisel: github.com/personalrobotics/OpenChisel

#ifndef VOXBLOX_ROS_MESH_VIS_H_
#define VOXBLOX_ROS_MESH_VIS_H_

#include <algorithm>
#include <limits>
#include <string>

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <voxblox/core/common.h>
#include <voxblox/integrator/esdf_integrator.h>
#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/mesh/mesh.h>
#include <voxblox/mesh/mesh_layer.h>
#include <voxblox_msgs/msg/mesh.hpp>

#include "voxblox_ros/conversions.h"

namespace voxblox {

enum ColorMode {
  kColor = 0,
  kHeight,
  kNormals,
  kGray,
  kLambert,
  kLambertColor
};

inline ColorMode getColorModeFromString(const std::string& color_mode_string) {
  if (color_mode_string.empty()) {
    return ColorMode::kColor;
  } else {
    if (color_mode_string == "color" || color_mode_string == "colors") {
      return ColorMode::kColor;
    } else if (color_mode_string == "height") {
      return ColorMode::kHeight;
    } else if (color_mode_string == "normals") {
      return ColorMode::kNormals;
    } else if (color_mode_string == "lambert") {
      return ColorMode::kLambert;
    } else if (color_mode_string == "lambert_color") {
      return ColorMode::kLambertColor;
    } else {
      return ColorMode::kGray;
    }
  }
}

inline Point lambertShading(const Point& normal, const Point& light,
                            const Point& color) {
  return std::max<FloatingPoint>(normal.dot(light), 0.0f) * color;
}

inline void lambertColorFromColorAndNormal(
    const Color& color, const Point& normal,
    std_msgs::msg::ColorRGBA* color_msg) {
  const Point light_dir = Point(0.8f, -0.2f, 0.7f).normalized();
  const Point light_dir2 = Point(-0.5f, 0.2f, 0.2f).normalized();
  const Point ambient(0.2f, 0.2f, 0.2f);
  const Point color_pt(color.r / 255.0, color.g / 255.0, color.b / 255.0);

  Point lambert = lambertShading(normal, light_dir, color_pt) +
                  lambertShading(normal, light_dir2, color_pt) + ambient;

  color_msg->r = std::min<FloatingPoint>(lambert.x(), 1.0);
  color_msg->g = std::min<FloatingPoint>(lambert.y(), 1.0);
  color_msg->b = std::min<FloatingPoint>(lambert.z(), 1.0);
  color_msg->a = 1.0;
}

inline void lambertColorFromNormal(const Point& normal,
                                   std_msgs::msg::ColorRGBA* color_msg) {
  lambertColorFromColorAndNormal(Color(127, 127, 127), normal, color_msg);
}

inline void normalColorFromNormal(const Point& normal,
                                  std_msgs::msg::ColorRGBA* color_msg) {
  color_msg->r = normal.x() * 0.5 + 0.5;
  color_msg->g = normal.y() * 0.5 + 0.5;
  color_msg->b = normal.z() * 0.5 + 0.5;
  color_msg->a = 1.0;
}

inline void heightColorFromVertex(const Point& vertex,
                                  std_msgs::msg::ColorRGBA* color_msg) {
  const double min_z = -1;
  const double max_z = 10;
  double mapped_height = std::min<FloatingPoint>(
      std::max<FloatingPoint>((vertex.z() - min_z) / (max_z - min_z), 0.0),
      1.0);
  colorVoxbloxToMsg(rainbowColorMap(mapped_height), color_msg);
}

inline std_msgs::msg::ColorRGBA getVertexColor(const Mesh::ConstPtr& mesh,
                                               const ColorMode& color_mode,
                                               const size_t index) {
  std_msgs::msg::ColorRGBA color_msg;
  switch (color_mode) {
    case kColor:
      colorVoxbloxToMsg(mesh->colors[index], &color_msg);
      break;
    case kHeight:
      heightColorFromVertex(mesh->vertices[index], &color_msg);
      break;
    case kNormals:
      normalColorFromNormal(mesh->normals[index], &color_msg);
      break;
    case kLambert:
      lambertColorFromNormal(mesh->normals[index], &color_msg);
      break;
    case kLambertColor:
      lambertColorFromColorAndNormal(mesh->colors[index], mesh->normals[index],
                                     &color_msg);
      break;
    case kGray:
      color_msg.r = color_msg.g = color_msg.b = 0.5;
      color_msg.a = 1.0;
      break;
  }
  return color_msg;
}

// ROS 2 port note: the original took the message stamp from ros::Time::now()
// inside the helper. We expose it as a parameter so the caller (which has the
// node) provides it; default-constructed Time is also acceptable.
inline void generateVoxbloxMeshMsg(
    MeshLayer* mesh_layer, ColorMode color_mode,
    voxblox_msgs::msg::Mesh* mesh_msg,
    const builtin_interfaces::msg::Time& stamp =
        builtin_interfaces::msg::Time()) {
  CHECK_NOTNULL(mesh_msg);
  CHECK_NOTNULL(mesh_layer);

  mesh_msg->header.stamp = stamp;

  BlockIndexList mesh_indices;
  mesh_layer->getAllUpdatedMeshes(&mesh_indices);

  mesh_msg->block_edge_length = mesh_layer->block_size();
  mesh_msg->mesh_blocks.reserve(mesh_indices.size());

  for (const BlockIndex& block_index : mesh_indices) {
    Mesh::Ptr mesh = mesh_layer->getMeshPtrByIndex(block_index);

    voxblox_msgs::msg::MeshBlock mesh_block;
    mesh_block.index[0] = block_index.x();
    mesh_block.index[1] = block_index.y();
    mesh_block.index[2] = block_index.z();

    mesh_block.x.reserve(mesh->vertices.size());
    mesh_block.y.reserve(mesh->vertices.size());
    mesh_block.z.reserve(mesh->vertices.size());

    if (color_mode != kNormals) {
      mesh_block.r.reserve(mesh->vertices.size());
      mesh_block.g.reserve(mesh->vertices.size());
      mesh_block.b.reserve(mesh->vertices.size());
    }
    for (size_t i = 0u; i < mesh->vertices.size(); ++i) {
      const Point normalized_verticies =
          0.5f * (mesh_layer->block_size_inv() * mesh->vertices[i] -
                  block_index.cast<FloatingPoint>());

      CHECK_LE(normalized_verticies.squaredNorm(), 1.0f);
      CHECK((normalized_verticies.array() >= 0.0).all());

      mesh_block.x.push_back(std::numeric_limits<uint16_t>::max() *
                             normalized_verticies.x());
      mesh_block.y.push_back(std::numeric_limits<uint16_t>::max() *
                             normalized_verticies.y());
      mesh_block.z.push_back(std::numeric_limits<uint16_t>::max() *
                             normalized_verticies.z());

      if (color_mode != kNormals) {
        const std_msgs::msg::ColorRGBA color_msg =
            getVertexColor(mesh, color_mode, i);
        mesh_block.r.push_back(std::numeric_limits<uint8_t>::max() *
                               color_msg.r);
        mesh_block.g.push_back(std::numeric_limits<uint8_t>::max() *
                               color_msg.g);
        mesh_block.b.push_back(std::numeric_limits<uint8_t>::max() *
                               color_msg.b);
      }
    }

    mesh_msg->mesh_blocks.push_back(mesh_block);

    if (!mesh->hasVertices()) {
      mesh_layer->removeMesh(block_index);
    }

    mesh->updated = false;
  }
}

inline void generateVoxbloxMeshMsg(
    const MeshLayer::Ptr& mesh_layer, ColorMode color_mode,
    voxblox_msgs::msg::Mesh* mesh_msg,
    const builtin_interfaces::msg::Time& stamp =
        builtin_interfaces::msg::Time()) {
  CHECK_NOTNULL(mesh_msg);
  CHECK(mesh_layer);
  generateVoxbloxMeshMsg(mesh_layer.get(), color_mode, mesh_msg, stamp);
}

inline void fillMarkerWithMesh(
    const MeshLayer::ConstPtr& mesh_layer, ColorMode color_mode,
    visualization_msgs::msg::Marker* marker,
    const builtin_interfaces::msg::Time& stamp =
        builtin_interfaces::msg::Time()) {
  CHECK_NOTNULL(marker);
  marker->header.stamp = stamp;
  marker->ns = "mesh";
  marker->scale.x = 1;
  marker->scale.y = 1;
  marker->scale.z = 1;
  marker->pose.orientation.x = 0;
  marker->pose.orientation.y = 0;
  marker->pose.orientation.z = 0;
  marker->pose.orientation.w = 1;
  marker->type = visualization_msgs::msg::Marker::TRIANGLE_LIST;

  BlockIndexList mesh_indices;
  mesh_layer->getAllAllocatedMeshes(&mesh_indices);

  for (const BlockIndex& block_index : mesh_indices) {
    Mesh::ConstPtr mesh = mesh_layer->getMeshPtrByIndex(block_index);

    if (!mesh->hasVertices()) continue;
    if (color_mode == kColor || color_mode == kLambertColor) {
      CHECK(mesh->hasColors());
    }
    if (color_mode == kNormals || color_mode == kLambert ||
        color_mode == kLambertColor) {
      CHECK(mesh->hasNormals());
    }

    for (size_t i = 0u; i < mesh->vertices.size(); i++) {
      geometry_msgs::msg::Point point_msg;
      // Inline replacement for tf::pointEigenToMsg.
      point_msg.x = static_cast<double>(mesh->vertices[i].x());
      point_msg.y = static_cast<double>(mesh->vertices[i].y());
      point_msg.z = static_cast<double>(mesh->vertices[i].z());
      marker->points.push_back(point_msg);
      marker->colors.push_back(getVertexColor(mesh, color_mode, i));
    }
  }
}

inline void fillPointcloudWithMesh(
    const MeshLayer::ConstPtr& mesh_layer, ColorMode color_mode,
    pcl::PointCloud<pcl::PointXYZRGB>* pointcloud) {
  CHECK_NOTNULL(pointcloud);
  pointcloud->clear();

  BlockIndexList mesh_indices;
  mesh_layer->getAllAllocatedMeshes(&mesh_indices);

  for (const BlockIndex& block_index : mesh_indices) {
    Mesh::ConstPtr mesh = mesh_layer->getMeshPtrByIndex(block_index);

    if (!mesh->hasVertices()) continue;
    if (color_mode == kColor || color_mode == kLambertColor) {
      CHECK(mesh->hasColors());
    }
    if (color_mode == kNormals || color_mode == kLambert ||
        color_mode == kLambertColor) {
      CHECK(mesh->hasNormals());
    }

    for (size_t i = 0u; i < mesh->vertices.size(); i++) {
      pcl::PointXYZRGB point;
      point.x = mesh->vertices[i].x();
      point.y = mesh->vertices[i].y();
      point.z = mesh->vertices[i].z();

      Color color;
      colorMsgToVoxblox(getVertexColor(mesh, color_mode, i), &color);
      point.r = color.r;
      point.g = color.g;
      point.b = color.b;

      pointcloud->push_back(point);
    }
  }
}

}  // namespace voxblox

#endif  // VOXBLOX_ROS_MESH_VIS_H_
