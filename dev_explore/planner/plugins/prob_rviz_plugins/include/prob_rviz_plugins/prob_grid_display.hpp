#ifndef PROB_RVIZ_PLUGINS__PROB_GRID_DISPLAY_HPP_
#define PROB_RVIZ_PLUGINS__PROB_GRID_DISPLAY_HPP_

#include <memory>
#include <string>
#include <vector>

#ifndef Q_MOC_RUN
#include <OgreTexture.h>
#include <OgreSharedPtr.h>
#endif

#include "nav_msgs/msg/occupancy_grid.hpp"
#include "prob_msgs/msg/prob_grid.hpp"
#include "rclcpp/time.hpp"

#include "rviz_common/message_filter_display.hpp"
#include "rviz_default_plugins/displays/map/swatch.hpp"
#include "rviz_default_plugins/visibility_control.hpp"

namespace rviz_common
{
namespace properties
{
class BoolProperty;
class EnumProperty;
class FloatProperty;
class IntProperty;
class QuaternionProperty;
class VectorProperty;
}  // namespace properties
}  // namespace rviz_common

namespace prob_rviz_plugins
{
using rviz_default_plugins::displays::Swatch;

class ProbGridDisplay : public rviz_common::MessageFilterDisplay<prob_msgs::msg::ProbGrid>
{
  Q_OBJECT

public:
  ProbGridDisplay();
  ~ProbGridDisplay() override;

  void onInitialize() override;
  void fixedFrameChanged() override;
  void reset() override;

public Q_SLOTS:
  void showMap();

Q_SIGNALS:
  void mapUpdated();

protected Q_SLOTS:
  void updateAlpha();
  void updateDrawUnder() const;
  void updatePalette();
  void transformMap();

protected:
  void updateTopic() override;
  void update(float wall_dt, float ros_dt) override;
  void onEnable() override;
  void processMessage(const prob_msgs::msg::ProbGrid::ConstSharedPtr msg) override;

  void clear();
  void showValidMap();
  void resetSwatchesIfNecessary(size_t width, size_t height, float resolution);
  void createSwatches();
  void doubleSwatchNumber(
    size_t & swatch_width, size_t & swatch_height,
    int & number_swatches) const;
  void tryCreateSwatches(
    size_t width,
    size_t height,
    float resolution,
    size_t swatch_width,
    size_t swatch_height,
    int number_swatches);
  size_t getEffectiveDimension(size_t map_dimension, size_t swatch_dimension, size_t position);
  void updateSwatches() const;

  std::vector<std::shared_ptr<Swatch>> swatches_{};
  std::vector<Ogre::TexturePtr> palette_textures_{};
  std::vector<bool> color_scheme_transparency_{};
  bool loaded_{false};

  float resolution_{0.0F};
  size_t width_{0};
  size_t height_{0};
  std::string frame_{};
  nav_msgs::msg::OccupancyGrid current_map_{};

  rviz_common::properties::FloatProperty * resolution_property_;
  rviz_common::properties::IntProperty * width_property_;
  rviz_common::properties::IntProperty * height_property_;
  rviz_common::properties::VectorProperty * position_property_;
  rviz_common::properties::QuaternionProperty * orientation_property_;
  rviz_common::properties::FloatProperty * alpha_property_;
  rviz_common::properties::BoolProperty * draw_under_property_;
  rviz_common::properties::EnumProperty * color_scheme_property_;
  rviz_common::properties::BoolProperty * transform_timestamp_property_;
};

}  // namespace prob_rviz_plugins

#endif  // PROB_RVIZ_PLUGINS__PROB_GRID_DISPLAY_HPP_
