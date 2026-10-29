#include "prob_rviz_plugins/prob_grid_display.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <OgreSceneManager.h>
#include <OgreSceneNode.h>
#include <OgreTechnique.h>
#include <OgreTextureManager.h>

#include "rviz_common/display_context.hpp"
#include "rviz_common/logging.hpp"
#include "rviz_common/msg_conversions.hpp"
#include "rviz_common/properties/bool_property.hpp"
#include "rviz_common/properties/enum_property.hpp"
#include "rviz_common/properties/float_property.hpp"
#include "rviz_common/properties/int_property.hpp"
#include "rviz_common/properties/quaternion_property.hpp"
#include "rviz_common/properties/vector_property.hpp"
#include "rviz_common/validate_floats.hpp"
#include "rviz_default_plugins/displays/map/palette_builder.hpp"
#include "rviz_rendering/material_manager.hpp"

namespace prob_rviz_plugins {
  using rviz_common::properties::StatusProperty;
  using rviz_default_plugins::displays::makeCostmapPalette;
  using rviz_default_plugins::displays::makeMapPalette;
  using rviz_default_plugins::displays::makeRawPalette;

  namespace {
    Ogre::TexturePtr makePaletteTexture(std::vector<unsigned char> palette_bytes) {
      Ogre::DataStreamPtr palette_stream;
      palette_stream.reset(
          new Ogre::MemoryDataStream(palette_bytes.data(), 256 * 4)
      );

      static int palette_tex_count = 0;
      const std::string texture_name =
          "ProbGridPaletteTexture" + std::to_string(palette_tex_count++);
      return Ogre::TextureManager::getSingleton().loadRawData(
          texture_name,
          "rviz_rendering",
          palette_stream,
          256,
          1,
          Ogre::PF_BYTE_RGBA,
          Ogre::TEX_TYPE_1D,
          0
      );
    }

    bool validateFloats(const nav_msgs::msg::OccupancyGrid& msg) {
      return rviz_common::validateFloats(msg.info.resolution) &&
             rviz_common::validateFloats(msg.info.origin);
    }
  } // namespace

  ProbGridDisplay::ProbGridDisplay()
      : loaded_(false), resolution_(0.0f), width_(0), height_(0) {
    connect(this, SIGNAL(mapUpdated()), this, SLOT(showMap()));

    alpha_property_ = new rviz_common::properties::FloatProperty(
        "Alpha",
        0.7f,
        "Amount of transparency to apply to the confidence map.",
        this,
        SLOT(updateAlpha())
    );
    alpha_property_->setMin(0);
    alpha_property_->setMax(1);

    color_scheme_property_ = new rviz_common::properties::EnumProperty(
        "Color Scheme",
        "raw",
        "How to color the confidence values.",
        this,
        SLOT(updatePalette())
    );
    color_scheme_property_->addOption("map", 0);
    color_scheme_property_->addOption("costmap", 1);
    color_scheme_property_->addOption("raw", 2);

    draw_under_property_ = new rviz_common::properties::BoolProperty(
        "Draw Behind",
        false,
        "Rendering option, controls whether or not the map is always drawn behind "
        "everything else.",
        this,
        SLOT(updateDrawUnder())
    );

    resolution_property_ = new rviz_common::properties::FloatProperty(
        "Resolution", 0, "Resolution of the map. (not editable)", this
    );
    resolution_property_->setReadOnly(true);

    width_property_ = new rviz_common::properties::IntProperty(
        "Width", 0, "Width of the map, in meters. (not editable)", this
    );
    width_property_->setReadOnly(true);

    height_property_ = new rviz_common::properties::IntProperty(
        "Height", 0, "Height of the map, in meters. (not editable)", this
    );
    height_property_->setReadOnly(true);

    position_property_ = new rviz_common::properties::VectorProperty(
        "Position",
        Ogre::Vector3::ZERO,
        "Position of the bottom left corner of the map, in meters. (not editable)",
        this
    );
    position_property_->setReadOnly(true);

    orientation_property_ = new rviz_common::properties::QuaternionProperty(
        "Orientation",
        Ogre::Quaternion::IDENTITY,
        "Orientation of the map. (not editable)",
        this
    );
    orientation_property_->setReadOnly(true);

    transform_timestamp_property_ = new rviz_common::properties::BoolProperty(
        "Use Timestamp",
        false,
        "Use map header timestamp when transforming",
        this,
        SLOT(transformMap())
    );
  }

  ProbGridDisplay::~ProbGridDisplay() { clear(); }

  void ProbGridDisplay::onInitialize() {
    MFDClass::onInitialize();
    palette_textures_.push_back(makePaletteTexture(makeMapPalette()));
    color_scheme_transparency_.push_back(false);
    palette_textures_.push_back(makePaletteTexture(makeCostmapPalette()));
    color_scheme_transparency_.push_back(true);
    palette_textures_.push_back(makePaletteTexture(makeRawPalette()));
    color_scheme_transparency_.push_back(true);
  }

  void ProbGridDisplay::updateTopic() { MFDClass::updateTopic(); }

  void ProbGridDisplay::processMessage(
      const prob_msgs::msg::ProbGrid::ConstSharedPtr msg
  ) {
    if (!msg) {
      return;
    }

    current_map_.header = msg->header;
    current_map_.info = msg->info;
    current_map_.data.assign(msg->data.size(), -1);

    for (size_t index = 0; index < msg->data.size(); ++index) {
      const float value = msg->data[index];
      if (!std::isfinite(value)) {
        current_map_.data[index] = -1;
        continue;
      }

      const float clamped_value = std::clamp(value, 0.0f, 1.0f);
      current_map_.data[index] =
          static_cast<int8_t>(std::lround(clamped_value * 100.0f));
    }

    loaded_ = true;
    Q_EMIT mapUpdated();
  }

  void ProbGridDisplay::showMap() {
    if (current_map_.data.empty()) {
      return;
    }

    if (!validateFloats(current_map_)) {
      setStatus(
          StatusProperty::Error,
          "Map",
          "Message contained invalid floating point values (nans or infs)"
      );
      return;
    }

    const size_t width = current_map_.info.width;
    const size_t height = current_map_.info.height;
    if (width * height == 0) {
      setStatus(StatusProperty::Error, "Map", "Map is zero-sized");
      return;
    }

    if (width * height != current_map_.data.size()) {
      setStatus(
          StatusProperty::Error, "Map", "Data size does not match width * height"
      );
      return;
    }

    setStatus(StatusProperty::Ok, "Message", "Confidence map received");
    showValidMap();
  }

  void ProbGridDisplay::showValidMap() {
    const size_t width = current_map_.info.width;
    const size_t height = current_map_.info.height;
    const float resolution = current_map_.info.resolution;

    resetSwatchesIfNecessary(width, height, resolution);

    frame_ = current_map_.header.frame_id;
    if (frame_.empty()) {
      frame_ = "/map";
    }

    updateSwatches();
    setStatus(StatusProperty::Ok, "Map", "Map OK");
    updatePalette();

    resolution_property_->setValue(resolution);
    width_property_->setValue(static_cast<unsigned int>(width));
    height_property_->setValue(static_cast<unsigned int>(height));
    position_property_->setVector(
        rviz_common::pointMsgToOgre(current_map_.info.origin.position)
    );
    orientation_property_->setQuaternion(
        rviz_common::quaternionMsgToOgre(current_map_.info.origin.orientation)
    );

    transformMap();
    updateDrawUnder();
    context_->queueRender();
  }

  void ProbGridDisplay::resetSwatchesIfNecessary(
      size_t width, size_t height, float resolution
  ) {
    if (width != width_ || height != height_ || resolution_ != resolution) {
      createSwatches();
      width_ = width;
      height_ = height;
      resolution_ = resolution;
    }
  }

  void ProbGridDisplay::createSwatches() {
    const size_t width = current_map_.info.width;
    const size_t height = current_map_.info.height;
    const float resolution = current_map_.info.resolution;

    size_t swatch_width = width;
    size_t swatch_height = height;
    int number_swatches = 1;
    const size_t maximum_number_swatch_splittings = 4;

    for (size_t split_count = 0; split_count < maximum_number_swatch_splittings;
         ++split_count) {
      swatches_.clear();
      try {
        tryCreateSwatches(
            width, height, resolution, swatch_width, swatch_height, number_swatches
        );
        updateDrawUnder();
        return;
      } catch (Ogre::InvalidParametersException&) {
        doubleSwatchNumber(swatch_width, swatch_height, number_swatches);
      } catch (Ogre::RenderingAPIException&) {
        doubleSwatchNumber(swatch_width, swatch_height, number_swatches);
      }
    }

    RVIZ_COMMON_LOG_ERROR_STREAM(
        "Failed to create confidence map display. Map is too large for RViz."
    );
    swatches_.clear();
  }

  void ProbGridDisplay::doubleSwatchNumber(
      size_t& swatch_width, size_t& swatch_height, int& number_swatches
  ) const {
    if (swatch_width > swatch_height) {
      swatch_width /= 2;
    } else {
      swatch_height /= 2;
    }
    number_swatches *= 2;
  }

  void ProbGridDisplay::tryCreateSwatches(
      size_t width, size_t height, float resolution, size_t swatch_width,
      size_t swatch_height, int number_swatches
  ) {
    size_t x = 0;
    size_t y = 0;
    for (int index = 0; index < number_swatches; ++index) {
      const size_t effective_width = getEffectiveDimension(width, swatch_width, x);
      const size_t effective_height =
          getEffectiveDimension(height, swatch_height, y);

      swatches_.push_back(
          std::make_shared<Swatch>(
              scene_manager_,
              scene_node_,
              x,
              y,
              effective_width,
              effective_height,
              resolution,
              draw_under_property_->getValue().toBool()
          )
      );
      swatches_.back()->updateData(current_map_);

      x += effective_width;
      if (x >= width) {
        x = 0;
        y += effective_height;
      }
    }
    updateAlpha();
  }

  size_t ProbGridDisplay::getEffectiveDimension(
      size_t map_dimension, size_t swatch_dimension, size_t position
  ) {
    return map_dimension - position - swatch_dimension >= swatch_dimension
               ? swatch_dimension
               : map_dimension - position;
  }

  void ProbGridDisplay::updateSwatches() const {
    for (const auto& swatch : swatches_) {
      swatch->updateData(current_map_);

      Ogre::Pass* pass = swatch->getTechniquePass();
      Ogre::TextureUnitState* texture_unit = nullptr;
      if (pass->getNumTextureUnitStates() > 0) {
        texture_unit = pass->getTextureUnitState(0);
      } else {
        texture_unit = pass->createTextureUnitState();
      }

      texture_unit->setTextureName(swatch->getTextureName());
      texture_unit->setTextureFiltering(Ogre::TFO_NONE);
      swatch->setVisible(true);
      swatch->resetOldTexture();
    }
  }

  void ProbGridDisplay::updatePalette() {
    const int palette_index = color_scheme_property_->getOptionInt();
    for (const auto& swatch : swatches_) {
      Ogre::Pass* pass = swatch->getTechniquePass();
      Ogre::TextureUnitState* palette_texture_unit = nullptr;
      if (pass->getNumTextureUnitStates() > 1) {
        palette_texture_unit = pass->getTextureUnitState(1);
      } else {
        palette_texture_unit = pass->createTextureUnitState();
      }
      palette_texture_unit->setTexture(palette_textures_[palette_index]);
      palette_texture_unit->setTextureFiltering(Ogre::TFO_NONE);
    }

    updateAlpha();
    updateDrawUnder();
  }

  void ProbGridDisplay::updateAlpha() {
    const float alpha = alpha_property_->getFloat();
    Ogre::SceneBlendType scene_blending;
    bool depth_write;
    rviz_rendering::MaterialManager::enableAlphaBlending(
        scene_blending, depth_write, alpha
    );

    for (const auto& swatch : swatches_) {
      swatch->updateAlpha(scene_blending, depth_write, alpha);
    }
  }

  void ProbGridDisplay::updateDrawUnder() const {
    const bool draw_under = draw_under_property_->getBool();
    if (alpha_property_->getFloat() >= rviz_rendering::unit_alpha_threshold) {
      for (const auto& swatch : swatches_) {
        swatch->setDepthWriteEnabled(!draw_under);
      }
    }

    const uint8_t group =
        draw_under ? Ogre::RENDER_QUEUE_4 : Ogre::RENDER_QUEUE_MAIN;
    for (const auto& swatch : swatches_) {
      swatch->setRenderQueueGroup(group);
    }
  }

  void ProbGridDisplay::transformMap() {
    if (!loaded_) {
      return;
    }

    rclcpp::Time transform_time = context_->getClock()->now();
    if (transform_timestamp_property_->getBool()) {
      transform_time = current_map_.header.stamp;
    }

    Ogre::Vector3 position;
    Ogre::Quaternion orientation;
    if (!context_->getFrameManager()->transform(
            frame_, transform_time, current_map_.info.origin, position, orientation
        ) &&
        !context_->getFrameManager()->transform(
            frame_,
            rclcpp::Time(0, 0, context_->getClock()->get_clock_type()),
            current_map_.info.origin,
            position,
            orientation
        )) {
      setMissingTransformToFixedFrame(frame_);
      scene_node_->setVisible(false);
    } else {
      setTransformOk();
      scene_node_->setPosition(position);
      scene_node_->setOrientation(orientation);
      scene_node_->setVisible(true);
    }
  }

  void ProbGridDisplay::fixedFrameChanged() { transformMap(); }

  void ProbGridDisplay::update(float wall_dt, float ros_dt) {
    (void)wall_dt;
    (void)ros_dt;
    transformMap();
  }

  void ProbGridDisplay::onEnable() {
    MFDClass::onEnable();
    setStatus(StatusProperty::Warn, "Message", "No confidence map received");
  }

  void ProbGridDisplay::clear() {
    if (isEnabled()) {
      setStatus(StatusProperty::Warn, "Message", "No confidence map received");
    }

    if (!loaded_) {
      return;
    }

    swatches_.clear();
    height_ = 0;
    width_ = 0;
    resolution_ = 0.0f;
    loaded_ = false;
  }

  void ProbGridDisplay::reset() {
    MFDClass::reset();
    clear();
  }

} // namespace prob_rviz_plugins

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(prob_rviz_plugins::ProbGridDisplay, rviz_common::Display)
