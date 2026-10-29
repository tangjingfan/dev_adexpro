#include "prob_rviz_plugins/prediction_panel.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include <QFormLayout>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPixmap>
#include <QPushButton>
#include <QVBoxLayout>

namespace prob_rviz_plugins {

  namespace {
    constexpr int kQueueSize = 1;
    constexpr int kMinimumImageSize = 240;

    QRgb occupancyValueToColor(const int8_t value) {
      if (value < 0) {
        return qRgb(110, 110, 110);
      }
      if (value >= 50) {
        return qRgb(45, 25, 60);
      }
      return qRgb(240, 240, 240);
    }

    QRgb confidenceValueToColor(const float value) {
      if (!std::isfinite(value)) {
        return qRgb(80, 80, 80);
      }

      const float clampedValue = std::clamp(value, 0.0f, 1.0f);
      const int red = static_cast<int>(255.0f * clampedValue);
      const int green =
          static_cast<int>(255.0f * (1.0f - std::fabs(clampedValue - 0.5f) * 2.0f));
      const int blue = static_cast<int>(255.0f * (1.0f - clampedValue));
      return qRgb(red, green, blue);
    }
  } // namespace

  PredictionPanel::PredictionPanel(QWidget* parent) : rviz_common::Panel(parent) {
    setupUi();

    connect(
        this,
        &PredictionPanel::predictedMapUpdated,
        this,
        &PredictionPanel::renderPredictedMap
    );
    connect(
        this,
        &PredictionPanel::confidenceMapUpdated,
        this,
        &PredictionPanel::renderConfidenceMap
    );
    connect(
        applyButton_, &QPushButton::clicked, this, &PredictionPanel::applyTopics
    );

    std::string nodeName = "prediction_panel_" + std::to_string(reinterpret_cast<uintptr_t>(this));
    node_ = std::make_shared<rclcpp::Node>(nodeName);
    executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    executor_->add_node(node_);

    refreshSubscriptions();
    startRosThread();
  }

  PredictionPanel::~PredictionPanel() { stopRosThread(); }

  void PredictionPanel::save(rviz_common::Config config) const {
    rviz_common::Panel::save(config);
    config.mapSetValue(
        "predicted_map_topic", QString::fromStdString(predictedMapTopic_)
    );
    config.mapSetValue(
        "confidence_map_topic", QString::fromStdString(confidenceMapTopic_)
    );
  }

  void PredictionPanel::load(const rviz_common::Config& config) {
    rviz_common::Panel::load(config);

    QString predictedMapTopic;
    if (config.mapGetString("predicted_map_topic", &predictedMapTopic)) {
      predictedMapTopic_ = predictedMapTopic.toStdString();
      predictedMapTopicEdit_->setText(predictedMapTopic);
    }

    QString confidenceMapTopic;
    if (config.mapGetString("confidence_map_topic", &confidenceMapTopic)) {
      confidenceMapTopic_ = confidenceMapTopic.toStdString();
      confidenceMapTopicEdit_->setText(confidenceMapTopic);
    }

    refreshSubscriptions();
  }

  void PredictionPanel::applyTopics() {
    predictedMapTopic_ = predictedMapTopicEdit_->text().trimmed().toStdString();
    confidenceMapTopic_ = confidenceMapTopicEdit_->text().trimmed().toStdString();
    refreshSubscriptions();
  }

  void PredictionPanel::renderPredictedMap() {
    std::lock_guard<std::mutex> lock(imageMutex_);
    updateImageLabel(predictedMapLabel_, predictedMapImage_);
    updateStatusLabel(predictedMapStatusLabel_, "Receiving predicted map");
  }

  void PredictionPanel::renderConfidenceMap() {
    std::lock_guard<std::mutex> lock(imageMutex_);
    updateImageLabel(confidenceMapLabel_, confidenceMapImage_);
    updateStatusLabel(confidenceMapStatusLabel_, "Receiving confidence map");
  }

  void PredictionPanel::setupUi() {
    auto* rootLayout = new QVBoxLayout(this);

    auto* topicLayout = new QFormLayout();
    predictedMapTopicEdit_ =
        new QLineEdit(QString::fromStdString(predictedMapTopic_), this);
    confidenceMapTopicEdit_ =
        new QLineEdit(QString::fromStdString(confidenceMapTopic_), this);
    topicLayout->addRow("Predicted Map", predictedMapTopicEdit_);
    topicLayout->addRow("Confidence Map", confidenceMapTopicEdit_);
    rootLayout->addLayout(topicLayout);

    applyButton_ = new QPushButton("Apply Topics", this);
    rootLayout->addWidget(applyButton_);

    auto* imageLayout = new QHBoxLayout();

    auto* predictedFrame = new QFrame(this);
    predictedFrame->setFrameShape(QFrame::StyledPanel);
    auto* predictedLayout = new QVBoxLayout(predictedFrame);
    predictedLayout->addWidget(new QLabel("Predicted Map", predictedFrame));
    predictedMapStatusLabel_ =
        new QLabel("Waiting for predicted map", predictedFrame);
    predictedLayout->addWidget(predictedMapStatusLabel_);
    predictedMapLabel_ = new QLabel(predictedFrame);
    predictedMapLabel_->setMinimumSize(kMinimumImageSize, kMinimumImageSize);
    predictedMapLabel_->setAlignment(Qt::AlignCenter);
    predictedLayout->addWidget(predictedMapLabel_);

    auto* confidenceFrame = new QFrame(this);
    confidenceFrame->setFrameShape(QFrame::StyledPanel);
    auto* confidenceLayout = new QVBoxLayout(confidenceFrame);
    confidenceLayout->addWidget(new QLabel("Confidence Map", confidenceFrame));
    confidenceMapStatusLabel_ =
        new QLabel("Waiting for confidence map", confidenceFrame);
    confidenceLayout->addWidget(confidenceMapStatusLabel_);
    confidenceMapLabel_ = new QLabel(confidenceFrame);
    confidenceMapLabel_->setMinimumSize(kMinimumImageSize, kMinimumImageSize);
    confidenceMapLabel_->setAlignment(Qt::AlignCenter);
    confidenceLayout->addWidget(confidenceMapLabel_);

    imageLayout->addWidget(predictedFrame);
    imageLayout->addWidget(confidenceFrame);
    rootLayout->addLayout(imageLayout);
    setLayout(rootLayout);
  }

  void PredictionPanel::startRosThread() {
    rosThread_ = std::thread([this]() { executor_->spin(); });
  }

  void PredictionPanel::stopRosThread() {
    if (executor_) {
      executor_->cancel();
    }

    if (rosThread_.joinable()) {
      rosThread_.join();
    }

    if (executor_ && node_) {
      executor_->remove_node(node_);
    }
  }

  void PredictionPanel::refreshSubscriptions() {
    subscribePredictedMap();
    subscribeConfidenceMap();
    updateStatusLabel(predictedMapStatusLabel_, "Waiting for predicted map");
    updateStatusLabel(confidenceMapStatusLabel_, "Waiting for confidence map");
  }

  void PredictionPanel::subscribePredictedMap() {
    if (predictedMapTopic_.empty()) {
      predictedMapSubscription_.reset();
      return;
    }

    predictedMapSubscription_ =
        node_->create_subscription<nav_msgs::msg::OccupancyGrid>(
            predictedMapTopic_,
            kQueueSize,
            [this](nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
              onPredictedMap(std::move(msg));
            }
        );
  }

  void PredictionPanel::subscribeConfidenceMap() {
    if (confidenceMapTopic_.empty()) {
      confidenceMapSubscription_.reset();
      return;
    }

    confidenceMapSubscription_ =
        node_->create_subscription<prob_msgs::msg::ProbGrid>(
            confidenceMapTopic_,
            kQueueSize,
            [this](prob_msgs::msg::ProbGrid::SharedPtr msg) {
              onConfidenceMap(std::move(msg));
            }
        );
  }

  void PredictionPanel::updateStatusLabel(QLabel* label, const QString& text) {
    if (label != nullptr) {
      label->setText(text);
    }
  }

  void PredictionPanel::updateImageLabel(QLabel* label, const QImage& image) {
    if (label == nullptr) {
      return;
    }

    if (image.isNull()) {
      label->clear();
      return;
    }

    const QPixmap pixmap = QPixmap::fromImage(image);
    label->setPixmap(
        pixmap.scaled(label->size(), Qt::KeepAspectRatio, Qt::FastTransformation)
    );
  }

  void PredictionPanel::onPredictedMap(
      const nav_msgs::msg::OccupancyGrid::SharedPtr msg
  ) {
    if (msg == nullptr) {
      return;
    }

    {
      std::lock_guard<std::mutex> lock(imageMutex_);
      predictedMapImage_ = buildPredictedMapImage(*msg);
    }
    Q_EMIT predictedMapUpdated();
  }

  void
  PredictionPanel::onConfidenceMap(const prob_msgs::msg::ProbGrid::SharedPtr msg) {
    if (msg == nullptr) {
      return;
    }

    {
      std::lock_guard<std::mutex> lock(imageMutex_);
      confidenceMapImage_ = buildConfidenceMapImage(*msg);
    }
    Q_EMIT confidenceMapUpdated();
  }

  QImage PredictionPanel::buildPredictedMapImage(
      const nav_msgs::msg::OccupancyGrid& map
  ) const {
    const int width = static_cast<int>(map.info.width);
    const int height = static_cast<int>(map.info.height);
    if (width <= 0 || height <= 0 ||
        static_cast<size_t>(width * height) != map.data.size()) {
      return {};
    }

    QImage image(width, height, QImage::Format_RGB32);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int sourceIndex = (height - 1 - y) * width + x;
        image.setPixel(x, y, occupancyValueToColor(map.data[sourceIndex]));
      }
    }
    return image;
  }

  QImage PredictionPanel::buildConfidenceMapImage(
      const prob_msgs::msg::ProbGrid& map
  ) const {
    const int width = static_cast<int>(map.info.width);
    const int height = static_cast<int>(map.info.height);
    if (width <= 0 || height <= 0 ||
        static_cast<size_t>(width * height) != map.data.size()) {
      return {};
    }

    QImage image(width, height, QImage::Format_RGB32);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int sourceIndex = (height - 1 - y) * width + x;
        image.setPixel(x, y, confidenceValueToColor(map.data[sourceIndex]));
      }
    }
    return image;
  }

} // namespace prob_rviz_plugins

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(prob_rviz_plugins::PredictionPanel, rviz_common::Panel)
