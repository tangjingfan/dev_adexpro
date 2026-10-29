#ifndef PROB_RVIZ_PLUGINS__PREDICTION_PANEL_HPP_
#define PROB_RVIZ_PLUGINS__PREDICTION_PANEL_HPP_

#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <QImage>

#include "nav_msgs/msg/occupancy_grid.hpp"
#include "prob_msgs/msg/prob_grid.hpp"
#include "rclcpp/executors/single_threaded_executor.hpp"
#include "rclcpp/node.hpp"
#include "rviz_common/panel.hpp"

class QLabel;
class QLineEdit;
class QPushButton;

namespace prob_rviz_plugins
{

class PredictionPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit PredictionPanel(QWidget * parent = nullptr);
  ~PredictionPanel() override;

  void save(rviz_common::Config config) const override;
  void load(const rviz_common::Config & config) override;

Q_SIGNALS:
  void predictedMapUpdated();
  void confidenceMapUpdated();

private Q_SLOTS:
  void applyTopics();
  void renderPredictedMap();
  void renderConfidenceMap();

private:
  void setupUi();
  void startRosThread();
  void stopRosThread();
  void refreshSubscriptions();
  void subscribePredictedMap();
  void subscribeConfidenceMap();
  void updateStatusLabel(QLabel * label, const QString & text);
  void updateImageLabel(QLabel * label, const QImage & image);

  void onPredictedMap(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
  void onConfidenceMap(const prob_msgs::msg::ProbGrid::SharedPtr msg);

  QImage buildPredictedMapImage(const nav_msgs::msg::OccupancyGrid & map) const;
  QImage buildConfidenceMapImage(const prob_msgs::msg::ProbGrid & map) const;

  std::shared_ptr<rclcpp::Node> node_{};
  rclcpp::executors::SingleThreadedExecutor::SharedPtr executor_{};
  std::thread rosThread_{};

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr predictedMapSubscription_{};
  rclcpp::Subscription<prob_msgs::msg::ProbGrid>::SharedPtr confidenceMapSubscription_{};

  std::mutex imageMutex_{};
  QImage predictedMapImage_{};
  QImage confidenceMapImage_{};

  std::string predictedMapTopic_{"/predicted_map"};
  std::string confidenceMapTopic_{"/confidence_map"};

  QLineEdit * predictedMapTopicEdit_{nullptr};
  QLineEdit * confidenceMapTopicEdit_{nullptr};
  QPushButton * applyButton_{nullptr};
  QLabel * predictedMapStatusLabel_{nullptr};
  QLabel * confidenceMapStatusLabel_{nullptr};
  QLabel * predictedMapLabel_{nullptr};
  QLabel * confidenceMapLabel_{nullptr};
};

}  // namespace prob_rviz_plugins

#endif  // PROB_RVIZ_PLUGINS__PREDICTION_PANEL_HPP_
