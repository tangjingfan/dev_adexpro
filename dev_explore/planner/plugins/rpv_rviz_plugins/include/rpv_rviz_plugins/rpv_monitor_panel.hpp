#ifndef RPV_RVIZ_PLUGINS__RPV_MONITOR_PANEL_HPP_
#define RPV_RVIZ_PLUGINS__RPV_MONITOR_PANEL_HPP_

#ifndef Q_MOC_RUN
#include <rviz_common/panel.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rpv_msgs/msg/planning_stats.hpp>
#endif

#include <QLabel>
#include <QTimer>
#include <QVBoxLayout>
#include <QGridLayout>
#include <mutex>

namespace rpv_rviz_plugins
{

class RPVMonitorPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit RPVMonitorPanel(QWidget * parent = nullptr);
  ~RPVMonitorPanel() override;

  void onInitialize() override;
  void load(const rviz_common::Config & config) override;
  void save(rviz_common::Config config) const override;

protected Q_SLOTS:
  void updateUi();

private:
  void statsCallback(const rpv_msgs::msg::PlanningStats::SharedPtr msg);

  // ROS
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<rpv_msgs::msg::PlanningStats>::SharedPtr statsSub_;
  std::mutex mutex_;
  rpv_msgs::msg::PlanningStats latestStats_;
  bool hasData_{false};

  // UI Elements
  QLabel* reachabilityTimeLabel_;
  QLabel* clusterFrontiersTimeLabel_;
  QLabel* distanceMatrixTimeLabel_;
  QLabel* localPlanTimeLabel_;
  QLabel* globalPlanTimeLabel_;
  QLabel* totalPlanningTimeLabel_;
  QLabel* replanNeededLabel_;
  QLabel* timeExceededLabel_;
  QLabel* pathDistanceExceededLabel_;

  QTimer* uiTimer_;
};

}  // namespace rpv_rviz_plugins

#endif  // RPV_RVIZ_PLUGINS__RPV_MONITOR_PANEL_HPP_