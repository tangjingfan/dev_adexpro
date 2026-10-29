#include "rpv_rviz_plugins/rpv_monitor_panel.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <QHBoxLayout>

namespace rpv_rviz_plugins
{

RPVMonitorPanel::RPVMonitorPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  QVBoxLayout * layout = new QVBoxLayout;
  QGridLayout * gridLayout = new QGridLayout;

  reachabilityTimeLabel_ = new QLabel("N/A");
  clusterFrontiersTimeLabel_ = new QLabel("N/A");
  distanceMatrixTimeLabel_ = new QLabel("N/A");
  localPlanTimeLabel_ = new QLabel("N/A");
  globalPlanTimeLabel_ = new QLabel("N/A");
  totalPlanningTimeLabel_ = new QLabel("N/A");
  
  replanNeededLabel_ = new QLabel("N/A");
  timeExceededLabel_ = new QLabel("N/A");
  pathDistanceExceededLabel_ = new QLabel("N/A");

  int row = 0;
  gridLayout->addWidget(new QLabel("<html><b>Planning Timings:</b></html>"), row++, 0, 1, 2);
  gridLayout->addWidget(new QLabel("Reachability Time:"), row, 0);
  gridLayout->addWidget(reachabilityTimeLabel_, row++, 1);
  
  gridLayout->addWidget(new QLabel("Cluster Frontiers Time:"), row, 0);
  gridLayout->addWidget(clusterFrontiersTimeLabel_, row++, 1);
  
  gridLayout->addWidget(new QLabel("Distance Matrix Time:"), row, 0);
  gridLayout->addWidget(distanceMatrixTimeLabel_, row++, 1);
  
  gridLayout->addWidget(new QLabel("Local Plan Time:"), row, 0);
  gridLayout->addWidget(localPlanTimeLabel_, row++, 1);
  
  gridLayout->addWidget(new QLabel("Global Plan Time:"), row, 0);
  gridLayout->addWidget(globalPlanTimeLabel_, row++, 1);
  
  gridLayout->addWidget(new QLabel("Total Planning Cycle Time:"), row, 0);
  gridLayout->addWidget(totalPlanningTimeLabel_, row++, 1);
  
  gridLayout->addWidget(new QLabel(""), row++, 0, 1, 2); // Spacer
  gridLayout->addWidget(new QLabel("<html><b>Global Planner State:</b></html>"), row++, 0, 1, 2);
  
  gridLayout->addWidget(new QLabel("Replan Needed:"), row, 0);
  gridLayout->addWidget(replanNeededLabel_, row++, 1);
  
  gridLayout->addWidget(new QLabel("Time Exceeded:"), row, 0);
  gridLayout->addWidget(timeExceededLabel_, row++, 1);
  
  gridLayout->addWidget(new QLabel("Path Distance Exceeded:"), row, 0);
  gridLayout->addWidget(pathDistanceExceededLabel_, row++, 1);
  
  layout->addLayout(gridLayout);
  layout->addStretch();
  setLayout(layout);

  uiTimer_ = new QTimer(this);
  connect(uiTimer_, SIGNAL(timeout()), this, SLOT(updateUi()));
  uiTimer_->start(100); // 10Hz UI update
}

RPVMonitorPanel::~RPVMonitorPanel() = default;

void RPVMonitorPanel::onInitialize()
{
  std::string nodeName = "rpv_control_panel_" + std::to_string(reinterpret_cast<uintptr_t>(this));
  node_ = std::make_shared<rclcpp::Node>(nodeName);

  statsSub_ = node_->create_subscription<rpv_msgs::msg::PlanningStats>(
    "/global_planner_stats", rclcpp::QoS(1),
    std::bind(&RPVMonitorPanel::statsCallback, this, std::placeholders::_1));

  // Use an executor in a separate thread if needed, or spin some in the UI thread.
  // Wait, rviz already runs spinning or the panel might not spin it for us.
  // Actually, easiest is to spin some in the timer. Let's do that in updateUi()
}

void RPVMonitorPanel::statsCallback(const rpv_msgs::msg::PlanningStats::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  latestStats_ = *msg;
  hasData_ = true;
}

void RPVMonitorPanel::updateUi()
{
  if (node_) {
    rclcpp::spin_some(node_);
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (!hasData_) return;

  reachabilityTimeLabel_->setText(QString::number(latestStats_.compute_reachability_time_ms, 'f', 2) + " ms");
  clusterFrontiersTimeLabel_->setText(QString::number(latestStats_.cluster_frontiers_time_ms, 'f', 2) + " ms");
  distanceMatrixTimeLabel_->setText(QString::number(latestStats_.build_distance_matrix_time_ms, 'f', 2) + " ms");
  localPlanTimeLabel_->setText(QString::number(latestStats_.local_plan_time_ms, 'f', 2) + " ms");
  globalPlanTimeLabel_->setText(QString::number(latestStats_.global_plan_time_ms, 'f', 2) + " ms");
  totalPlanningTimeLabel_->setText(QString::number(latestStats_.total_planning_time_ms, 'f', 2) + " ms");

  replanNeededLabel_->setText(latestStats_.replan_needed ? "Yes" : "No");

  QString timeStr = QString("%1 / %2 s").arg(latestStats_.time_exceeded_current, 0, 'f', 2).arg(latestStats_.time_exceeded_target, 0, 'f', 2);
  timeExceededLabel_->setText((latestStats_.time_exceeded ? "Yes" : "No") + QString(" (") + timeStr + ")");

  QString distStr = QString("%1 / %2 m").arg(latestStats_.path_distance_current, 0, 'f', 2).arg(latestStats_.path_distance_target, 0, 'f', 2);
  pathDistanceExceededLabel_->setText((latestStats_.path_distance_exceeded ? "Yes" : "No") + QString(" (") + distStr + ")");
}

void RPVMonitorPanel::save(rviz_common::Config config) const
{
  rviz_common::Panel::save(config);
}

void RPVMonitorPanel::load(const rviz_common::Config & config)
{
  rviz_common::Panel::load(config);
}

}  // namespace rpv_rviz_plugins

PLUGINLIB_EXPORT_CLASS(rpv_rviz_plugins::RPVMonitorPanel, rviz_common::Panel)