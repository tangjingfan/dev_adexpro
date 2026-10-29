#include "planner_common/random_sampler.h"

#include <iostream>

namespace {
// Helper: declare-or-get a parameter, returning whether it was supplied.
// ROS 2 forbids re-declaring; guard against that.
template <typename T>
bool getOrDeclare(rclcpp::Node* node, const std::string& name, T& out) {
  if (!node->has_parameter(name)) {
    rclcpp::ParameterValue default_value;
    if constexpr (std::is_same_v<T, std::string>) {
      default_value = rclcpp::ParameterValue(std::string());
    } else {
      default_value = rclcpp::ParameterValue(T{});
    }
    try {
      node->declare_parameter(name, default_value);
    } catch (const rclcpp::exceptions::ParameterAlreadyDeclaredException&) {
      // ignore
    }
  }
  rclcpp::Parameter p;
  if (!node->get_parameter(name, p)) return false;
  // get_parameter returns the declared default if the user did not set it,
  // so additionally check if the parameter was explicitly set via the
  // configuration. Here we simply read whatever is present.
  try {
    out = p.get_value<T>();
  } catch (const rclcpp::exceptions::InvalidParameterTypeException&) {
    return false;
  }
  return true;
}
}  // namespace

RandomSamplerBase::RandomSamplerBase() {
  mean_val_ = 0.0;
  std_val_ = 0.0;
  min_val_ = 0.0;
  max_val_ = 0.0;
}

RandomSamplerBase::~RandomSamplerBase() {}

bool RandomSamplerBase::loadParams(rclcpp::Node* node, const std::string& ns) {
  ROSPARAM_INFO("Loading: " + ns);
  std::string param_name;
  std::string parse_str;

  param_name = ns + ".pdf_type";
  getOrDeclare<std::string>(node, param_name, parse_str);
  if (parse_str == "kUniform")
    pdf_type_ = RandomDistributionType::kUniform;
  else if (parse_str == "kNormal")
    pdf_type_ = RandomDistributionType::kNormal;
  else if (parse_str == "kCauchy")
    pdf_type_ = RandomDistributionType::kCauchy;
  else if (parse_str == "kConst")
    pdf_type_ = RandomDistributionType::kConst;
  else {
    ROSPARAM_ERROR(ns);
    return false;
  }

  param_name = ns + ".sample_mode";
  getOrDeclare<std::string>(node, param_name, parse_str);
  if (parse_str == "kManual")
    sample_mode_ = SampleModeType::kManual;
  else if (parse_str == "kLocal")
    sample_mode_ = SampleModeType::kLocal;
  else if (parse_str == "kGlobal")
    sample_mode_ = SampleModeType::kGlobal;
  else if (parse_str == "kExploredMap")
    sample_mode_ = SampleModeType::kExploredMap;
  else {
    ROSPARAM_ERROR(ns);
    return false;
  }

  if (sample_mode_ == SampleModeType::kManual) {
    if (pdf_type_ == RandomDistributionType::kUniform) {
      param_name = ns + ".min_val";
      if (!getOrDeclare<double>(node, param_name, min_val_)) {
        ROSPARAM_ERROR(param_name);
        return false;
      }
      param_name = ns + ".max_val";
      if (!getOrDeclare<double>(node, param_name, max_val_)) {
        ROSPARAM_ERROR(param_name);
        return false;
      }
    } else if (pdf_type_ == RandomDistributionType::kNormal) {
      use_bound_ = true;
      param_name = ns + ".min_val";
      if (!getOrDeclare<double>(node, param_name, min_val_)) {
        ROSPARAM_WARN(param_name, "not used");
        use_bound_ = false;
      }
      param_name = ns + ".max_val";
      if (!getOrDeclare<double>(node, param_name, max_val_)) {
        ROSPARAM_WARN(param_name, "not used");
        use_bound_ = false;
      }
      param_name = ns + ".mean_val";
      if (!getOrDeclare<double>(node, param_name, mean_val_)) {
        ROSPARAM_ERROR(param_name);
        return false;
      }
      param_name = ns + ".std_val";
      if (!getOrDeclare<double>(node, param_name, std_val_)) {
        ROSPARAM_ERROR(param_name);
        return false;
      }
    } else if (pdf_type_ == RandomDistributionType::kConst) {
      param_name = ns + ".const_val";
      if (!getOrDeclare<double>(node, param_name, const_val_)) {
        ROSPARAM_ERROR(param_name);
        return false;
      }
    }
  }
  ROSPARAM_INFO("Done.");
  return true;
}

void RandomSamplerBase::setParams(int ind, BoundedSpaceParams& global_params,
                                  BoundedSpaceParams& local_params) {
  BoundedSpaceParams* params;
  if (SampleModeType::kGlobal == sample_mode_)
    params = &global_params;
  else if (sample_mode_ == SampleModeType::kLocal)
    params = &local_params;
  else
    return;

  if (pdf_type_ == RandomDistributionType::kUniform) {
    if (params->type == BoundedSpaceType::kCuboid) {
      min_val_ = params->min_val[ind];
      max_val_ = params->max_val[ind];
    } else if (params->type == BoundedSpaceType::kSphere) {
      min_val_ = -params->radius;
      max_val_ = params->radius;
    }
  }
}

void RandomSamplerBase::setBound(double min_val, double max_val) {
  min_val_ = min_val;
  max_val_ = max_val;
}

void RandomSamplerBase::setDistributionParams(double mean_val, double std_val) {
  mean_val_ = mean_val;
  std_val_ = std_val;
}

void RandomSamplerBase::reset() {
  std::random_device rd;
  generator_.seed(rd());
  if (pdf_type_ == RandomDistributionType::kUniform) {
    uniform_pdf_.reset(
        new std::uniform_real_distribution<>(min_val_, max_val_));
  } else if (pdf_type_ == RandomDistributionType::kNormal) {
    normal_pdf_.reset(new std::normal_distribution<>(mean_val_, std_val_));
  } else if (pdf_type_ == RandomDistributionType::kCauchy) {
    normal_pdf_.reset(new std::normal_distribution<>(0, std_val_));
  }
}

double RandomSamplerBase::generate(double current_val, double chi_sq_var) {
  double r;
  if (pdf_type_ == RandomDistributionType::kUniform) {
    r = (*uniform_pdf_)(generator_);
  } else if (pdf_type_ == RandomDistributionType::kNormal) {
    r = normal_pdf_->operator()(generator_);
  } else if (pdf_type_ == RandomDistributionType::kCauchy) {
    r = normal_pdf_->operator()(generator_) / std::sqrt(chi_sq_var) + mean_val_;
  } else if (pdf_type_ == RandomDistributionType::kConst) {
    r = const_val_;
  } else {
    if (global_verbosity >= Verbosity::ERROR) {
      std::cout << "[RandomSamplerBase] pdf_type not supported." << std::endl;
    }
    return 0.0;
  }

  if (sample_mode_ != SampleModeType::kGlobal) r += current_val;
  return r;
}

double RandomSamplerBase::getZOffset() {
  if (pdf_type_ == RandomDistributionType::kConst) {
    return const_val_;
  }
  return 0.0;
}

RandomSampler::RandomSampler() {
  random_sampler_base_.clear();
  random_sampler_base_.resize(4);
  valid_samples_.clear();
}

RandomSampler::~RandomSampler() {}

bool RandomSampler::loadParams(rclcpp::Node* node, const std::string& ns) {
  if (!random_sampler_base_[0].loadParams(node, ns + ".X")) return false;
  if (!random_sampler_base_[1].loadParams(node, ns + ".Y")) return false;
  if (!random_sampler_base_[2].loadParams(node, ns + ".Z")) return false;
  if (!random_sampler_base_[3].loadParams(node, ns + ".Heading")) return false;
  isReady = true;
  return true;
}

bool RandomSampler::setBound(Eigen::Vector3d& min_val,
                             Eigen::Vector3d& max_val) {
  for (int i = 0; i < 3; ++i) {
    random_sampler_base_[i].setBound(min_val[i], max_val[i]);
  }
  reset();
  return true;
}

bool RandomSampler::setRotation(Eigen::Vector3d& rotations) {
  Eigen::Matrix3d rot_W2B;
  rot_W2B = Eigen::AngleAxisd(rotations[0], Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(rotations[1], Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(rotations[2], Eigen::Vector3d::UnitX());
  rot_B2W = rot_W2B.inverse();
  return true;
}

bool RandomSampler::setDistributionParams(Eigen::Vector3d& mean_val,
                                          Eigen::Vector3d& std_val) {
  for (int i = 0; i < 3; ++i) {
    random_sampler_base_[i].setDistributionParams(mean_val[i], std_val[i]);
  }
  reset();
  return true;
}

bool RandomSampler::setParams(BoundedSpaceParams& global_params,
                              BoundedSpaceParams& local_params, bool rotation) {
  for (int i = 0; i < 3; ++i) {
    random_sampler_base_[i].setParams(i, global_params, local_params);
  }
  if (rotation) {
    Eigen::Matrix3d rot_W2B;
    rot_W2B =
        Eigen::AngleAxisd(local_params.rotations[0], Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(local_params.rotations[1], Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(local_params.rotations[2], Eigen::Vector3d::UnitX());
    rot_B2W = rot_W2B.inverse();
  } else {
    rot_B2W = Eigen::Matrix3d::Identity();
  }

  reset();
  return true;
}

void RandomSampler::reset() {
  std::random_device rd;
  generator_.seed(rd());
  chi_squared_.reset(new std::chi_squared_distribution<>(1));
  for (int i = 0; i < 4; ++i) {
    random_sampler_base_[i].reset();
  }
  valid_samples_.clear();
  invalid_samples_.clear();
}

void RandomSampler::generate(StateVec& current_state, StateVec& sample_state) {
  double r = chi_squared_->operator()(generator_);
  for (int i = 0; i < 4; ++i) {
    sample_state[i] = random_sampler_base_[i].generate(0, r);
  }
  Eigen::Vector3d sample_xyz = rot_B2W * sample_state.head(3);
  sample_state[0] = sample_xyz[0] + current_state[0];
  sample_state[1] = sample_xyz[1] + current_state[1];
  sample_state[2] = sample_xyz[2] + current_state[2];
  sample_state[3] = sample_state[3] + current_state[3];
}

void RandomSampler::pushSample(StateVec& state, bool valid) {
  if (valid) {
    if (static_cast<int>(valid_samples_.size()) < kValidSamplesMaxSize)
      valid_samples_.emplace_back(state);
  } else {
    if (static_cast<int>(invalid_samples_.size()) < kInvalidSamplesMaxSize)
      invalid_samples_.emplace_back(state);
  }
}
