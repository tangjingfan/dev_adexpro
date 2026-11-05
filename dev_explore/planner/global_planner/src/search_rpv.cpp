#include "../include/search_rpv.hpp"
#include "../include/debug.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace raplab {

namespace {

// How often the wall-clock limit is polled. SimpleTimer wraps std::clock(),
// which is ~178ns per call (CLOCK_PROCESS_CPUTIME_ID, not in the vDSO), so
// polling every iteration cost more than the check is worth. At roughly a
// microsecond per iteration this overshoots a deadline by ~1ms at most.
constexpr long kTimeCheckStride = 1024;

} // namespace

template <class Policy, class Store, bool IOG>
void RPVStar<Policy, Store, IOG>::SetGraph(PlannerGraph *g) {
  n_ = static_cast<int>(g->NumVertex());
  if (n_ > kRPVMaxVertices) {
    throw std::invalid_argument(
        "RPV search supports at most 64 vertices (visited sets are 64-bit "
        "masks); a 2^n-state search is intractable well before that anyway");
  }
  full_mask_ = RPVFullMask(n_);

  // Snapshot adjacency and edge costs once. The base graph API returns
  // vector copies on every call; the inner search loop is hot enough that
  // this single copy pays for itself many times over.
  adj_.assign(n_, {});
  adj_cost_.assign(n_, {});
  cost_matrix_.assign(static_cast<size_t>(n_) * n_, 0.0);
  for (int u = 0; u < n_; ++u) {
    adj_[u] = g->GetSuccs(u);
    auto cv = g->GetSuccCosts(u);
    adj_cost_[u].resize(cv.size());
    for (size_t i = 0; i < cv.size(); ++i) {
      const double c = cv[i][0];
      adj_cost_[u][i] = c;
      cost_matrix_[static_cast<size_t>(u) * n_ + adj_[u][i]] = c;
    }
  }

  states_.Reset(n_, RPVInitialStateCapacity(n_));
  labels_.clear();
  labels_.reserve(1 << 14);
  parent_.clear();
  parent_.reserve(1 << 14);
}

template <class Policy, class Store, bool IOG>
double RPVStar<Policy, Store, IOG>::RunGreedy(long start,
                                  std::vector<long> &path_out) const {
  path_out.clear();
  std::vector<bool> visited(n_, false);
  long curr = start;
  visited[curr] = true;
  path_out.push_back(curr);
  double current_q = 0.0;
  double total = 0.0;

  for (int i = 0; i < n_ - 1; ++i) {
    double min_dist = std::numeric_limits<double>::infinity();
    long next = -1;
    const auto &succs = adj_[curr];
    const auto &costs = adj_cost_[curr];
    for (size_t j = 0; j < succs.size(); ++j) {
      if (visited[succs[j]]) continue;
      if (costs[j] < min_dist) {
        min_dist = costs[j];
        next = succs[j];
      }
    }
    if (next == -1) {
      path_out.clear();
      return std::numeric_limits<double>::infinity();
    }
    const double new_q = current_q + min_dist;
    total = Policy::accumulate(total, prob_[next], new_q);
    current_q = new_q;
    curr = next;
    visited[curr] = true;
    path_out.push_back(curr);
  }
  return total;
}

template <class Policy, class Store, bool IOG>
RPVLabel RPVStar<Policy, Store, IOG>::CreateRootLabel(long vo) {
  double p_total = 0.0;
  for (int i = 0; i < n_; ++i) p_total += prob_[i];

  RPVLabel l0(0, vo, 1ull << vo, 1, 0.0, 0.0, p_total - prob_[vo]);
  if constexpr (IOG) Admit(l0);
  l0.h = Heuristic(l0);
  l0.f = Policy::f_value(l0.g, l0.h);

  labels_.push_back(l0);
  parent_.push_back(-1);
  res_.n_generated++;
  return l0;
}

template <class Policy, class Store, bool IOG>
long RPVStar<Policy, Store, IOG>::Search(long vo, PlannerGraph *g, double time_limit,
                             std::vector<double> &prob_vec) {
  SetGraph(g);
  SetProb(prob_vec);
  time_limit_ = time_limit;
  open_heap_.clear();
  open_heap_.reserve(1 << 14);

  SimpleTimer timer;
  timer.Start();

  // Greedy nearest-neighbor tour as an initial upper bound. Labels whose
  // f exceeds best_cost cannot improve on this UB (f is an admissible LB
  // on the completed-tour cost), so they get pruned at three points:
  // before the label is created (via new_g), after Heuristic (via
  // l_prime.f), and at pop (via l.f). Strict `>` so greedy-optimal ties
  // survive.
  std::vector<long> greedy_path;
  double best_cost = RunGreedy(vo, greedy_path);
  if (std::isfinite(best_cost)) {
    res_.path = greedy_path;
    res_.final_cost = {best_cost};
    res_.rt_greedy = timer.GetDurationSecond();
  } else {
    best_cost = std::numeric_limits<double>::infinity();
  }

  const RPVLabel l0 = CreateRootLabel(vo);
  open_heap_.emplace_back(l0.f, l0.label_id);
  std::push_heap(open_heap_.begin(), open_heap_.end(),
                 std::greater<std::pair<double, long>>());

  long iter_count = 0;
  while (!open_heap_.empty()) {
    if ((++iter_count % kTimeCheckStride) == 0 &&
        timer.GetDurationSecond() > time_limit_) {
      std::cout << "[INFO] RPVStar<" << Policy::name()
                << ">::Search timeout!" << std::endl;
      res_.timeout = true;
      break;
    }

    // Copy label out before any push_back can invalidate references.
    const long top_id = open_heap_.front().second;
    std::pop_heap(open_heap_.begin(), open_heap_.end(),
                  std::greater<std::pair<double, long>>());
    open_heap_.pop_back();
    RPVLabel l = labels_[top_id];

    if (l.f > best_cost) continue;
    if (IOG ? !StillLive(l) : Dominated(l)) continue;

    if (l.visited_set == full_mask_) {
      res_.final_label_id = l.label_id;
      res_.final_cost = {l.g};
      res_.rt_search = timer.GetDurationSecond();
      break;
    }

    if constexpr (!IOG) Admit(l);
    res_.n_expanded++;

    const auto &succs = adj_[l.current_vertex];
    const auto &costs = adj_cost_[l.current_vertex];
    for (size_t i = 0; i < succs.size(); ++i) {
      const long v = succs[i];
      if ((l.visited_set >> v) & 1ull) continue;
      const double new_q = l.q + costs[i];
      const double new_g =
          Policy::step(l.g, prob_[v], new_q, costs[i], l.p_rem);

      if (new_g > best_cost) continue;

      const long new_id = static_cast<long>(labels_.size());
      RPVLabel l_prime(new_id, v, l.visited_set | (1ull << v),
                       l.visited_count + 1, new_g, new_q,
                       l.p_rem - prob_[v]);
      if (Dominated(l_prime)) continue;

      l_prime.h = Heuristic(l_prime);
      l_prime.f = Policy::f_value(l_prime.g, l_prime.h);

      // Admit only after the f-bound: a label the bound rejects would
      // otherwise occupy a table slot it can never use.
      if (l_prime.f > best_cost) continue;
      if constexpr (IOG) Admit(l_prime);

      const double f = l_prime.f;
      labels_.push_back(std::move(l_prime));
      parent_.push_back(l.label_id);
      res_.n_generated++;
      open_heap_.emplace_back(f, new_id);
      std::push_heap(open_heap_.begin(), open_heap_.end(),
                     std::greater<std::pair<double, long>>());
    }
  }

  if (res_.timeout) return -1;
  FinalizeLabelStats();
  return res_.final_label_id;
}

template <class Policy, class Store, bool IOG>
long RPVStar<Policy, Store, IOG>::FocalSearch(long vo, PlannerGraph *g, double time_limit,
                                  std::vector<double> &prob_vec) {
  SetGraph(g);
  SetProb(prob_vec);
  time_limit_ = time_limit;

  SimpleTimer timer;
  timer.Start();

  std::vector<long> greedy_path;
  double greedy_cost = RunGreedy(vo, greedy_path);
  double best_cost = greedy_cost;

  const RPVLabel l0 = CreateRootLabel(vo);
  open_.insert({l0.f, l0.label_id});
  focal_.insert({l0.visited_count, l0.label_id});

  double current_f_min = open_.begin()->first;

  if (best_cost >= std::numeric_limits<double>::max()) {
    eps_ = std::numeric_limits<double>::infinity();
  } else {
    eps_ = (best_cost / current_f_min) - 1.0;
    if (eps_ < 0) eps_ = 0;
    res_.path = greedy_path;
    res_.final_cost = {greedy_cost};
    res_.rt_greedy = timer.GetDurationSecond();
    res_.rt_search = res_.rt_greedy;
    res_.rt_optimization.push_back(res_.rt_greedy);
    res_.cost_optimization.push_back(greedy_cost);
    res_.eps_optimization.push_back(eps_);
  }

  long focal_iter = 0;
  while (!open_.empty()) {
    if ((++focal_iter % kTimeCheckStride) == 0 &&
        timer.GetDurationSecond() > time_limit_) {
      std::cout << "[INFO] RPVStar<" << Policy::name()
                << ">::FocalSearch timeout!" << std::endl;
      res_.timeout = true;
      break;
    }

    const double new_f_min = open_.begin()->first;
    if (best_cost < std::numeric_limits<double>::max() &&
        new_f_min >= best_cost) {
      break;
    }
    if (best_cost < std::numeric_limits<double>::max()) {
      double dyn = (best_cost / new_f_min) - 1.0;
      eps_ = dyn < 0 ? 0 : dyn;
      // Bound-only checkpoint: incumbent unchanged but the open lower bound
      // tightened. Record when eps drops by a meaningful relative amount so
      // omega(t) reflects post-incumbent bound improvements (and can reach 1
      // when the search proves the incumbent optimal).
      if (!res_.eps_optimization.empty() &&
          eps_ < res_.eps_optimization.back() * 0.95) {
        res_.rt_optimization.push_back(timer.GetDurationSecond());
        res_.cost_optimization.push_back(best_cost);
        res_.eps_optimization.push_back(eps_);
      }
    }

    if (std::abs(new_f_min - current_f_min) > 1e-6) {
      UpdateFocalList(current_f_min, new_f_min);
      current_f_min = new_f_min;
    }

    // open_ is non-empty here (loop condition; nothing above removes from it).
    if (focal_.empty()) {
      long best_id = open_.begin()->second;
      focal_.insert({labels_[best_id].visited_count, best_id});
    }

    // Pick the focal element with greatest visited_count, then greatest f for
    // tie-break (preferring deeper labels).
    RPVLabel l = labels_[focal_.rbegin()->second];
    open_.erase({l.f, l.label_id});
    focal_.erase(std::prev(focal_.end()));

    if (IOG ? !StillLive(l) : Dominated(l)) continue;
    if (l.f >= best_cost) continue;

    if (l.visited_set == full_mask_) {
      if (l.g < best_cost) {
        best_cost = l.g;
        res_.final_label_id = l.label_id;
        res_.final_cost = {l.g};
        res_.rt_search = timer.GetDurationSecond();
        res_.rt_optimization.push_back(res_.rt_search);
        res_.cost_optimization.push_back(l.g);
        // Reported eps reflects the NEW incumbent against the current open
        // lower bound, so omega = 1 when the incumbent is provably optimal.
        double reported_eps = 0.0;
        if (!open_.empty()) {
          const double lb = open_.begin()->first;
          if (lb > 0.0) {
            reported_eps = (best_cost / lb) - 1.0;
            if (reported_eps < 0.0) reported_eps = 0.0;
          }
        }
        res_.eps_optimization.push_back(reported_eps);
      }
      if (open_.empty()) break;
      double next_f_min = open_.begin()->first;
      double new_eps = ((best_cost / next_f_min) - 1.0) * 0.8;
      if (new_eps < 0) new_eps = 0;
      if (std::abs(new_eps - eps_) < 0.01 && eps_ < 0.01) break;
      eps_ = new_eps;
      PruneAndRebuildFocal(best_cost);
      if (open_.empty()) break;
      current_f_min = open_.begin()->first;
      continue;
    }

    if constexpr (!IOG) Admit(l);
    res_.n_expanded++;

    const auto &succs = adj_[l.current_vertex];
    const auto &costs = adj_cost_[l.current_vertex];
    const double focal_bound = (1.0 + eps_) * current_f_min;

    for (size_t i = 0; i < succs.size(); ++i) {
      const long v = succs[i];
      if ((l.visited_set >> v) & 1ull) continue;
      const double new_q = l.q + costs[i];
      const double new_g =
          Policy::step(l.g, prob_[v], new_q, costs[i], l.p_rem);

      if (new_g >= best_cost) continue;

      const long new_id = static_cast<long>(labels_.size());
      RPVLabel l_prime(new_id, v, l.visited_set | (1ull << v),
                       l.visited_count + 1, new_g, new_q,
                       l.p_rem - prob_[v]);
      if (Dominated(l_prime)) continue;

      l_prime.h = Heuristic(l_prime);
      l_prime.f = Policy::f_value(l_prime.g, l_prime.h);

      // Admit only after the f-bound: a label the bound rejects would
      // otherwise occupy a table slot it can never use.
      if (l_prime.f >= best_cost) continue;
      if constexpr (IOG) Admit(l_prime);

      const double f = l_prime.f;
      const long vcount = l_prime.visited_count;
      labels_.push_back(std::move(l_prime));
      parent_.push_back(l.label_id);
      res_.n_generated++;
      open_.insert({f, new_id});
      if (f <= focal_bound) {
        focal_.insert({vcount, new_id});
      }
    }
  }

  // Non-timeout exit means we proved optimality: either the open list got
  // exhausted (no more candidates can improve the incumbent) or new_f_min
  // caught up to best_cost (lower bound met upper bound). In both cases the
  // recorded incumbent is provably optimal, so eps collapses to 0.
  const bool proved_optimal =
      !res_.timeout && res_.final_label_id != 0 &&
      best_cost < std::numeric_limits<double>::max();
  if (proved_optimal) {
    eps_ = 0.0;
    if (!res_.cost_optimization.empty() &&
        res_.eps_optimization.back() > 0.0) {
      res_.rt_optimization.push_back(timer.GetDurationSecond());
      res_.cost_optimization.push_back(best_cost);
      res_.eps_optimization.push_back(0.0);
    }
  }
  res_.final_sub_e = eps_;
  if (res_.timeout) {
    return res_.final_label_id != 0 ? res_.final_label_id : -1;
  }
  FinalizeLabelStats();
  return res_.final_label_id;
}

template <class Policy, class Store, bool IOG>
void RPVStar<Policy, Store, IOG>::UpdateFocalList(double old_f_min, double new_f_min) {
  const double old_bound = (1.0 + eps_) * old_f_min;
  const double new_bound = (1.0 + eps_) * new_f_min;
  if (new_bound <= old_bound) return;
  auto it_start = open_.upper_bound({old_bound, std::numeric_limits<long>::max()});
  auto it_end = open_.upper_bound({new_bound, std::numeric_limits<long>::max()});
  for (auto it = it_start; it != it_end; ++it) {
    focal_.insert({labels_[it->second].visited_count, it->second});
  }
}

template <class Policy, class Store, bool IOG>
void RPVStar<Policy, Store, IOG>::PruneAndRebuildFocal(double upper_bound) {
  auto cut_off = open_.lower_bound({upper_bound, -1});
  open_.erase(cut_off, open_.end());
  focal_.clear();
  if (open_.empty()) return;
  const double f_min = open_.begin()->first;
  const double bound = (1.0 + eps_) * f_min;
  for (const auto &kv : open_) {
    if (kv.first > bound) break;
    focal_.insert({labels_[kv.second].visited_count, kv.second});
  }
}

template <class Policy, class Store, bool IOG>
void RPVStar<Policy, Store, IOG>::FinalizeLabelStats() {
  // Now measured per (visited_set, current_vertex) state rather than per
  // vertex, so these report the size of a single state's Pareto front.
  double avg = -1.0;
  double max_points = -1.0;
  states_.PointStats(&avg, &max_points);
  res_.num_nondom_labels_avg = avg;
  res_.num_nondom_labels_max = max_points;
}

template <class Policy, class Store, bool IOG>
void RPVStar<Policy, Store, IOG>::ConstructPath() {
  if (res_.final_label_id == -1) return;
  std::vector<long> path;
  long id = res_.final_label_id;
  while (id != -1) {
    path.push_back(labels_[id].current_vertex);
    id = parent_[id];
  }
  std::reverse(path.begin(), path.end());
  res_.path = std::move(path);
}

// Explicit instantiations.
template class RPVStar<MinSumLegacyPolicy, RPVVertexFrontierStore, false>;
template class RPVStar<MinSumLegacyPolicy, RPVStateTable, false>;
template class RPVStar<MinSumPolicy, RPVScalarStateTable, true>;
template class RPVStar<MinMaxPolicy, RPVVertexFrontierStore, false>;
template class RPVStar<MinMaxPolicy, RPVStateTable, false>;
template class RPVStar<MinMaxPolicy, RPVStateTable, true>;

// ---- Free-function entry points ----

namespace {

template <class Policy, class Store, bool IOG>
int RunRPVAStar(SparseGraph *g, long vo, std::vector<double> &p,
                double time_limit, bool use_heuristic, HeuristicMode mode,
                RPVResult &res) {
  std::cout << "[INFO] Start to find Hamiltonian Path in " << time_limit
            << " s." << std::endl;
  RPVStar<Policy, Store, IOG> solver(use_heuristic, mode);
  long final_id = solver.Search(vo, g, time_limit, p);
  if (final_id == -1) {
    res = solver.GetRPVResult();
    return -1;
  }
  solver.ConstructPath();
  res = solver.GetRPVResult();
  return 0;
}

template <class Policy, class Store, bool IOG>
int RunRPVFocal(SparseGraph *g, long vo, std::vector<double> &p,
                double time_limit, bool use_heuristic, HeuristicMode mode,
                RPVResult &res) {
  std::cout << "[INFO] Start to find Hamiltonian Path in " << time_limit
            << " s." << std::endl;
  RPVStar<Policy, Store, IOG> solver(use_heuristic, mode);
  solver.FocalSearch(vo, g, time_limit, p);
  solver.ConstructPath();
  res = solver.GetRPVResult();
  return res.path.empty() ? -1 : 0;
}

} // namespace

// --- Dominance-mode dispatch -------------------------------------------
//
// The store and the admission timing are template parameters so the inner
// loop stays free of indirect calls: a virtual dominance interface would
// perturb exactly the quantity this ablation measures. Each mode therefore
// instantiates its own search; the mode is resolved once, here.

int RunMinSumRPV(SparseGraph *g, long vo, std::vector<double> &p,
                 double time_limit, bool use_heuristic, HeuristicMode mode,
                 DominanceMode dom, RPVResult &res) {
  switch (dom) {
  case DominanceMode::Vertex:
    return RunRPVAStar<MinSumLegacyPolicy, RPVVertexFrontierStore, false>(
        g, vo, p, time_limit, use_heuristic, mode, res);
  case DominanceMode::State2D:
    return RunRPVAStar<MinSumLegacyPolicy, RPVStateTable, false>(
        g, vo, p, time_limit, use_heuristic, mode, res);
  case DominanceMode::State:
    break;
  }
  return RunRPVAStar<MinSumPolicy, RPVScalarStateTable, true>(
      g, vo, p, time_limit, use_heuristic, mode, res);
}

int RunMinSumRPVFocal(SparseGraph *g, long vo, std::vector<double> &p,
                      double time_limit, bool use_heuristic, HeuristicMode mode,
                      DominanceMode dom, RPVResult &res) {
  switch (dom) {
  case DominanceMode::Vertex:
    return RunRPVFocal<MinSumLegacyPolicy, RPVVertexFrontierStore, false>(
        g, vo, p, time_limit, use_heuristic, mode, res);
  case DominanceMode::State2D:
    return RunRPVFocal<MinSumLegacyPolicy, RPVStateTable, false>(
        g, vo, p, time_limit, use_heuristic, mode, res);
  case DominanceMode::State:
    break;
  }
  return RunRPVFocal<MinSumPolicy, RPVScalarStateTable, true>(
      g, vo, p, time_limit, use_heuristic, mode, res);
}

// Min-max has no suffix-mass rewrite, so State and State2D share a store and
// differ only in when labels are admitted.
int RunMinMaxRPV(SparseGraph *g, long vo, std::vector<double> &p,
                 double time_limit, bool use_heuristic, HeuristicMode mode,
                 DominanceMode dom, RPVResult &res) {
  switch (dom) {
  case DominanceMode::Vertex:
    return RunRPVAStar<MinMaxPolicy, RPVVertexFrontierStore, false>(
        g, vo, p, time_limit, use_heuristic, mode, res);
  case DominanceMode::State2D:
    return RunRPVAStar<MinMaxPolicy, RPVStateTable, false>(
        g, vo, p, time_limit, use_heuristic, mode, res);
  case DominanceMode::State:
    break;
  }
  return RunRPVAStar<MinMaxPolicy, RPVStateTable, true>(
      g, vo, p, time_limit, use_heuristic, mode, res);
}

int RunMinMaxRPVFocal(SparseGraph *g, long vo, std::vector<double> &p,
                      double time_limit, bool use_heuristic, HeuristicMode mode,
                      DominanceMode dom, RPVResult &res) {
  switch (dom) {
  case DominanceMode::Vertex:
    return RunRPVFocal<MinMaxPolicy, RPVVertexFrontierStore, false>(
        g, vo, p, time_limit, use_heuristic, mode, res);
  case DominanceMode::State2D:
    return RunRPVFocal<MinMaxPolicy, RPVStateTable, false>(
        g, vo, p, time_limit, use_heuristic, mode, res);
  case DominanceMode::State:
    break;
  }
  return RunRPVFocal<MinMaxPolicy, RPVStateTable, true>(
      g, vo, p, time_limit, use_heuristic, mode, res);
}

// --- Short overloads ---------------------------------------------------
//
// Defaults: use_heuristic = true, HeuristicMode::PT, DominanceMode::State.

int RunMinSumRPV(SparseGraph *g, long vo, std::vector<double> &p,
                 double time_limit, RPVResult &res) {
  return RunMinSumRPV(g, vo, p, time_limit, true, HeuristicMode::PT,
                      DominanceMode::State, res);
}
int RunMinSumRPV(SparseGraph *g, long vo, std::vector<double> &p,
                 double time_limit, bool use_heuristic, RPVResult &res) {
  return RunMinSumRPV(g, vo, p, time_limit, use_heuristic, HeuristicMode::PT,
                      DominanceMode::State, res);
}
int RunMinSumRPV(SparseGraph *g, long vo, std::vector<double> &p,
                 double time_limit, bool use_heuristic, HeuristicMode mode,
                 RPVResult &res) {
  return RunMinSumRPV(g, vo, p, time_limit, use_heuristic, mode,
                      DominanceMode::State, res);
}

int RunMinSumRPVFocal(SparseGraph *g, long vo, std::vector<double> &p,
                      double time_limit, RPVResult &res) {
  return RunMinSumRPVFocal(g, vo, p, time_limit, true, HeuristicMode::PT,
                           DominanceMode::State, res);
}
int RunMinSumRPVFocal(SparseGraph *g, long vo, std::vector<double> &p,
                      double time_limit, bool use_heuristic, RPVResult &res) {
  return RunMinSumRPVFocal(g, vo, p, time_limit, use_heuristic,
                           HeuristicMode::PT, DominanceMode::State, res);
}
int RunMinSumRPVFocal(SparseGraph *g, long vo, std::vector<double> &p,
                      double time_limit, bool use_heuristic,
                      HeuristicMode mode, RPVResult &res) {
  return RunMinSumRPVFocal(g, vo, p, time_limit, use_heuristic, mode,
                           DominanceMode::State, res);
}

int RunMinMaxRPV(SparseGraph *g, long vo, std::vector<double> &p,
                 double time_limit, RPVResult &res) {
  return RunMinMaxRPV(g, vo, p, time_limit, true, HeuristicMode::PT,
                      DominanceMode::State, res);
}
int RunMinMaxRPV(SparseGraph *g, long vo, std::vector<double> &p,
                 double time_limit, bool use_heuristic, RPVResult &res) {
  return RunMinMaxRPV(g, vo, p, time_limit, use_heuristic, HeuristicMode::PT,
                      DominanceMode::State, res);
}
int RunMinMaxRPV(SparseGraph *g, long vo, std::vector<double> &p,
                 double time_limit, bool use_heuristic, HeuristicMode mode,
                 RPVResult &res) {
  return RunMinMaxRPV(g, vo, p, time_limit, use_heuristic, mode,
                      DominanceMode::State, res);
}

int RunMinMaxRPVFocal(SparseGraph *g, long vo, std::vector<double> &p,
                      double time_limit, RPVResult &res) {
  return RunMinMaxRPVFocal(g, vo, p, time_limit, true, HeuristicMode::PT,
                           DominanceMode::State, res);
}
int RunMinMaxRPVFocal(SparseGraph *g, long vo, std::vector<double> &p,
                      double time_limit, bool use_heuristic, RPVResult &res) {
  return RunMinMaxRPVFocal(g, vo, p, time_limit, use_heuristic,
                           HeuristicMode::PT, DominanceMode::State, res);
}
int RunMinMaxRPVFocal(SparseGraph *g, long vo, std::vector<double> &p,
                      double time_limit, bool use_heuristic,
                      HeuristicMode mode, RPVResult &res) {
  return RunMinMaxRPVFocal(g, vo, p, time_limit, use_heuristic, mode,
                           DominanceMode::State, res);
}

} // namespace raplab
