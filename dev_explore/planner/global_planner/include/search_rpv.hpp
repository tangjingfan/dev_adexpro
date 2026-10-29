#ifndef SEARCH_RPV_HPP
#define SEARCH_RPV_HPP

#include "graph.hpp"
#include "type_def.hpp"
#include "vec_type.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <set>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace raplab {

// Which dominance store the search uses. The three modes form a ladder that
// changes one thing at a time, so an ablation can attribute the speedup:
//   Vertex  -> one Pareto frontier per current_vertex, subset dominance over
//              (g, q); the original design.
//   State2D -> same rule and the same accounting, but indexed by the whole
//              state (visited_set, current_vertex). Isolates the indexing win.
//   State   -> for min-sum, additionally switches to suffix-mass accounting so
//              the state is scalar (see MinSumPolicyT). Isolates the
//              reformulation win. For min-max no such rewrite exists, so this
//              differs from State2D only in when labels are admitted.
enum class DominanceMode {
  Vertex,
  State2D,
  State,
};

inline int RPVPopcount(uint64_t x) {
#if defined(_MSC_VER)
  return static_cast<int>(__popcnt64(x));
#else
  return __builtin_popcountll(x);
#endif
}

// Choice of admissible heuristic for min-sum search.
//   Off    -> h = 0 (Dijkstra-like, no informed pruning).
//   PT     -> h = sum_{v in U} p(v) * (q + c(curr, v)). Cheap O(n).
//   MST    -> h = q*S + p_min * MST(U + {curr}). O(m^2) via Prim's.
//   MaxBoth-> max(PT, MST). Both admissible -> max admissible.
// Min-max only implements PT; the other values fall back to PT.
enum class HeuristicMode {
  Off,
  PT,
  MST,
  MaxBoth,
};

// Label for an RPV search node: at vertex `current_vertex`, having visited
// `visited_set`, with accumulated objective g, accumulated travel distance q,
// heuristic h, and f-value f.
//
// `visited_set` is a plain bitmask: bit v is set iff vertex v has been
// visited. This caps the search at 64 vertices, which is far beyond what a
// 2^n-state search can finish anyway; SetGraph() rejects larger graphs.
// `p_rem` is the probability mass still unvisited, P(V \ visited_set). It is
// a pure function of visited_set, carried along so the min-sum edge weight
// (see MinSumPolicy::step) needs no recomputation.
struct RPVLabel {
  long label_id;
  long current_vertex;
  uint64_t visited_set;
  long visited_count;
  double g;
  double q;
  double p_rem;
  double h;
  double f;

  RPVLabel() = default;
  RPVLabel(long id_, long vertex_, uint64_t visited_, long visited_count_,
           double g_, double q_, double p_rem_)
      : label_id(id_), current_vertex(vertex_), visited_set(visited_),
        visited_count(visited_count_), g(g_), q(q_), p_rem(p_rem_), h(0.0),
        f(0.0) {}
};

// Largest instance the solver accepts: one visited_set bit per vertex.
constexpr int kRPVMaxVertices = 64;

// Initial capacity for the dominance table. Reached-state counts grow roughly
// exponentially in n, so sizing off n up front removes most of the doubling
// rehashes without over-allocating on small instances.
inline size_t RPVInitialStateCapacity(int n) {
  int shift = n - 5;
  if (shift < 12) shift = 12;
  if (shift > 21) shift = 21;
  return static_cast<size_t>(1) << shift;
}

// All-visited mask for an n-vertex instance.
inline uint64_t RPVFullMask(int n) {
  return n >= 64 ? ~0ull : ((1ull << n) - 1ull);
}

// Home slot of state (mask, v) in an open-addressing table of cap_mask + 1
// (a power of two) slots, shared by the state-keyed dominance stores.
inline size_t RPVHomeSlot(uint64_t mask, long v, size_t cap_mask) {
  uint64_t x = mask + 0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(v + 1);
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  return (x ^ (x >> 33)) & cap_mask;
}

// Dominance store keyed by the *whole* state (visited_set, current_vertex).
//
// Dominance needs two conditions to hold at once: same current_vertex, and a
// visited-set relation. The previous layout bucketed on current_vertex alone
// and searched for the set relation by linear scan, so every query walked
// thousands of labels whose sets were unrelated to the query's (measured at
// ~3k entries per check for n=24, which is where ~99% of search time went).
// Keying on the full state makes that lookup O(1); what remains inside one
// bucket is the (g, q) Pareto front of a single state, which stays tiny.
//
// This narrows the rule from `visited_set(l2) ⊆ visited_set(l1)` to
// `visited_set(l2) == visited_set(l1)`, dropping the proper-subset case the
// old frontier also caught. That is a strict *weakening* of pruning: it can
// only retain labels the old rule discarded, never discard one the old rule
// kept. Optimality is therefore unaffected -- the cost is at most extra
// expansions, which the O(1) lookup more than repays.
//
// Layout: open addressing with linear probing for the state slots; the Pareto
// points live in one shared arena and are chained per slot, so no state ever
// allocates on its own. Points unlinked by Insert() stay in the arena as dead
// entries (bounded by the total number of insertions), which keeps Insert()
// allocation-free.
class RPVStateTable {
public:
  RPVStateTable() { Allocate(1 << 12); }

  void Reset(int /*n*/, size_t initial_cap) {
    Allocate(initial_cap);
    pts_.clear();
  }

  // True if some stored point of this state dominates (g, q).
  bool IsDominated(uint64_t mask, long v, double g, double q) const {
    const size_t i = Probe(mask, v);
    for (int c = heads_[i]; c != -1; c = pts_[c].next) {
      if (pts_[c].g <= g && pts_[c].q <= q) return true;
    }
    return false;
  }

  // True if (g, q) is still a live point of this state, i.e. it was inserted
  // and has not since been unlinked by a dominating insert. Used at pop time
  // to discard stale queue entries.
  bool Contains(uint64_t mask, long v, double g, double q) const {
    const size_t i = Probe(mask, v);
    for (int c = heads_[i]; c != -1; c = pts_[c].next) {
      if (pts_[c].g == g && pts_[c].q == q) return true;
    }
    return false;
  }

  // Drop the points of this state dominated by (g, q), then store it.
  void Insert(uint64_t mask, long v, double g, double q) {
    if ((used_ + 1) * 10 >= (cap_mask_ + 1) * 7) Grow();
    InsertAt(Probe(mask, v), mask, v, g, q);
  }

  size_t NumStates() const { return used_; }

  // Mean and max number of live Pareto points over all reached states.
  void PointStats(double *avg, double *max_out) const {
    size_t total = 0;
    size_t largest = 0;
    for (size_t i = 0; i <= cap_mask_; ++i) {
      if (heads_[i] == -1) continue;
      size_t count = 0;
      for (int c = heads_[i]; c != -1; c = pts_[c].next) ++count;
      total += count;
      if (count > largest) largest = count;
    }
    *avg = used_ == 0 ? -1.0 : static_cast<double>(total) / used_;
    *max_out = used_ == 0 ? -1.0 : static_cast<double>(largest);
  }

private:
  struct Point {
    double g;
    double q;
    int next;
  };

  // Unlink this state's points dominated by (g, q), then chain (g, q) on.
  // `i` must be the slot returned by Probe(mask, v) with capacity already
  // assured by the caller.
  void InsertAt(size_t i, uint64_t mask, long v, double g, double q) {
    if (heads_[i] == -1) {
      key_mask_[i] = mask;
      key_vertex_[i] = v;
      ++used_;
    } else {
      int prev = -1;
      int c = heads_[i];
      while (c != -1) {
        const int nxt = pts_[c].next;
        if (g <= pts_[c].g && q <= pts_[c].q) {
          if (prev == -1) {
            heads_[i] = nxt;
          } else {
            pts_[prev].next = nxt;
          }
        } else {
          prev = c;
        }
        c = nxt;
      }
    }
    pts_.push_back({g, q, heads_[i]});
    heads_[i] = static_cast<int>(pts_.size()) - 1;
  }

  void Allocate(size_t cap) {
    key_mask_.assign(cap, 0);
    key_vertex_.assign(cap, -1);
    heads_.assign(cap, -1);
    cap_mask_ = cap - 1;
    used_ = 0;
  }

  void Grow() {
    std::vector<uint64_t> old_mask;
    std::vector<long> old_vertex;
    std::vector<int> old_heads;
    old_mask.swap(key_mask_);
    old_vertex.swap(key_vertex_);
    old_heads.swap(heads_);
    Allocate(old_heads.size() * 2);
    for (size_t j = 0; j < old_heads.size(); ++j) {
      if (old_heads[j] == -1) continue;
      const size_t i = Probe(old_mask[j], old_vertex[j]);
      key_mask_[i] = old_mask[j];
      key_vertex_[i] = old_vertex[j];
      heads_[i] = old_heads[j];
      ++used_;
    }
  }

  // Returns the slot holding (mask, v), or the first empty slot if absent.
  // Insert() keeps the load factor at or below 0.7, so an empty slot always
  // exists and the probe terminates.
  inline size_t Probe(uint64_t mask, long v) const {
    size_t i = RPVHomeSlot(mask, v, cap_mask_);
    while (heads_[i] != -1 &&
           !(key_mask_[i] == mask && key_vertex_[i] == v)) {
      i = (i + 1) & cap_mask_;
    }
    return i;
  }

  std::vector<uint64_t> key_mask_;
  std::vector<long> key_vertex_;
  std::vector<int> heads_;
  std::vector<Point> pts_;
  size_t cap_mask_ = 0;
  size_t used_ = 0;
};

// The original dominance store, kept so the ablation can measure what keying
// on the whole state actually bought. One Pareto frontier per current_vertex,
// held in a vector sorted by (popcount desc, g asc): the outer key lets a
// query stop once popcount drops below its own, and within a tier the g-asc
// order lets it skip the rest the moment it sees g > query.g, since a
// dominator needs g <= query.g.
//
// Its rule is strictly stronger than the state-keyed stores' -- it also
// catches the proper-subset case visited_set(query) subset-of visited_set(cand)
// -- but finding those candidates costs a linear scan over every label parked
// at that vertex, whatever their visited set.
class RPVVertexFrontierStore {
public:
  void Reset(int n, size_t /*initial_cap*/) {
    frontiers_.assign(n, {});
  }

  bool IsDominated(uint64_t mask, long v, double g, double q) const {
    const std::vector<Entry> &labels = frontiers_[v];
    const int vc = RPVPopcount(mask);
    const size_t sz = labels.size();
    size_t i = 0;
    while (i < sz) {
      const int tier_vc = labels[i].vcount;
      if (tier_vc < vc) break;
      while (i < sz && labels[i].vcount == tier_vc) {
        const Entry &cand = labels[i];
        if (cand.g > g) {
          // g ascends within a tier: nothing later here can dominate.
          while (i < sz && labels[i].vcount == tier_vc) ++i;
          break;
        }
        if (cand.q <= q && (mask & cand.mask) == mask) return true;
        ++i;
      }
    }
    return false;
  }

  void Insert(uint64_t mask, long v, double g, double q) {
    std::vector<Entry> &labels = frontiers_[v];
    const int vc = RPVPopcount(mask);
    // A dominator needs vcount >= the dominated one's, so only entries at or
    // after the first vcount <= vc can be dominated by this one.
    auto remove_start = std::lower_bound(
        labels.begin(), labels.end(), vc,
        [](const Entry &a, int c) { return a.vcount > c; });
    auto new_end = std::remove_if(
        remove_start, labels.end(), [&](const Entry &e) {
          return vc >= e.vcount && g <= e.g && q <= e.q &&
                 (e.mask & mask) == e.mask;
        });
    labels.erase(new_end, labels.end());
    const Entry entry{mask, g, q, vc};
    auto pos = std::lower_bound(
        labels.begin(), labels.end(), entry, [](const Entry &a, const Entry &b) {
          if (a.vcount != b.vcount) return a.vcount > b.vcount;
          return a.g < b.g;
        });
    labels.insert(pos, entry);
  }

  bool Contains(uint64_t mask, long v, double g, double q) const {
    for (const Entry &e : frontiers_[v]) {
      if (e.mask == mask && e.g == g && e.q == q) return true;
    }
    return false;
  }

  size_t NumStates() const {
    size_t total = 0;
    for (const auto &f : frontiers_) total += f.size();
    return total;
  }

  void PointStats(double *avg, double *max_out) const {
    if (frontiers_.empty()) {
      *avg = -1.0;
      *max_out = -1.0;
      return;
    }
    size_t total = 0;
    size_t largest = 0;
    for (const auto &f : frontiers_) {
      total += f.size();
      if (f.size() > largest) largest = f.size();
    }
    *avg = static_cast<double>(total) / frontiers_.size();
    *max_out = static_cast<double>(largest);
  }

private:
  struct Entry {
    uint64_t mask;
    double g;
    double q;
    int vcount;
  };
  std::vector<std::vector<Entry>> frontiers_;
};

struct RPVResult {
  long final_label_id = -1;
  std::vector<long> path;
  CostVec final_cost;
  std::vector<double> rt_optimization;
  std::vector<double> cost_optimization;
  std::vector<double> eps_optimization;
  long n_generated = 0;
  long n_expanded = 0;
  long n_domCheck = 0;
  double rt_search = 0.0;
  double rt_greedy = -1.0;
  double final_sub_e = 0.0;
  bool timeout = false;
  double num_nondom_labels_avg = -1;
  double num_nondom_labels_max = -1;
};

// Scalar counterpart of RPVStateTable, for objectives whose state carries a
// single value (min-sum, after the suffix-mass rewrite below). Since such a
// state's frontier is always exactly one point, the value lives inline in the
// slot: no arena, no chaining, no second coordinate. Same interface as
// RPVStateTable so RPVStar can select one via Policy::StateStore.
class RPVScalarStateTable {
public:
  RPVScalarStateTable() { Allocate(1 << 12); }

  void Reset(int /*n*/, size_t initial_cap) { Allocate(initial_cap); }

  bool IsDominated(uint64_t mask, long v, double g, double /*q*/) const {
    const size_t i = Probe(mask, v);
    return key_vertex_[i] != -1 && val_[i] <= g;
  }

  void Insert(uint64_t mask, long v, double g, double /*q*/) {
    if ((used_ + 1) * 10 >= (cap_mask_ + 1) * 7) Grow();
    const size_t i = Probe(mask, v);
    if (key_vertex_[i] == -1) {
      key_mask_[i] = mask;
      key_vertex_[i] = v;
      val_[i] = g;
      ++used_;
      return;
    }
    if (g < val_[i]) val_[i] = g;
  }

  bool Contains(uint64_t mask, long v, double g, double /*q*/) const {
    const size_t i = Probe(mask, v);
    return key_vertex_[i] != -1 && val_[i] == g;
  }

  size_t NumStates() const { return used_; }

  void PointStats(double *avg, double *max_out) const {
    *avg = used_ == 0 ? -1.0 : 1.0;
    *max_out = used_ == 0 ? -1.0 : 1.0;
  }

private:
  void Allocate(size_t cap) {
    key_mask_.assign(cap, 0);
    key_vertex_.assign(cap, -1);
    val_.assign(cap, 0.0);
    cap_mask_ = cap - 1;
    used_ = 0;
  }

  void Grow() {
    std::vector<uint64_t> om;
    std::vector<long> ov;
    std::vector<double> oval;
    om.swap(key_mask_);
    ov.swap(key_vertex_);
    oval.swap(val_);
    Allocate(ov.size() * 2);
    for (size_t j = 0; j < ov.size(); ++j) {
      if (ov[j] == -1) continue;
      const size_t i = Probe(om[j], ov[j]);
      key_mask_[i] = om[j];
      key_vertex_[i] = ov[j];
      val_[i] = oval[j];
      ++used_;
    }
  }

  inline size_t Probe(uint64_t mask, long v) const {
    size_t i = RPVHomeSlot(mask, v, cap_mask_);
    while (key_vertex_[i] != -1 &&
           !(key_mask_[i] == mask && key_vertex_[i] == v)) {
      i = (i + 1) & cap_mask_;
    }
    return i;
  }

  std::vector<uint64_t> key_mask_;
  std::vector<long> key_vertex_;
  std::vector<double> val_;
  size_t cap_mask_ = 0;
  size_t used_ = 0;
};

// Policy: minimum-sum objective, in suffix-mass coordinates.
//
// The natural recursion g = g + p(v)*(q + c(u,v)) is not a sufficient
// statistic: the cost still to be paid depends on q, because every yet
// unvisited vertex is charged for the distance already travelled. That forces
// a 2-D (g, q) Pareto front per state.
//
// Charging each edge up front for every vertex that has not been served yet
// removes the dependence. Writing U = V \ S and P(U) for its probability mass,
//
//   g'(S u {v}, v) = g'(S, u) + c(u,v) * P(U)     with  g' = g + q * P(U),
//
// and the objective splits as  J = g'(S,u) + sum_{w in U} p(w) * s_w, where
// s_w is travel *after* leaving u. The second term depends only on (S, u) and
// the future, so two labels of one state compare on the scalar g' alone --
// no q, no Pareto front. Both accountings agree at the goal (P(empty) = 0), so
// the reported cost is unchanged; the f-values agree everywhere, so the search
// order is unchanged too.
//
//   admissible h': sum over unvisited w of p(w) * c(curr, w)   (no q term --
//   it now lives inside g'), which is consistent by the triangle inequality.
//   f(g', h') = g' + h'
// SuffixMass selects the accounting: true is the rewrite described above
// (scalar state), false the natural g = g + p(v)*(q + c) recursion (2-D
// state). Both are exact and report the same optimum; the parameter exists so
// the dominance ablation can hold the accounting fixed while it varies the
// store, and vary it while holding the store fixed.
template <bool SuffixMass> struct MinSumPolicyT {
  static constexpr const char *name() { return "min-sum"; }

  // Label recursion in the search's cost coordinates.
  static inline double step(double g, double prob_v, double new_q,
                            double edge_cost, double p_rem) {
    if constexpr (SuffixMass) {
      (void)prob_v;
      (void)new_q;
      return g + edge_cost * p_rem;
    } else {
      (void)edge_cost;
      (void)p_rem;
      return g + prob_v * new_q;
    }
  }

  // True objective accumulation. Used by the greedy upper bound, which only
  // needs the value of a complete tour, where the two coordinates coincide.
  static inline double accumulate(double g, double prob_v, double new_q) {
    return g + prob_v * new_q;
  }

  // Second dominance coordinate. Under suffix-mass accounting g' already
  // absorbs q, so the state is scalar and a 2-D store degenerates to a scalar
  // test; under the natural accounting q is a real second dimension.
  static inline double dom_q(const RPVLabel &l) {
    if constexpr (SuffixMass) {
      (void)l;
      return 0.0;
    } else {
      return l.q;
    }
  }

  static inline double f_value(double g, double h) { return g + h; }

  // h = sum_{v unvisited} p(v) * c(curr, v).
  static inline double heuristic_pt(const RPVLabel &l,
                                       const std::vector<double> &cost_matrix,
                                       const std::vector<double> &prob,
                                       int n) {
    double h = 0.0;
    const long curr = l.current_vertex;
    const uint64_t vis = l.visited_set;
    const double *row = &cost_matrix[curr * n];
    for (int v = 0; v < n; ++v) {
      if ((vis >> v) & 1ull) {
        continue;
      }
      h += prob[v] * (SuffixMass ? row[v] : (l.q + row[v]));
    }
    return h;
  }

  // h = p_min * MST({curr} U U). The q * P(U) term the natural accounting
  // needed here is already part of g'. Computed via Prim's; thread-local
  // buffers avoid per-call allocation in the hot succ-loop.
  static inline double heuristic_mst(const RPVLabel &l,
                                     const std::vector<double> &cost_matrix,
                                     const std::vector<double> &prob,
                                     int n) {
    thread_local std::vector<int> unv;
    thread_local std::vector<double> key;
    thread_local std::vector<char> in_tree;

    unv.clear();
    double p_sum = 0.0;
    double p_min = std::numeric_limits<double>::infinity();
    const uint64_t vis = l.visited_set;
    for (int v = 0; v < n; ++v) {
      if ((vis >> v) & 1ull) continue;
      unv.push_back(v);
      p_sum += prob[v];
      if (prob[v] < p_min) p_min = prob[v];
    }

    const double h_q = SuffixMass ? 0.0 : l.q * p_sum;
    const int m = static_cast<int>(unv.size());
    if (m == 0) return h_q;

    // Prim's from `curr`, expanding to all unvisited vertices.
    const long curr = l.current_vertex;
    const double *curr_row = &cost_matrix[curr * n];
    key.assign(m, 0.0);
    in_tree.assign(m, 0);
    for (int i = 0; i < m; ++i) key[i] = curr_row[unv[i]];

    double mst = 0.0;
    for (int added = 0; added < m; ++added) {
      int best = -1;
      double best_d = std::numeric_limits<double>::infinity();
      for (int i = 0; i < m; ++i) {
        if (in_tree[i]) continue;
        if (key[i] < best_d) {
          best_d = key[i];
          best = i;
        }
      }
      if (best < 0) break;
      in_tree[best] = 1;
      mst += best_d;
      const double *row = &cost_matrix[unv[best] * n];
      for (int i = 0; i < m; ++i) {
        if (in_tree[i]) continue;
        const double c = row[unv[i]];
        if (c < key[i]) key[i] = c;
      }
    }

    return h_q + p_min * mst;
  }
};

using MinSumPolicy = MinSumPolicyT<true>;        // suffix-mass, scalar state
using MinSumLegacyPolicy = MinSumPolicyT<false>; // natural accounting, 2-D

// Policy: minimum-max objective.
//   g(pi') = max(g(pi), p(v_{k+1}) * (q(pi) + c(v_k, v_{k+1})))
//   admissible h: max over unvisited v of p(v) * (q + d(curr, v))
//   f(g, h) = max(g, h)
struct MinMaxPolicy {
  static constexpr const char *name() { return "min-max"; }

  // No suffix-mass rewrite exists here: q sits inside a max(), so it cannot be
  // factored out into a coefficient that depends on the visited set alone.
  // The state therefore stays 2-D and step() is just accumulate().
  static inline double step(double g, double prob_v, double new_q,
                            double edge_cost, double p_rem) {
    (void)edge_cost;
    (void)p_rem;
    return accumulate(g, prob_v, new_q);
  }

  static inline double accumulate(double g, double prob_v, double new_q) {
    const double cand = prob_v * new_q;
    return g >= cand ? g : cand;
  }

  static inline double dom_q(const RPVLabel &l) { return l.q; }

  static inline double f_value(double g, double h) {
    return g >= h ? g : h;
  }

  static inline double heuristic_pt(const RPVLabel &l,
                                       const std::vector<double> &cost_matrix,
                                       const std::vector<double> &prob,
                                       int n) {
    double h = 0.0;
    const long curr = l.current_vertex;
    const uint64_t vis = l.visited_set;
    const double *row = &cost_matrix[curr * n];
    for (int v = 0; v < n; ++v) {
      if ((vis >> v) & 1ull) {
        continue;
      }
      const double cand = prob[v] * (l.q + row[v]);
      if (cand > h) {
        h = cand;
      }
    }
    return h;
  }

  // MST is not natively meaningful for min-max (max over cumulative arrivals,
  // not a sum). Fall back to the PT heuristic so the same dispatch code
  // can be used for both policies.
  static inline double heuristic_mst(const RPVLabel &l,
                                     const std::vector<double> &cost_matrix,
                                     const std::vector<double> &prob,
                                     int n) {
    return heuristic_pt(l, cost_matrix, prob, n);
  }
};

// InsertOnGenerate controls when a surviving label joins the dominance store:
// at generation (it then prunes its siblings immediately, and stale queue
// entries are dropped at pop by a liveness check) or at expansion (the
// original behaviour, where the pop-time check is an ordinary dominance test).
template <class Policy, class Store, bool InsertOnGenerate>
class RPVStar {
public:
  // eps_ exists only for FocalSearch, which recomputes it each iteration
  // from (best_cost / f_min - 1); the constructor's initial value is
  // never read before being overwritten. Search() (A*) never reads it.
  explicit RPVStar(bool use_heuristic = true,
                   HeuristicMode mode = HeuristicMode::PT)
      : eps_(0.0), use_heuristic_(use_heuristic),
        heuristic_mode_(mode), focal_(FocalComparator(&labels_)) {}

  RPVResult GetRPVResult() const { return res_; }

  // A* search: returns optimal label id on success, -1 on timeout.
  long Search(long vo, PlannerGraph *g, double time_limit,
              std::vector<double> &prob_vec);

  // Anytime focal search: returns best label id found; on success path is
  // populated with an (eventually optimal) solution.
  long FocalSearch(long vo, PlannerGraph *g, double time_limit,
                   std::vector<double> &prob_vec);

  // Rebuild path from final_label_id via parent_ chain.
  void ConstructPath();

private:
  void SetGraph(PlannerGraph *g);
  void SetProb(std::vector<double> &p) { prob_ = p; }

  inline double GetCost(long u, long v) const {
    return cost_matrix_[u * n_ + v];
  }

  inline double Heuristic(const RPVLabel &l) const {
    if (!use_heuristic_) return 0.0;
    switch (heuristic_mode_) {
      case HeuristicMode::PT:
        return Policy::heuristic_pt(l, cost_matrix_, prob_, n_);
      case HeuristicMode::MST:
        return Policy::heuristic_mst(l, cost_matrix_, prob_, n_);
      case HeuristicMode::MaxBoth: {
        const double a = Policy::heuristic_pt(l, cost_matrix_, prob_, n_);
        const double b = Policy::heuristic_mst(l, cost_matrix_, prob_, n_);
        return a >= b ? a : b;
      }
      case HeuristicMode::Off:
        return 0.0;
    }
    return 0.0;
  }

  // Read-only dominance test. Kept separate from Admit() so a label can be
  // rejected, or later dropped by the f-bound, without ever claiming a slot:
  // states that are generated but never expanded used to inflate the table
  // ~5x, and a bigger table slows down every probe, not just the inserts.
  bool Dominated(const RPVLabel &l) {
    res_.n_domCheck++;
    return states_.IsDominated(l.visited_set, l.current_vertex, l.g,
                               Policy::dom_q(l));
  }

  // Put l on its state's frontier so it prunes siblings and later labels.
  void Admit(const RPVLabel &l) {
    states_.Insert(l.visited_set, l.current_vertex, l.g, Policy::dom_q(l));
  }

  // Returns true if l is still on its state's frontier, i.e. no later label
  // has superseded it while it sat in the open list.
  bool StillLive(const RPVLabel &l) const {
    return states_.Contains(l.visited_set, l.current_vertex, l.g,
                            Policy::dom_q(l));
  }

  // Greedy nearest-neighbor tour to seed the upper bound.
  double RunGreedy(long start, std::vector<long> &path_out) const;

  // Create the root label at vo, store it as label 0 and return a copy; the
  // caller pushes it onto its own open list.
  RPVLabel CreateRootLabel(long vo);

  // Move labels from open with f in (old_bound, new_bound] into focal.
  void UpdateFocalList(double old_f_min, double new_f_min);

  // Drop labels with f >= upper_bound from open and rebuild focal.
  void PruneAndRebuildFocal(double upper_bound);

  void FinalizeLabelStats();

  // --- state ---
  int n_ = 0;
  uint64_t full_mask_ = 0;                   // all-visited bitmask
  std::vector<double> cost_matrix_;          // flat n*n
  std::vector<std::vector<long>> adj_;       // adj_[u][i] = i-th successor of u
  std::vector<std::vector<double>> adj_cost_; // adj_cost_[u][i] = scalar edge cost
  std::vector<double> prob_;
  double time_limit_ = 0.0;

  double eps_;
  bool use_heuristic_;
  HeuristicMode heuristic_mode_ = HeuristicMode::PT;

  // labels_[id] is owned by the solver; we copy/index by label_id everywhere.
  // Stored in a vector (faster than deque) but we never hold long-lived
  // references across push_back.
  std::vector<RPVLabel> labels_;
  std::vector<long> parent_;

  // Dominance record for every reached state.
  Store states_;

  // FocalSearch uses open_ for f-ordered range queries (UpdateFocalList,
  // PruneAndRebuildFocal). Search() uses open_heap_ (binary min-heap) since
  // it only needs push and pop-min. Each routine touches only its own
  // container; the other stays empty for that call.
  std::set<std::pair<double, long>> open_;
  std::vector<std::pair<double, long>> open_heap_;

  struct FocalComparator {
    const std::vector<RPVLabel> *labels;
    FocalComparator(const std::vector<RPVLabel> *l = nullptr) : labels(l) {}
    bool operator()(const std::pair<long, long> &a,
                    const std::pair<long, long> &b) const {
      if (a.first != b.first) return a.first < b.first;
      const auto &la = (*labels)[a.second];
      const auto &lb = (*labels)[b.second];
      if (std::abs(la.f - lb.f) > 1e-6) return la.f > lb.f;
      return a.second < b.second;
    }
  };
  std::set<std::pair<long, long>, FocalComparator> focal_;

  RPVResult res_;
};

// A* entry points. The shorter overloads default to use_heuristic = true,
// HeuristicMode::PT and DominanceMode::State.
int RunMinSumRPV(raplab::SparseGraph *g, long vo, std::vector<double> &p,
                 double time_limit, raplab::RPVResult &res);
int RunMinSumRPV(raplab::SparseGraph *g, long vo, std::vector<double> &p,
                 double time_limit, bool use_heuristic, raplab::RPVResult &res);
int RunMinSumRPV(raplab::SparseGraph *g, long vo, std::vector<double> &p,
                 double time_limit, bool use_heuristic,
                 raplab::HeuristicMode mode, raplab::RPVResult &res);
int RunMinSumRPV(raplab::SparseGraph *g, long vo, std::vector<double> &p,
                 double time_limit, bool use_heuristic,
                 raplab::HeuristicMode mode, raplab::DominanceMode dom,
                 raplab::RPVResult &res);

int RunMinMaxRPV(raplab::SparseGraph *g, long vo, std::vector<double> &p,
                 double time_limit, raplab::RPVResult &res);
int RunMinMaxRPV(raplab::SparseGraph *g, long vo, std::vector<double> &p,
                 double time_limit, bool use_heuristic, raplab::RPVResult &res);
int RunMinMaxRPV(raplab::SparseGraph *g, long vo, std::vector<double> &p,
                 double time_limit, bool use_heuristic,
                 raplab::HeuristicMode mode, raplab::RPVResult &res);
int RunMinMaxRPV(raplab::SparseGraph *g, long vo, std::vector<double> &p,
                 double time_limit, bool use_heuristic,
                 raplab::HeuristicMode mode, raplab::DominanceMode dom,
                 raplab::RPVResult &res);

// Anytime focal-search entry points. Sub-optimality bound is derived
// dynamically from (best_cost / f_min - 1) each iteration; there is no
// caller-controlled epsilon. Same overload defaults as the A* entry points.
int RunMinSumRPVFocal(raplab::SparseGraph *g, long vo, std::vector<double> &p,
                      double time_limit, raplab::RPVResult &res);
int RunMinSumRPVFocal(raplab::SparseGraph *g, long vo, std::vector<double> &p,
                      double time_limit, bool use_heuristic,
                      raplab::RPVResult &res);
int RunMinSumRPVFocal(raplab::SparseGraph *g, long vo, std::vector<double> &p,
                      double time_limit, bool use_heuristic,
                      raplab::HeuristicMode mode, raplab::RPVResult &res);
int RunMinSumRPVFocal(raplab::SparseGraph *g, long vo, std::vector<double> &p,
                      double time_limit, bool use_heuristic,
                      raplab::HeuristicMode mode, raplab::DominanceMode dom,
                      raplab::RPVResult &res);

int RunMinMaxRPVFocal(raplab::SparseGraph *g, long vo, std::vector<double> &p,
                      double time_limit, raplab::RPVResult &res);
int RunMinMaxRPVFocal(raplab::SparseGraph *g, long vo, std::vector<double> &p,
                      double time_limit, bool use_heuristic,
                      raplab::RPVResult &res);
int RunMinMaxRPVFocal(raplab::SparseGraph *g, long vo, std::vector<double> &p,
                      double time_limit, bool use_heuristic,
                      raplab::HeuristicMode mode, raplab::RPVResult &res);
int RunMinMaxRPVFocal(raplab::SparseGraph *g, long vo, std::vector<double> &p,
                      double time_limit, bool use_heuristic,
                      raplab::HeuristicMode mode, raplab::DominanceMode dom,
                      raplab::RPVResult &res);

} // namespace raplab

#endif
