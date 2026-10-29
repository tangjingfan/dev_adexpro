/*******************************************
 * Author: Zhongqiang Richard Ren.
 * All Rights Reserved.
 *******************************************/

#ifndef ZHONGQIANGREN_BASIC_BINARY_SET_H_
#define ZHONGQIANGREN_BASIC_BINARY_SET_H_

#pragma once
#include "vec_type.hpp"
#include <algorithm>
#include <bitset>
#include <cstdlib>
#include <string>
#include <vector>

namespace raplab {

class BinaryVec {
public:
  std::vector<bool> data{};
    BinaryVec(){};
    BinaryVec(int n) : data(std::vector<bool>(n, false)){};
  BinaryVec(const BinaryVec &other) : data(other.data){};

  inline std::vector<bool>::size_type size() const { return data.size(); }
  inline bool get(int idx) const { return data.at(idx); }

  inline void set(int idx, bool v) { data[idx] = v; };

  inline bool is_subset(const BinaryVec &other) const {
    // check if `this` is subset of `other`
    for (std::vector<bool>::size_type i = 0; i < data.size(); ++i) {
      if (data.at(i) && (!other.get(static_cast<int>(i))))
        return false;
    }
    return true;
  }

  inline bool is_all(bool v) const {
    for (const auto &i : data)
      if (i != v)
        return false;
    return true;
  }

  inline void all_exists(std::vector<int> &idx) const {
    idx.clear();
    for (std::vector<bool>::size_type i = 0; i < data.size(); ++i)
      if (data.at(i) == true) {
        idx.push_back(static_cast<int>(i));
      }
  }

  inline void all_notexists(std::vector<int> &idx) const {
    idx.clear();
    for (std::vector<bool>::size_type i = 0; i < data.size(); ++i)
      if (data.at(i) == false) {
        idx.push_back(static_cast<int>(i));
      }
  }

  std::string to_str() const {
    std::string res = "|";
    for (std::vector<bool>::size_type i = 0; i < data.size(); ++i)
      res += std::to_string(data[i]);
    res += "|";
    std::reverse(res.begin(), res.end());
    return res;
  }
};

class BinaryBits {
  typedef unsigned long long ull;

public:
  // support max 256 bits
  // data[0]: 0 - 63
  // data[1]: 64 - 127
  // data[2]: 128 - 191
  // data[3]: 192 - 255
  static const ull B = 6;
  static const ull N = 1 << B;
  std::bitset<N> data[4]{};
  BinaryBits() { _size = -1; };
  BinaryBits(int n) { // init n bits to represent todo tasks
    // cannot larger than 256
    if (n > static_cast<int>(4ull * N))
      exit(1);
    _size = n;

    // 1. all tasks are marked as done
    data[0].set();
    data[1].set();
    data[2].set();
    data[3].set();

    // 2. mark [0, n) tasks as todo (false)
    int bid = 0;
    while (n > 0) {
      if (n > static_cast<int>(N)) {
        // set all bits to 0
        data[bid].reset();
        n -= static_cast<int>(N);
        bid++;
      } else {
        // set [0, n) bits to 0
        data[bid] ^= ((ull)1 << (ull)n) - (ull)1;
        break;
      }
    }
  };

  BinaryBits(const BinaryBits &other) {
    data[0] = other.data[0];
    data[1] = other.data[1];
    data[2] = other.data[2];
    data[3] = other.data[3];
    _size = other.size();
  };

  inline bool is_subset(const BinaryBits &other) const {
    return _is_subset(0, other.data[0]) && _is_subset(1, other.data[1]) &&
           _is_subset(2, other.data[2]) && _is_subset(3, other.data[3]);
  }

  inline bool is_all(bool v) const {
    for (int i = 0; i < _size; i++)
      if (get(i) != v)
        return false;
    return true;
    if (v)
      return data[0].all() && data[1].all() && data[2].all() && data[3].all();
    else {
      int n = _size;
      int bid = 0;
      while (n > 0) {
        if (n > static_cast<int>(N)) {
          // [bid*N, (bid+1)*N) bits must be 0
          if (data[bid].any())
            return false;
          n -= static_cast<int>(N);
          bid++;
        } else {
          // [bid*N..n) bits must be 0
          if ((~data[bid]) != (((ull)1 << (ull)n) - (ull)1))
            return false;
        }
      }
    }
    return true;
  }

  inline int size() const { return _size; }

  inline void set(int pos, bool v) {
    auto bid = pos >> B;
    auto idx = pos & (N - 1);
    data[bid][idx] = v;
  };

  inline bool get(int pos) const {
    auto bid = pos >> B;
    auto idx = pos & (N - 1);
    return data[bid][idx];
  }

  std::string to_str() const {
    std::string res = "|";
    for (int i = 0; i < _size; i++)
      res += std::to_string(get(i));
    res += "|";
    std::reverse(res.begin(), res.end());
    return res;
  }

  inline bool _is_subset(int bid, const std::bitset<N> &other) const {
    return (other & data[bid]) == data[bid];
  }

  inline void all_exists(std::vector<int> &idx) const {
    idx.clear();
    for (int i = 0; i < _size; i++)
      if (get(i) == true) {
        idx.push_back(i);
      }
  }

  inline void all_notexists(std::vector<int> &idx) const {
    idx.clear();
    for (int i = 0; i < _size; i++)
      if (get(i) == false) {
        idx.push_back(i);
      }
  }

private:
  int _size{0};
};

std::ostream &operator<<(std::ostream &os, BinaryBits &bs);
} // namespace raplab

#endif // ZHONGQIANGREN_BASIC_BINARY_SET_H_
