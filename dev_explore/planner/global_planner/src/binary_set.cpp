#include "../include/binary_set.hpp"

namespace raplab {
  std::ostream& operator<<(std::ostream& os, BinaryBits& bs) {
    os << "[";
    for (int i = 0; i < bs.size(); ++i) {
      os << int(bs.get(i));
      if (bs.size() - 1 != i) {
        os << ",";
      }
    }
    os << "]";
    return os;
  };
} // namespace raplab
