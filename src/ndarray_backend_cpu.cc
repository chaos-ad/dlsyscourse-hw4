#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <sstream>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}

void IncrementMultiIndex(std::vector<std::size_t> * index_ptr, std::vector<uint32_t> const& shape) {
  assert(index_ptr && "index is null");
  std::vector<std::size_t> & index = *index_ptr;
  // TODO: ugly int64_t to avoid underflow, refactor me
  for(int64_t i = shape.size() - 1; i >= 0; --i) {
    if(index.at(i) < shape.at(i) - 1) {
        index[i] += 1;
        break;
    } else {
      index.at(i) = 0;
    }
  }
}

std::size_t MultiIndexToFlatIndex(std::vector<std::size_t> const& index, std::vector<uint32_t> const& strides, uint32_t offset) {
  std::size_t result = offset;
  for(std::size_t dim = 0, dim_end = index.size(); dim < dim_end; ++dim) {
    result += (index[dim] * strides[dim]);
  }
  return result;
}

template <typename T>
std::string ArrayToString(std::vector<T> const& index) {
  std::ostringstream out;
  out << "(";
  for(std::size_t dim = 0, dim_end = index.size(); dim < dim_end; ++dim) {
    out << index[dim];
    if(dim < dim_end - 1) {
      out << ", ";
    }
  }
  out << ")";
  return out.str();
}

std::string ArrayToString(AlignedArray const& index) {
  std::ostringstream out;
  out << "(";
  for(std::size_t dim = 0, dim_end = index.size; dim < dim_end; ++dim) {
    out << index.ptr[dim];
    if(dim < dim_end - 1) {
      out << ", ";
    }
  }
  out << ")";
  return out.str();
}


std::vector<uint32_t> GetCompactStrides(std::vector<uint32_t> const& shape) {
  std::vector<uint32_t> result(shape.size(), 0);
  uint32_t multiple = 1;
  size_t size = shape.size();
  for(size_t i = 0; i < size; ++i) {
    size_t idx = (size - 1) - i;
    // std::cout << "DEBUG[GetCompactStrides]: step=" << i << ", "
    //           << "idx=" << idx << ", "
    //           << "multiple=" << multiple << std::endl; 
    result[idx] = multiple;
    multiple *= shape[idx];
  }
  return result;
}

size_t GetSparseIdx(
    size_t compact_idx,
    std::vector<uint32_t> const& compact_strides,
    size_t sparse_offset,
    std::vector<uint32_t> const& sparse_strides
) {
  // std::cout << "DEBUG[Compact]: GetSparseIdx("
  //           << "compact_idx=" << compact_idx << ", "
  //           << "compact_strides=" << ArrayToString(compact_strides) << ", "
  //           << "sparse_offset=" << sparse_offset << ", "
  //           << "sparse_strides=" << ArrayToString(sparse_strides) << "):"
  //           << std::endl;
  size_t sparse_idx = sparse_offset;
  size_t size = compact_strides.size();
  for(size_t i = 0; i < size; ++i) {
    uint32_t compact_stride = compact_strides[i];
    uint32_t sparse_pos = compact_idx / compact_stride;
    sparse_idx += sparse_pos * sparse_strides[i];
    compact_idx %= compact_stride;
    // std::cout << "DEBUG[Compact:GetSparseIdx]: step " << i << ": "
    //           << "compact_strides[" << i << "]=" << compact_strides[i] << ", "
    //           << "sparse_strides[" << i << "]=" << sparse_strides[i] << ", "
    //           << "sparse_pos=" << sparse_pos << ", "
    //           << "sparse_idx=" << sparse_idx << ", "
    //           << "compact_idx=" << compact_idx
    //           << std::endl;

  }
  // std::cout << "DEBUG[Compact]: GetSparseIdx("
  //           << "compact_idx=" << compact_idx << ", "
  //           << "compact_strides=" << ArrayToString(compact_strides) << ", "
  //           << "sparse_offset=" << sparse_offset << ", "
  //           << "sparse_strides=" << ArrayToString(sparse_strides) << ") -> "
  //           << sparse_idx
  //           << std::endl;
  return sparse_idx;
}

void Compact(AlignedArray const& a, AlignedArray* out, std::vector<uint32_t> shape,
             std::vector<uint32_t> in_strides, size_t in_offset) {
  /**
   * Compact an array in memory
   * 
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   in_strides: strides of the *in* array (not out, which has compact strides)
   *   in_offset: offset of the *in* array (not out, which has zero offset, being compact)
   * 
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */

  /// BEGIN YOUR SOLUTION
  
  // std::cout << "DEBUG[Compact]: input.offset = " << in_offset << std::endl;
  // std::cout << "DEBUG[Compact]: input.shape = " << ArrayToString(shape) << std::endl;
  // std::cout << "DEBUG[Compact]: input.strides = " << ArrayToString(in_strides) << std::endl;
  // std::cout << "DEBUG[Compact]: output.size = " << out->size << std::endl;

  std::vector<uint32_t> out_strides = GetCompactStrides(shape);

  // std::cout << "DEBUG[Compact]: output.strides = " << ArrayToString(out_strides) << std::endl;

  size_t size = out->size;
  for(size_t out_idx = 0; out_idx < size; ++out_idx) {
    size_t in_idx = GetSparseIdx(out_idx, out_strides, in_offset, in_strides);
    // std::cout << "DEBUG[Compact]: copying out->ptr[" << out_idx << "] = a.ptr[" << in_idx << "] = " << a.ptr[in_idx] << std::endl;
    out->ptr[out_idx] = a.ptr[in_idx];
  }

  /// END YOUR SOLUTION
}

void EwiseSetitem(const AlignedArray& in, AlignedArray* out, std::vector<uint32_t> shape,
                  std::vector<uint32_t> out_strides, size_t out_offset) {
  /**
   * Set items in a (non-compact) array
   * 
   * Args:
   *   in: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   out_strides: strides of the *out* array (not a, which has compact strides)
   *   out_offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN YOUR SOLUTION
  
  // std::size_t size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<std::size_t>());
  
  std::vector<uint32_t> in_strides = GetCompactStrides(shape);

  size_t size = in.size;
  for(size_t in_idx = 0; in_idx < size; ++in_idx) {
    size_t out_idx = GetSparseIdx(in_idx, in_strides, out_offset, out_strides);
    // std::cout << "DEBUG[Compact]: copying out->ptr[" << out_idx << "] = in.ptr[" << in_idx << "] = " << in.ptr[in_idx] << std::endl;
    out->ptr[out_idx] = in.ptr[in_idx];
  }

  /// END YOUR SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<uint32_t> shape,
                   std::vector<uint32_t> out_strides, size_t out_offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   out_strides: strides of the out array
   *   out_offset: offset of the out array
   */

  /// BEGIN YOUR SOLUTION
  
  // std::size_t out_size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<std::size_t>());
  std::vector<uint32_t> in_strides = GetCompactStrides(shape);

  for(size_t in_idx = 0; in_idx < size; ++in_idx) {
    size_t out_idx = GetSparseIdx(in_idx, in_strides, out_offset, out_strides);
    // std::cout << "DEBUG[Compact]: setting out->ptr[" << out_idx << "] = val = " << val << std::endl;
    out->ptr[out_idx] = val;
  }

  /// END YOUR SOLUTION
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}


/**
 * In the code the follows, use the above template to create analogous element-wise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

/// BEGIN YOUR SOLUTION

void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * val;
  }
}

void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] / val;
  }
}

void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = pow(a.ptr[i], val);
  }
}

void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * b.ptr[i];
  }
}

void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] / b.ptr[i];
  }
}

void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max(a.ptr[i], b.ptr[i]);
  }
}

void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max(a.ptr[i], val);
  }
}

void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] == val ? 1 : 0;
  }
}


void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] >= val ? 1 : 0;
  }
}
 
void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] == b.ptr[i] ? 1 : 0;
  }
}

void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] >= b.ptr[i] ? 1 : 0;
  }
}

void EwiseLog(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::log(a.ptr[i]);
  }
}

void EwiseExp(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::exp(a.ptr[i]);
  }
}

void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::tanh(a.ptr[i]);
  }
}


/// END YOUR SOLUTION

void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN YOUR SOLUTION
  
  size_t m_size = M;
  size_t n_size = N;
  size_t p_size = P;


  for(size_t m = 0; m < m_size; ++m) {
    for(size_t p = 0; p < p_size; ++p) {
      size_t out_idx = m * p_size + p;
      // std::cout << "calculating out->ptr[(" << m << ", " << p << ") -> " << out_idx << "]..." << std::endl;

      // Do the reduction along n:
      scalar_t val = 0;
      for(size_t n = 0; n < n_size; ++n) {
        size_t a_idx = m * n_size + n;
        size_t b_idx = n * p_size + p;
        val += a.ptr[a_idx] * b.ptr[b_idx];
        // std::cout << "\tval += a.ptr[(" << m << ", " << n << ") -> " << a_idx 
        //           <<      "] * b.ptr[(" << n << ", " << p << ") -> " << b_idx << "] = " << val << std::endl;
      }
      
      // std::cout << "setting out->ptr[(" << m << ", " << p << ") -> " << out_idx << "] = " << val << std::endl;
      out->ptr[out_idx] = val;
    }
  }
  
  /// END YOUR SOLUTION
}

inline void AlignedDot(const float* __restrict__ a, 
                       const float* __restrict__ b, 
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement 
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and 
   * out don't have any overlapping memory (which is necessary in order for vector operations to be 
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b, 
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the 
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN YOUR SOLUTION 
   
  size_t m_size = TILE;
  size_t n_size = TILE;
  size_t p_size = TILE;

  for(size_t m = 0; m < m_size; ++m) {
    for(size_t p = 0; p < p_size; ++p) {
      size_t out_idx = m * p_size + p;
      // std::cout << "calculating out->ptr[(" << m << ", " << p << ") -> " << out_idx << "]..." << std::endl;

      scalar_t val = 0;
      for(size_t n = 0; n < n_size; ++n) {
        size_t a_idx = m * n_size + n;
        size_t b_idx = n * p_size + p;
        val += a[a_idx] * b[b_idx];
        // std::cout << "\tval += a[(" << m << ", " << n << ") -> " << a_idx 
        //           <<      "] * b[(" << n << ", " << p << ") -> " << b_idx << "] = " << val << std::endl;
      }
      // std::cout << "setting out[(" << m << ", " << p << ") -> " << out_idx << "] = " << out[out_idx] << " += " << val << std::endl;
      out[out_idx] += val;
      // std::cout << "updated out[(" << m << ", " << p << ") -> " << out_idx << "] = " << out[out_idx] << std::endl;
    }
  }
  
  /// END YOUR SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t M,
                 uint32_t N, uint32_t P) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   * 
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   * 
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   * 
   */
  /// BEGIN YOUR SOLUTION
  
  size_t m_size = M / TILE;
  size_t n_size = N / TILE;
  size_t p_size = P / TILE;

  Fill(out, 0);

  for(size_t m = 0, m_end = m_size; m < m_end; ++m) {
    for(size_t p = 0, p_end = p_size; p < p_end; ++p) {
      size_t out_idx = (m * p_size + p) * (TILE * TILE);
      // std::cout << "calculating out->ptr[(" << m << ", " << p << ") -> " << out_idx << "]..." << std::endl;

      // reduction along N:
      for(size_t n = 0, n_end = n_size; n < n_end; ++n) {
        size_t a_idx = (m * n_size + n) * (TILE * TILE);
        size_t b_idx = (n * p_size + p) * (TILE * TILE);
        // std::cout << "\tmultiplying tiles a.ptr[(" << m << ", " << n << ") -> " << a_idx 
        //           <<                 "] * b.ptr[(" << n << ", " << p << ") -> " << b_idx << "]" << std::endl;
        AlignedDot(&a.ptr[a_idx], &b.ptr[b_idx], &out->ptr[out_idx]);
      }
    }
  }
  
  /// END YOUR SOLUTION
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN YOUR SOLUTION
  assert(reduce_size >= 1 && "reduce_size assumed to be greater than 0");
  size_t in_idx = 0;
  for (size_t out_idx = 0; out_idx < out->size; ++out_idx) {
    scalar_t out_val = a.ptr[in_idx];
    for (size_t reduce_idx = 0; reduce_idx < reduce_size; ++reduce_idx) {
      out_val = std::max(out_val, a.ptr[in_idx++]);
    }
    out->ptr[out_idx] = out_val;
  }
  /// END YOUR SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN YOUR SOLUTION
  assert(reduce_size >= 1 && "reduce_size assumed to be greater than 0");
  size_t in_idx = 0;
  for (size_t out_idx = 0; out_idx < out->size; ++out_idx) {
    scalar_t out_val = 0;
    for (size_t reduce_idx = 0; reduce_idx < reduce_size; ++reduce_idx) {
      out_val += a.ptr[in_idx++];
    }
    out->ptr[out_idx] = out_val;
  }
  /// END YOUR SOLUTION
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);
  
  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
