#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);
typedef ssize_t ptrdiff_t;

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
    // std::cerr << "Allocating new CudaArray of size " << size << ": " << this->ptr << std::endl;
  }
  ~CudaArray() { 
    // std::cerr << "Deallocating CudaArray: " << ptr << std::endl;
    cudaFree(ptr); 
  }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
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

CudaVec GetCompactStrides(std::vector<int32_t> const& shape) {
  CudaVec result;
  result.size = shape.size();

  int32_t multiple = 1;
  for(size_t i = 0; i < result.size; ++i) {
    size_t idx = (result.size - 1) - i;
    result.data[idx] = multiple;
    multiple *= shape[idx];
  }
  return result;
}

__device__ size_t GetSparseIdx(
    size_t compact_idx,
    CudaVec const& compact_strides,
    size_t sparse_offset,
    CudaVec const& sparse_strides) {

  int32_t sparse_idx = static_cast<int32_t>(sparse_offset);
  size_t size = compact_strides.size;
  for(size_t i = 0; i < size; ++i) {
    int32_t sparse_pos = static_cast<int32_t>(compact_idx) / compact_strides.data[i];
    sparse_idx += sparse_pos * sparse_strides.data[i];
    compact_idx %= compact_strides.data[i];
  }
  assert(sparse_idx >= 0 && "negative strides error");
  return static_cast<size_t>(sparse_idx);
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

__global__ void CompactKernel(const scalar_t* in, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec out_strides, CudaVec in_strides, size_t in_offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   in: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   in_strides: vector of strides of *in* array
   *   in_offset: offset of *in* array
   */

  /// BEGIN YOUR SOLUTION
  
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (gid < size) {
    // 'in' is sparse, 'out' is compact, and we're always iterating over the compact indexes:
    size_t out_idx = gid;
    size_t in_idx = GetSparseIdx(out_idx, out_strides, in_offset, in_strides);
    // printf("DEBUG[CompactKernel]: setting out[%lu]=in[%lu]=%f\n", 
    //   (long unsigned)out_idx, (long unsigned)in_idx, (float)in[in_idx]);
    out[out_idx] = in[in_idx];
  }

  /// END YOUR SOLUTION
}

void Compact(const CudaArray& in, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> in_strides, size_t in_offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   in: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   in_strides: strides of the *in* array (not out, which has compact strides)
   *   in_offset: offset of the *in* array (not out, which has zero offset, being compact)
   */

  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(in.ptr, out->ptr, out->size, VecToCuda(shape),
                                         GetCompactStrides(shape), VecToCuda(in_strides), in_offset);
}

////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseSetitemKernel(const scalar_t* in, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec in_strides, CudaVec out_strides, size_t out_offset) {
  
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    // 'in' is compact, 'out' is sparse, and we're always iterating over the compact indexes:
    size_t in_idx = gid;
    size_t out_idx = GetSparseIdx(in_idx, in_strides, out_offset, out_strides);
    out[out_idx] = in[in_idx];
    // printf("DEBUG[EwiseSetitemKernel]: setting out[%lu]=in[%lu]=%f\n", 
    //   (unsigned long)out_idx, (unsigned long)in_idx, (float)in[in_idx]);
  }
}

void EwiseSetitem(const CudaArray& in, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> out_strides, size_t out_offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  You will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   in: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   out_strides: strides of the *out* array (not in, which has compact strides)
   *   out_offset: offset of the *out* array (not in, which has zero offset, being compact)
   */
  /// BEGIN YOUR SOLUTION

  // std::cerr << "DEBUG[EwiseSetitem]: in.size=" << in.size << ", "
  //           << "out->size=" << out->size << ", "
  //           << "shape=" << ArrayToString(shape) << ", "
  //           << "out_strides=" << ArrayToString(out_strides) << ", "
  //           << "out_offset=" << out_offset << std::endl;

  CudaDims dim = CudaOneDim(out->size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(in.ptr, out->ptr, in.size, VecToCuda(shape),
                                         GetCompactStrides(shape), VecToCuda(out_strides), out_offset);
  
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////

__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec in_strides, CudaVec out_strides, size_t out_offset) {
  
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    // 'in' is a scalar here, 'out' is sparse, and we're always iterating over the compact indexes:
    size_t in_idx = gid;
    size_t out_idx = GetSparseIdx(in_idx, in_strides, out_offset, out_strides);
    out[out_idx] = val;
    // printf("DEBUG[EwiseSetitemKernel]: setting out[%lu]=val=%f\n", (long unsigned)out_idx, val);
  }
}


void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> out_strides, size_t out_offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   out_strides: strides of the out array
   *   out_offset: offset of the out array
   */
  /// BEGIN YOUR SOLUTION

  CudaDims dim = CudaOneDim(out->size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                         GetCompactStrides(shape), VecToCuda(out_strides), out_offset);

  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
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

// EwiseMul:

__global__ void EwiseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * b[gid];
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

// ScalarMul

__global__ void ScalarMulKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * val;
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

// EwiseDiv:

__global__ void EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / b[gid];
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

// ScalarDiv

__global__ void ScalarDivKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / val;
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

// ScalarPower

__global__ void ScalarPowerKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = pow(a[gid], val);
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

// EwiseMaximum

__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = a[gid] > b[gid] ? a[gid] : b[gid];
  }
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

// ScalarMaximum

__global__ void ScalarMaximumKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = a[gid] > val ? a[gid] : val;
  }
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

// EwiseEq

__global__ void EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = a[gid] == b[gid] ? 1 : 0;
  }
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}


// ScalarEq

__global__ void ScalarEqKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = a[gid] == val ? 1 : 0;
  }
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

// EwiseGe

__global__ void EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = a[gid] >= b[gid] ? 1 : 0;
  }
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

// ScalarGe

__global__ void ScalarGeKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = a[gid] >= val ? 1 : 0;
  }
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

// EwiseLog

__global__ void EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = log(a[gid]);
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

// EwiseExp

__global__ void EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = exp(a[gid]);
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

// EwiseTanh

__global__ void EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = tanh(a[gid]);
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

/// END YOUR SOLUTION

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

//// Reference single-threaded solution:
// size_t m_size = M;
// size_t n_size = N;
// size_t p_size = P;

// for(size_t m = 0; m < m_size; ++m) {
//   for(size_t p = 0; p < p_size; ++p) {
//     size_t out_idx = m * p_size + p;
//     std::cerr << "calculating out->ptr[(" << m << ", " << p << ") -> " << out_idx << "]..." << std::endl;

//     // Do the reduction along n:
//     scalar_t val = 0;
//     for(size_t n = 0; n < n_size; ++n) {
//       size_t a_idx = m * n_size + n;
//       size_t b_idx = n * p_size + p;
//       val += a.ptr[a_idx] * b.ptr[b_idx];
//       std::cerr << "\tval += a.ptr[(" << m << ", " << n << ") -> " << a_idx 
//                 <<      "] * b.ptr[(" << n << ", " << p << ") -> " << b_idx << "] = " << val << std::endl;
//     }
    
//     std::cerr << "setting out->ptr[(" << m << ", " << p << ") -> " << out_idx << "] = " << val << std::endl;
//     out->ptr[out_idx] = val;
//   }
// }

__global__ void MatmulKernel(
    const scalar_t* a, const scalar_t* b, scalar_t* out, size_t out_size,
    int32_t M, int32_t N, int32_t P
) {

  // a: compact 2D array of size M x N
  // b: comapct 2D array of size N x P
  // out: compact 2D array of size M x P to write the output to

  size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (out_idx < out_size) {
    int32_t m = out_idx / P;
    int32_t p = out_idx % P;

    // Do the reduction along N:
    scalar_t val = 0;
    for(int32_t n = 0; n < N; ++n) {
      int32_t a_idx = m * N + n;
      int32_t b_idx = n * P + p;
      val += a[a_idx] * b[b_idx];
      // printf("\tval += a[(%lu, %lu) -> %lu] * b[(%lu, %lu) -> %lu] = %f\n",
      //   static_cast<unsigned long>(m),
      //   static_cast<unsigned long>(n),
      //   static_cast<unsigned long>(a_idx),
      //   static_cast<unsigned long>(n),
      //   static_cast<unsigned long>(p),
      //   static_cast<unsigned long>(b_idx),
      //   val
      // );
    }

    // printf("DEBUG[Matmul]: out_idx=%lu, out[%lu, %lu] = %f\n", 
    //   static_cast<unsigned long>(out_idx),
    //   static_cast<unsigned long>(m),
    //   static_cast<unsigned long>(p),
    //   val
    // );
    out[out_idx] = val;
  }
}


void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, 
            int32_t M, int32_t N, int32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size M x N
   *   b: comapct 2D array of size N x P
   *   out: compact 2D array of size M x P to write the output to
   *   M: rows of 'a', rows of 'out'
   *   N: columns of 'a', rows of 'b'
   *   P: columns of 'b', rows of 'out'
   */

  /// BEGIN YOUR SOLUTION

  // std::cerr << "DEBUG[Matmul]: a.size = " << a.size << std::endl;
  // std::cerr << "DEBUG[Matmul]: b.size = " << b.size << std::endl;
  // std::cerr << "DEBUG[Matmul]: out->size = " << out->size << std::endl;
  // std::cerr << "DEBUG[Matmul]: M=" << M << ", N=" << N << ", P=" << P << std::endl;

  CudaDims dim = CudaOneDim(out->size);
  MatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, M, N, P);
  
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Tiled version of Matmul
////////////////////////////////////////////////////////////////////////////////

// void MatmulTiled(const CudaArray& a, const CudaArray& b, CudaArray* out, 
//             int32_t M, int32_t N, int32_t P) {

//   std::cerr << "DEBUG[MatmulTiled]: calling Matmul..." << std::endl;
//   Matmul(a, b, out, M, N, P);
// }

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t out_size, size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < out_size) {
    size_t out_idx = gid;
    size_t in_idx = gid * reduce_size;
    // printf("DEBUG[ReduceMaxKernel]: gid=%lu: reducnig in[%lu:%lu] -> out[%lu]\n",
    //   static_cast<unsigned long>(gid),
    //   static_cast<unsigned long>(gid * reduce_size),
    //   static_cast<unsigned long>((gid + 1) * reduce_size - 1),
    //   static_cast<unsigned long>(out_idx)
    // );
    scalar_t val = a[in_idx];
    for(size_t i = 1; i < reduce_size; ++i) {
      val = max(val, a[in_idx+i]);
    }
    out[out_idx] = val;
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION

  // std::cerr << "DEBUG[ReduceMax]:" << std::endl;
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  // std::cerr << "DEBUG[ReduceMax]: done." << std::endl;
  
  /// END YOUR SOLUTION
}


__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t out_size, size_t reduce_size) {
  size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (out_idx < out_size) {
    scalar_t val = 0;
    size_t in_idx_end = (out_idx + 1) * reduce_size;
    for(size_t in_idx = out_idx * reduce_size; in_idx < in_idx_end; ++in_idx) {
      val += a[in_idx];
    }
    out[out_idx] = val;
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION

  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  
  /// END YOUR SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
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
  // m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
