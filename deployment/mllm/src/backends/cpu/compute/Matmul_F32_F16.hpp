#ifndef MLLM_MATMULF32F16_HPP
#define MLLM_MATMULF32F16_HPP

#include "VecDot.hpp"
using namespace mllm;

ErrorCode mat_mul_f32_f16(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias,
                  bool transpose0, bool transpose1, int thread_count);

#endif // MLLM_MATMULF32F16_HPP
