
#ifndef MLLM_MATMULF32F16_HPP
#define MLLM_MATMULF32F16_HPP

#include "VecDot.hpp"
using namespace mllm;

ErrorCode mat_mul_f32_f16(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, bool transpose0 = false, bool transpose1 = true, int thread_count = 4);

#endif // MLLM_MATMULF32F16_HPP
