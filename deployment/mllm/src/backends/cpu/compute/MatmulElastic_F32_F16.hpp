#ifndef MLLM_MATMULELASTIC_F32_F16_HPP
#define MLLM_MATMULELASTIC_F32_F16_HPP

#include "VecDot.hpp"
using namespace mllm;

ErrorCode mat_mul_elastic_f32_f16(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, int activate_input_dim = -1, int activate_output_dim = -1, bool transpose0 = false, bool transpose1 = true, int thread_count = 4);

#endif // MLLM_MATMULELASTIC_F32_F16_HPP
