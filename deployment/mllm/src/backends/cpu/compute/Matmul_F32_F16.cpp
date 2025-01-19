//
// Created by Rongjie Yi on 23-10-24.
//

#include "Matmul.hpp"
#include "Types.hpp"
#include "VecDotType.hpp"
#include "SGEMM_F32_F16.hpp"
#include <cassert>

#ifdef __ARM_NEON
#include <arm_neon.h>
#include <omp.h>
#endif

ErrorCode mat_mul_f32_f16(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias,
                  bool transpose0, bool transpose1, int thread_count) {
    // src1 = W  src0 = x
    // transpose0=false  transpose1=true

    const int M = transpose0 ? src0->dimension() : src0->sequence();
    const int K = transpose0 ? src0->sequence() : src0->dimension();
    const int N = transpose1 ? src1->sequence() : src1->dimension();

    auto src0_dtype = src0->dtype();
    auto src1_dtype = src1->dtype();

    auto src1_type_size = type_size(src1_dtype);
    auto src0_type_size = type_size(src0->dtype());

#ifdef LLAMAFILE_SGEMM
    int ld_src1 = src1->sequenceSkipDim();
    int ld_src0 = src0->sequenceSkipDim();
    int ld_dst = dst->sequenceSkipDim();
    if (check_llamafile_sgemm_f32_f16(N, M, K, src1->dtype(), src0->dtype(), dst->dtype(), ld_src1, ld_src0, ld_dst)
        && dst->aggregatedTensors().empty()) {
        int is_0 = (src1->batch() == 1 && src1->head() == 1 && src1->batch() != src0->batch()) ? 0 : 1;
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int64_t b = 0; b < dst->batch(); b++) {
            for (int64_t h = 0; h < dst->head(); h++) {
                for (int id = 0; id < thread_count; id++) {
                    std::cout<<"checked"<<std::endl;
                    src1->printDataTorchLike<mllm_fp16_t>();
                    llamafile_sgemm_f32_f16(
                        N, M, K,
                        (char *) src1->rawHostPtr() + src1->offset(b * is_0, h * is_0, 0, 0) * src1_type_size, // A
                        ld_src1,
                        (char *) src0->rawHostPtr() + src0->offset(b, h, 0, 0) * src0_type_size, // B
                        ld_src0,
                        (char *) dst->rawHostPtr() + dst->offset(b, h, 0, 0) * type_size(dst->dtype()), // C
                        ld_dst, 
                        id, thread_count, src1->dtype(), src0->dtype(), dst->dtype(),
                        /*bias=*/support_bias ? bias->hostPtr<float>() : nullptr,
                        /*BiasType=*/support_bias ? bias->dtype() : DataType::MLLM_TYPE_F32);
                }
            }
        }
        return MLLM_NO_ERROR;
    }

    return MLLM_NO_ERROR;
#endif
}
