
#ifndef MLLM_CPUMBBERTCOVEMBED_H
#define MLLM_CPUMBBERTCOVEMBED_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

// This is a specific op for mobilebert
// input: B, H, S, D, output: B, H, S, 3D
// [[[[t1, t2],       [[[[t3, t4, t1, t2, 0, 0],
//    [t3, t4],  -->     [t5, t6, t3, t4, t1, t2],
//    [t5, t6]]]]        [0, 0, t5, t6, t3, t4]]]]
class CPUMBBertCovEmbed final : public Op {
public:
    CPUMBBertCovEmbed(Backend *bn, string opName, int threadCount);
    ~CPUMBBertCovEmbed() override = default;
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
};

class CPUMBBertCovEmbedCreator : public CPUBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        return new CPUMBBertCovEmbed(bn, name, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUMBBERTCOVEMBED_H
