
#ifndef MLLM_CPUELASTICLINEARWITHLORA_H
#define MLLM_CPUELASTICLINEARWITHLORA_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUElasticLinearWithLoRA final : public Op {
public:
    CPUElasticLinearWithLoRA(Backend *bn, string opName, int in_features, int out_features, bool bias, int threadCount);
    ~CPUElasticLinearWithLoRA() override = default;
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode load(AbstructLoader &loader) override;
    ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int in_features_;
    int out_features_;
    bool support_bias_;
    int thread_count = 4;
    Tensor weight_;
    Tensor bias_;

    vector<Tensor> loras_a_;
    vector<Tensor> loras_b_;
};

class CPUElasticLinearWithLoRACreator : public CPUBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        int in_features = op_param["in_features"];
        int out_features = op_param["out_features"];
        int bias = op_param["bias"];
        return new CPUElasticLinearWithLoRA(bn, name, in_features, out_features, (bool)bias, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUELASTICLINEARWITHLORA_H
