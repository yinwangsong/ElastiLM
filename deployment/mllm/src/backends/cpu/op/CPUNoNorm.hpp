
#ifndef MLLM_CPUNONORM_H
#define MLLM_CPUNONORM_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUNoNorm final : public Op {
public:
    CPUNoNorm(Backend *bn, string opName, int normSize, bool bias = true, int threadCount = 4);
    ~CPUNoNorm() override = default;
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode load(AbstructLoader &loader) override;
    ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
private:
    int thread_count = 4;
    int normSize_ = 0;
    Tensor weight_;
    Tensor bias_;
    bool bias;};

class CPUNoNormCreator : public CPUBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        bool bias = (bool)op_param["bias"];
        // std::cout<<"bias is "<<bias<<std::endl;
        int normSize = (int)op_param["norm_size"];
        return new CPUNoNorm(bn, name, normSize, bias, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUNONORM_H
