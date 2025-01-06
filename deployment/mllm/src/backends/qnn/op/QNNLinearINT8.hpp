
#ifndef MLLM_QNNLINEARINT8_H
#define MLLM_QNNLINEARINT8_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNLinearINT8 : public QNNCommonOp {
public:
    QNNLinearINT8(Backend *bn, string opName, int in_features, int out_features, bool bias);
    virtual ~QNNLinearINT8() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int in_features_;
    int out_features_;
    bool support_bias_;
    Tensor weight_;
    Tensor bias_;
// #ifdef SMOOTHQUANT
    Tensor weightScale_;
    Tensor biasScale_;
// #endif
    Tensor outputScale_;
    Tensor inputScale_;
};

class QNNLinearINT8Creator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        int in_features = op_param["in_features"];
        int out_features = op_param["out_features"];
        int bias = op_param["bias"];
        return new QNNLinearINT8(bn, name, in_features, out_features, (bool)bias);
    }
};

} // namespace mllm

#endif
