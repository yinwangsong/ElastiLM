
#ifndef MLLM_QNNDEQUANTIZE_H
#define MLLM_QNNDEQUANTIZE_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNDequantize : public QNNCommonOp {
public:
    QNNDequantize(Backend *bn, string opName, bool isNSHD, bool isFP32);
    virtual ~QNNDequantize() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
private:
    bool isNSHD_;
    bool isFP32_;
    Tensor scale_;
};

class QNNDequantizeCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNDequantize(bn, name, (bool)op_param["isNSHD"], (bool)op_param["isFP32"]);
    }
};

} // namespace mllm

#endif
