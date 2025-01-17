//
// Created by Wangsong Yin on 25-1-13.
//

#ifndef CPUMOBILEBERTCONVEMBEDFUNC_HPP
#define CPUMOBILEBERTCONVEMBEDFUNC_HPP
#include "Tensor.hpp"
#include "Types.hpp"

namespace mllm {
class Tensor;

class CPUMobileBertConvEmbedFunc : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {

        int batch = inputs[0]->batch();
        int head = inputs[0]->head();
        int sequence = inputs[0]->sequence();
        int dimension = inputs[0]->dimension();

        outputs[0]->reshape(batch, head, sequence, 3*dimension);
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {        
        auto src = inputs[0];
        auto dst = outputs[0];

        // #pragma omp parallel for collapse(4) num_threads(thread_count)
        for (int64_t b = 0; b < dst->batch(); b++) {
            for (int64_t h = 0; h < dst->head(); h++) {
                for (int64_t s = 0; s < dst->sequence(); s++) {
                    for (int64_t d = src->dimension(); d < 2*src->dimension(); d++) {
                        if(s-1 >= 0)
                            dst->setDataAt<float>(b, h, s-1, d-src->dimension(), src->dataAt<float>(b, h, s, d-src->dimension()));
                        dst->setDataAt<float>(b, h, s, d, src->dataAt<float>(b, h, s, d-src->dimension()));
                        if(s+1 < dst->sequence())
                            dst->setDataAt<float>(b, h, s+1, d+src->dimension(), src->dataAt<float>(b, h, s, d-src->dimension()));
                        // std::cout<<src->dataAt<float>(b, h, s, d)<<std::endl;
                    }
                }
            }
        }
    }
};

} // namespace mllm
#endif // CPUMOBILEBERTCONVEMBEDFUNC_HPP