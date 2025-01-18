
#include "CPUMBBertCovEmbed.hpp"

namespace mllm {

CPUMBBertCovEmbed::CPUMBBertCovEmbed(Backend *bn,  string opName, int threadCount) : thread_count(threadCount), 
    Op(bn, opName) {
}

ErrorCode CPUMBBertCovEmbed::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), 3*inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUMBBertCovEmbed::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto src = inputs[0];
    auto dst = outputs[0];

    #pragma omp parallel for collapse(4) num_threads(thread_count)
    for (int64_t b = 0; b < dst->batch(); b++) {
        for (int64_t h = 0; h < dst->head(); h++) {
            for (int64_t s = 0; s < dst->sequence(); s++) {
                for (int64_t d = 0; d < src->dimension(); d++) {
                    if(s-1 >= 0)
                        dst->setDataAt<float>(b, h, s-1, d-src->dimension(), src->dataAt<float>(b, h, s, d));
                    dst->setDataAt<float>(b, h, s, d, src->dataAt<float>(b, h, s, d));
                    if(s+1 < dst->sequence())
                        dst->setDataAt<float>(b, h, s+1, d+src->dimension(), src->dataAt<float>(b, h, s, d));
                }
            }
        }
    }
    return Op::execute(inputs, outputs);
}


ErrorCode CPUMBBertCovEmbed::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::setUp(inputs, outputs);
}
} // namespace mllm