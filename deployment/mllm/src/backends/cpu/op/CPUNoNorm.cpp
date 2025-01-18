
#include "CPUNoNorm.hpp"

namespace mllm {

CPUNoNorm::CPUNoNorm(Backend *bn, string opName, int normSize, bool bias, int threadCount) : thread_count(threadCount), 
    Op(bn, opName), bias(bias) {
    normSize_ = normSize;
    weight_.setBackend(bn);
    if (bias) {
        bias_.setBackend(bn);
    }
}

ErrorCode CPUNoNorm::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout << name() << "  CPUNoNorm  reshape" << std::endl;
    assert(normSize_ == inputs[0]->dimension());
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUNoNorm::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    // std::cout << name() << "  CPUNoNorm  execute" << std::endl;
    auto input = inputs[0];
    auto output = outputs[0];
    int batch = input->batch();
    int dim = input->dimension();
    int seq = input->sequence();
    int head = input->head();

    // std::cout<<weight_.batch()<<" "<<weight_.head()<<" "<<weight_.sequence()<<" "<<weight_.dimension()<<std::endl;
    // weight_.printData0<float>();
    // bias_.printData0<float>();
    // std::cout<<inputs[0].get()->batch()<<" "<<inputs[0].get()->head()<<" "<<inputs[0].get()->sequence()<<" "<<inputs[0].get()->dimension()<<std::endl;
    // inputs[0].get()->printData0<float>();
    
    for (int h = 0; h < head; h++) {
        for (int n = 0; n < batch; n++) {
            for (int s = 0; s < seq; s++) {
                for (int d = 0; d < dim; d++) {
                    if (bias) {
                        output->setDataAt<float>(n, h, s, d, weight_.dataAt<float>(0, 0, 0, d) * input->dataAt<float>(n, h, s, d) + bias_.dataAt<float>(0, 0, 0, d));
                        // std::cout<<"cal"<<std::endl;
                    } else {
                        output->setDataAt<float>(n, h, s, d, weight_.dataAt<float>(0, 0, 0, d) * input->dataAt<float>(n, h, s, d));
                    }
                }
            }
        }
    }
    
    // std::cout<<outputs[0].get()->batch()<<" "<<outputs[0].get()->head()<<" "<<outputs[0].get()->sequence()<<" "<<outputs[0].get()->dimension()<<std::endl;
    // outputs[0].get()->printData0<float>();
    return Op::execute(inputs, outputs);
}

ErrorCode CPUNoNorm::load(AbstructLoader &loader) {
    // std::cout << name() << "  CPUNoNorm  load" << std::endl;
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, 1, normSize_); //
     if (loader.getDataType(weight_.name()) != MLLM_TYPE_COUNT) {
         weight_.setDtype(loader.getDataType(weight_.name()));
         weight_.alloc();
         loader.load(&weight_);
     } else {
         weight_.setDtype(MLLM_TYPE_F32);
         weight_.alloc();
     }
    //  std::cout<<bias<<std::endl;
    if (bias) {
        bias_.setName(name() + ".bias");
        bias_.reshape(1, 1, 1, normSize_); //
        if (loader.getDataType(bias_.name()) != MLLM_TYPE_COUNT) {
            bias_.setDtype(loader.getDataType(bias_.name()));
            bias_.alloc();
            loader.load(&bias_);
            // bias_.printData0<float>();
        } else {
            bias_.setDtype(MLLM_TYPE_F32);
            bias_.alloc();
        }
    }
    return Op::load(loader);
}

ErrorCode CPUNoNorm::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}

} // namespace mllm