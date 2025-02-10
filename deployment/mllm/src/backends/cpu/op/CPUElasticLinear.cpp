
#include "CPUElasticLinear.hpp"
#include "../compute/MatmulElastic.hpp"
#include "compute/MatmulElastic_F32_F16.hpp"

namespace mllm {

CPUElasticLinear::CPUElasticLinear(Backend *bn, string opName, int in_features, int out_features, bool bias, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    in_features_ = in_features;
    out_features_ = out_features;
    support_bias_ = bias;
    thread_count = threadCount;
    weight_.setBackend(bn);
    bias_.setBackend(bn);
}

ErrorCode CPUElasticLinear::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout << name() << "  CPUElasticLinear  reshape" << std::endl;
    // std::cout<<outputs[0].get()->dtype()<<" is before-reshape \n";
    assert(inputs.size() == 3);
    assert(outputs.size() == 1);
    int activate_input_dim = (int)inputs[1]->dataAt<float>(0, 0, 0, 0);
    int activate_output_dim = (int)inputs[2]->dataAt<float>(0, 0, 0, 0);
    if (inputs[0]->count() == 0) {
        outputs[0]->reshape(0, 0, 0, 0);
        return Op::reshape(inputs, outputs);
    }
    int in_dimension = (activate_input_dim == -1) ? in_features_ : activate_input_dim;
    int out_dimension = (activate_output_dim == -1) ? out_features_ : activate_output_dim;
    assert(inputs[0]->head() == 1);
    assert(in_dimension == inputs[0]->dimension());
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), out_dimension);
    // std::cout<<outputs[0].get()->dtype()<<" is after-reshape \n";
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUElasticLinear::load(AbstructLoader &loader) {
    // std::cout << name() << "  CPUElasticLinear load" << std::endl;
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, out_features_, in_features_);
    if (loader.getDataType(weight_.name()) != MLLM_TYPE_COUNT) {
        weight_.setDtype(loader.getDataType(weight_.name()));
        weight_.alloc();
        loader.load(&weight_);
        std::cout<<"loading the linear weights\n";
        weight_.printDataTorchLike<mllm_fp16_t>();
    } else {
        weight_.setDtype(MLLM_TYPE_F32);
        weight_.alloc();
    }
    if (support_bias_) {
        bias_.setName(name() + ".bias");
        bias_.reshape(1, 1, 1, out_features_);
        if (loader.getDataType(bias_.name()) != MLLM_TYPE_COUNT) {
            bias_.setDtype(loader.getDataType(bias_.name()));
            bias_.alloc();
            loader.load(&bias_);
        } else {
            bias_.setDtype(MLLM_TYPE_F32);
            bias_.alloc();
        }
    }
    return Op::load(loader);
}

ErrorCode CPUElasticLinear::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    int activate_input_dim = (int)inputs[1]->dataAt<float>(0, 0, 0, 0);
    int activate_output_dim = (int)inputs[2]->dataAt<float>(0, 0, 0, 0);

    //    auto start = mllm::mllm_time_us();
    if (inputs[0]->count() == 0) {
        return Op::execute(inputs, outputs);
    }
    // inputs[0].get()->printDataTorchLike<float>();
    std::cout<<"linear weights\n";
    weight_.printDataTorchLike<float>();
    if (weight_.dtype() == MLLM_TYPE_F16) {
        // std::cout<<outputs[0].get()->dtype()<<" is pre-kernel \n";
        // if (outputs[0].get()->name() == "out-model.layers.X.self_attn.k_proj") {
        //     std::cout<<"special flag";
        //     inputs[0].get()->printDataTorchLike<float>();
        //     weight_.printDataTorchLike<mllm_fp16_t>();
        //     outputs[0].get()->printDataTorchLike<float>();
        // }
        mat_mul_elastic_f32_f16(inputs[0].get(), &weight_, outputs[0].get(), support_bias_, &bias_, activate_input_dim, activate_output_dim, false, true, thread_count);
        // if (outputs[0].get()->name() == "out-model.layers.0.self_attn.v_proj") {
        //     std::cout<<"special flag";

        // }
        // outputs[0].get()->printDataTorchLike<float>();
        // std::cout<<"execute fp16"<<std::endl;
    } else {
        mat_mul_elastic(inputs[0].get(), &weight_, outputs[0].get(), support_bias_, &bias_, activate_input_dim, activate_output_dim, false, true, thread_count);
    }
    /*
    // std::cout << name() << "  CPUElasticLinear()" << std::endl;
    switch (weight_.dtype()) {
    case MLLM_TYPE_F32: {
        mat_mul_elastic_fp32(inputs[0].get(), &weight_, outputs[0].get(), support_bias_, &bias_, activate_input_dim,activate_output_dim, false, true, thread_count);
        break;
    }
    case MLLM_TYPE_F16: break;
    case MLLM_TYPE_Q4_0: {
        mat_mul_elastic_fp32_q4_0(inputs[0].get(), &weight_, outputs[0].get(), support_bias_, &bias_,  activate_input_dim,activate_output_dim, thread_count);
        break;
    }
    case MLLM_TYPE_Q4_K: {
        mat_mul_elastic_fp32_q4_K(inputs[0].get(), &weight_, outputs[0].get(), support_bias_, &bias_,  activate_input_dim,activate_output_dim, thread_count);
        break;
    }
    case MLLM_TYPE_Q6_K: {
        mat_mul_elastic_fp32_q6_K(inputs[0].get(), &weight_, outputs[0].get(), support_bias_, &bias_,  activate_input_dim,activate_output_dim, thread_count);
        break;
    }
    default:
        break;
    }
    */
    //    auto end = mllm::mllm_time_us();
    //    printf("exec time: %ld us\n", end - start);
    return Op::execute(inputs, outputs);
}
ErrorCode CPUElasticLinear::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    if (support_bias_) {
        bias_.free();
    }
    return Op::free(inputs, outputs);
}

} // namespace mllm
