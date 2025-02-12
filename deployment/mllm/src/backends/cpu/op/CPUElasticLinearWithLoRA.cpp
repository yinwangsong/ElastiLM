
#include "CPUElasticLinearWithLoRA.hpp"

#include "compute/MatmulElastic_F32_F16.hpp"
#include "compute/Matmul.hpp"
#include "compute/Arithmetic.hpp"

#include <cstring>

namespace mllm {

CPUElasticLinearWithLoRA::CPUElasticLinearWithLoRA(Backend *bn, string opName, int in_features, int out_features, bool bias, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    in_features_ = in_features;
    out_features_ = out_features;
    support_bias_ = bias;
    thread_count = threadCount;
    weight_.setBackend(bn);
    bias_.setBackend(bn);
    for (int i = 0; i < Elastilm::SUBMODEL_NUM; i++) {
        loras_a_.push_back(Tensor(bn));
        loras_b_.push_back(Tensor(bn));
    }
}

ErrorCode CPUElasticLinearWithLoRA::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
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

ErrorCode CPUElasticLinearWithLoRA::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout << name() << "  CPUElasticLinear  execute" << std::endl;
    int activate_input_dim = (int)inputs[1]->dataAt<float>(0, 0, 0, 0);
    int activate_output_dim = (int)inputs[2]->dataAt<float>(0, 0, 0, 0);


    if (inputs[0]->count() == 0) {
        return Op::execute(inputs, outputs);
    }

    if (weight_.dtype() == MLLM_TYPE_F16) {
        // std::cout << name() << "  mm1" << std::endl;
        mat_mul(inputs[0].get(), &loras_a_[Elastilm::LEVEL], &Elastilm::inner_rank_buffer, false, nullptr, false, true, thread_count);
        // Elastilm::inner_rank_buffer.printDataTorchLike<float>();
        // if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        //     exit(-1);
        // }

        // loras_b_[Elastilm::LEVEL].printDataTorchLike<float>();

        mat_mul(&Elastilm::inner_rank_buffer, &loras_b_[Elastilm::LEVEL], outputs[0].get(), false, nullptr, false, true, thread_count);


        // matmul does not need to clear the buffer C
        // std::memset(outputs[0].get()->ptrAt<float>(0, 0, 0, 0), 0, 4*outputs[0].get()->count());
        // outputs[0].get()->printDataTorchLike<float>();
        // std::cout<<outputs[0].get()->count()<<std::endl;
        // mllm_mul_fp32(outputs[0].get()->ptrAt<float>(0, 0, 0, 0), Elastilm::submodel_lora_scale, outputs[0].get()->ptrAt<float>(0, 0, 0, 0), outputs[0].get()->count());

        // outputs[0].get()->printDataTorchLike<float>();

        for (int b = 0; b < outputs[0].get()->batch(); b++){
            for (int h = 0; h < outputs[0].get()->head(); h++){
                for (int s = 0; s < outputs[0].get()->sequence(); s++){
                    for (int d = 0; d < outputs[0].get()->dimension(); d++){
                        outputs[0].get()->setDataAt<float>(b, h, s, d, outputs[0].get()->dataAt<float>(b, h, s, d) * Elastilm::submodel_lora_scale);
                    }
                }
            }
        }
        // std::cout<<outputs[0].get()->child_tensors_

        // outputs[0].get()->printDataTorchLike<float>();
        // if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        //     exit(-1);
        // }
        // we modify the mat_mul_elastic_f32_f16 to merge the lora output to the mm output
        mat_mul_elastic_f32_f16(inputs[0].get(), &weight_, outputs[0].get(), support_bias_, &bias_, activate_input_dim, activate_output_dim, false, true, thread_count);
        // inputs[0].get()->printDataTorchLike<float>();
        // weight_.printDataTorchLike<mllm_fp16_t>();
        // outputs[0].get()->printDataTorchLike<float>();
    } else {
        std::cout<<"Currently data type not supported!\n";
    }
    return Op::execute(inputs, outputs);
}

ErrorCode CPUElasticLinearWithLoRA::load(AbstructLoader &loader) {
    // std::cout << name() << "  CPUElasticLinear load" << std::endl;
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, out_features_, in_features_);
    if (loader.getDataType(weight_.name()) != MLLM_TYPE_COUNT) {
        weight_.setDtype(loader.getDataType(weight_.name()));
        weight_.alloc();
        loader.load(&weight_);
    } else {
        weight_.setDtype(MLLM_TYPE_F32);
        weight_.alloc();
    }
    // weight_.printDataTorchLike<mllm_fp16_t>();
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

    for (int i = 0; i < Elastilm::SUBMODEL_NUM; i++) {
        loras_a_[i].setName(name() + "_lora_a_level_" + std::to_string(i) + ".weight");
        loras_b_[i].setName(name() + "_lora_b_level_" + std::to_string(i) + ".weight");

        int in_features_lora;
        int out_features_lora;

        // std::cout<<Elastilm::LEVEL<<" "<<Elastilm::IS_ANCHOR_LAYER<<" "<<Elastilm::submodel_attn_hidden_dims[Elastilm::LEVEL][0]<<std::endl;
        
        if (weight_.name().find("attn") != std::string::npos) {
            if (weight_.name().find("q_proj") != std::string::npos || weight_.name().find("k_proj") != std::string::npos || weight_.name().find("v_proj") != std::string::npos) {
                in_features_lora = in_features_;
                out_features_lora = Elastilm::submodel_attn_hidden_dims[i][Elastilm::IS_ANCHOR_LAYER];
            }
            if (weight_.name().find("o_proj") != std::string::npos) {
                in_features_lora = Elastilm::submodel_attn_hidden_dims[i][Elastilm::IS_ANCHOR_LAYER];;
                out_features_lora = out_features_;
            }
        }

        if (weight_.name().find("mlp") != std::string::npos) {
            if (weight_.name().find("gate_proj") != std::string::npos || weight_.name().find("up_proj") != std::string::npos) {
                in_features_lora = in_features_;
                out_features_lora = Elastilm::submodel_mlp_hidden_dims[i][Elastilm::IS_ANCHOR_LAYER];
            }
            if (weight_.name().find("down_proj") != std::string::npos) {
                in_features_lora = Elastilm::submodel_mlp_hidden_dims[i][Elastilm::IS_ANCHOR_LAYER];;
                out_features_lora = out_features_;
            }
        }


        loras_a_[i].reshape(1, 1, Elastilm::RANK, in_features_lora);
        loras_b_[i].reshape(1, 1, out_features_lora, Elastilm::RANK);

        // std::cout<<in_features_lora<<" "<<out_features_lora<<std::endl;

        if (loader.getDataType(loras_a_[i].name()) != MLLM_TYPE_COUNT) {
            loras_a_[i].setDtype(loader.getDataType(loras_a_[i].name()));
            loras_a_[i].alloc();
            loader.load(&loras_a_[i]);
        } else {
            loras_a_[i].setDtype(MLLM_TYPE_F32);
            loras_a_[i].alloc();
        }

        if (loader.getDataType(loras_b_[i].name()) != MLLM_TYPE_COUNT) {
            loras_b_[i].setDtype(loader.getDataType(loras_b_[i].name()));
            loras_b_[i].alloc();
            loader.load(&loras_b_[i]);
        } else {
            loras_b_[i].setDtype(MLLM_TYPE_F32);
            loras_b_[i].alloc();
        }
    }

    return Op::load(loader);
}

ErrorCode CPUElasticLinearWithLoRA::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    if (support_bias_) {
        bias_.free();
    }
    for (int i = 0; i < Elastilm::SUBMODEL_NUM; i++) {
        loras_a_[i].free();
        loras_b_[i].free();
    }
    return Op::free(inputs, outputs);
}

} // namespace mllm

