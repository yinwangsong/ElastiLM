//
// Created by Rongjie Yi on 2024/1/26 0026.
//

#include <iostream>
#include "cmdline.h"
#include "models/llama/modeling_elastic_llama_with_lora.hpp"
#include "models/llama/tokenization_llama.hpp"
#include "processor/PostProcess.hpp"
#include <memory/MemInspect.hpp>

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/llama2_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/llama-2-7b-chat-q4_0_4_4.mllm");
    cmdParser.add<string>("lora", 'a', "specify lora weights path", false, "../models/llama-2-7b-chat-q4_0_4_4-lora/");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.add<float>("ratio", 'r', "elasticize ratio", false, 1.0);
    cmdParser.add<int>("slos", 's', "SLO levels", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    string lora_path = cmdParser.get<string>("lora");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");
    float ratio = cmdParser.get<float>("ratio");
    int slo_levels = cmdParser.get<int>("slos");

    auto tokenizer = LLaMATokenizer(vocab_path);

    LLaMAConfig config(tokens_limit, "3B", HFHUBROPE);
    auto model = ElasticLLaMAModelWithLoRA(config);
    // std::cout<<Backend::global_backends.size()<<std::endl;
    // std::cout<<"loading"<<model_path<<std::endl;
    model.load(model_path, config);
    std::cout << mllm::physical_memory_used_by_process()/1024 << "MB" <<std::endl;
    std::cout << mllm::virtual_memory_used_by_process()/1024 << "MB" <<std::endl;

    // load all loras
    vector<vector<Tensor>> LoRA_family = {};
    vector<vector<vector<float>>> LoRA_scale_family = {};
    for (int i = 0; i < slo_levels; ++i) {
        vector<Tensor> LoRAs = {}; // 7*2*26
        for(int j = 0; j < 7*2*config.block_num; j++) {

            std::ifstream infile(lora_path+"lora_"+std::to_string(i)+"_"+std::to_string(j)+".raw", std::ios::binary);
            if (!infile) {
                std::cerr << lora_path << " lora reading error!" << std::endl;
                exit(-1);
            }

            std::vector<int32_t> shape(4);
            infile.read(reinterpret_cast<char*>(shape.data()), shape.size() * sizeof(int32_t));

            const size_t num_elements = shape[0] * shape[1] * shape[2] * shape[3];
            const size_t element_size = sizeof(float);

            std::vector<float> data(num_elements);

            infile.read(reinterpret_cast<char*>(data.data()), num_elements * element_size);
            if (!infile) {
                std::cerr << "reading error: cannot read all data!" << std::endl;
                exit(-1);
            }
            infile.close();

            Tensor tmp_lora(shape[0], shape[1], shape[2], shape[3], Backend::global_backends[MLLM_CPU], true);
            tmp_lora.setName("lora"+std::to_string(j));
            Tensor::tensor_status = TENSOR_STATIC_INIT;
            tmp_lora.setTtype(INPUT_TENSOR);
            // tmp_lora.setTtype(NORMAL_TENSOR);
            for (int b = 0; b < shape[0]; ++b) {
                for (int h = 0; h < shape[1]; ++h) {
                    for (int s = 0; s < shape[2]; ++s) {
                        for (int d = 0; d < shape[3]; ++d) {
                            tmp_lora.setDataAt<float>(b, h, s, d, data[b*shape[1]*shape[2]*shape[3] + h*shape[2]*shape[3] + s*shape[3] + d]);
                        }
                    }
                }
            }
            LoRAs.push_back(tmp_lora);
        }
        LoRA_family.push_back(LoRAs);

        vector<vector<float>> LoRAs_scales = {}; // 26
        for(int j = 0; j < config.block_num; j++) {
            std::ifstream infile(lora_path+"lora_scales_"+std::to_string(i)+"_"+std::to_string(j)+".raw", std::ios::binary);
            if (!infile) {
                std::cerr << "lora reading error!" << std::endl;
                exit(-1);
            }

            std::vector<int32_t> shape(4);
            infile.read(reinterpret_cast<char*>(shape.data()), shape.size() * sizeof(int32_t));

            const size_t num_elements = shape[0] * shape[1] * shape[2] * shape[3];
            const size_t element_size = sizeof(float);

            std::vector<float> data(num_elements);

            infile.read(reinterpret_cast<char*>(data.data()), num_elements * element_size);
            if (!infile) {
                std::cerr << "reading error: cannot read all data!" << std::endl;
                exit(-1);
            }
            infile.close();
            LoRAs_scales.push_back(data);
        }
        LoRA_scale_family.push_back(LoRAs_scales);
    }

    vector<string> in_strs = {
        // " Hello, who are you?",
        // " What can you do?",
        "Please introduce Beijing University of Posts and Telecommunications."};

    for (int i = 0; i < in_strs.size(); ++i) {
        auto in_str = in_strs[i];
        // auto input_tensor = tokenizer.tokenize(in_str);
        std::cout << mllm::physical_memory_used_by_process()/1024 << "MB" <<std::endl;
        std::cout << mllm::virtual_memory_used_by_process()/1024 << "MB" <<std::endl;
        std::cout << "[Q] " << in_strs[i] << std::endl;
        std::cout << "[A] " << std::flush;
        vector<token_id_t> x = {1,  3820,  9630, 11285,  1246,   287, 18123,   291, 10277, 17658, 31843};
        // for (int i = 0;i < 512; i++){
        //     x.push_back(1);
        // }
        Tensor input_tensor = LLaMATokenizer::tokens2Input(x);

        for (int step = 0; step < 1; step++) {
            float ratio = 0.5; // 0.25; //0.5;
            vector<vector<int>> activate_dims = {
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 0
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 1
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 2
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 3
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 4
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 5
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 6
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 7
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 8
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 9
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 10
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 11
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 12
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 13
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 14
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 15
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 16
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 17
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 18
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 19
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 20
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 21
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 22
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 23
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 24
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 25
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 26
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 27
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 28
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 29
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}, // 30
                {(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)}  // 31
            };
            // vector<vector<int>>
            vector<Tensor> inputs = {input_tensor};
            inputs.insert(inputs.end(), LoRA_family[0].begin(), LoRA_family[0].end());
            inputs[1].printDataTorchLike<float>();
            auto result = model(inputs, activate_dims, LoRA_scale_family[0]);
            result[0].printDataTorchLike<float>();
            // auto [out_string, out_token] = tokenizer.detokenize(result[0]);
            // auto [not_end, output_string] = tokenizer.postprocess(out_string);
            // if (!not_end) { break; }
            // std::cout << output_string << std::flush;
            // chatPostProcessing(out_token, input_tensor, {});
        }
        printf("\n");
        model.clear_kvcache();
    
    }

    return 0;
}