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
    cmdParser.add<string>("lora", 'a', "specify mllm lora path", false, "../models/llama-2-7b-chat-q4_0_4_4.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.add<float>("ratio", 'r', "elasticize ratio", false, 1.0);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    string lora_path = cmdParser.get<string>("lora");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");
    float ratio = cmdParser.get<float>("ratio");

    auto tokenizer = LLaMATokenizer(vocab_path);

    LLaMAConfig config(tokens_limit, "3B", HFHUBROPE);
    Elastilm::submodel_attn_hidden_dims = {
        {(int)(config.head_size * 1.0) * (int)(config.hidden_dim/config.head_size), (int)(config.head_size * 1.0) * (int)(config.hidden_dim/config.head_size)}, // 0 level
        {(int)(config.head_size * 0.9) * (int)(config.hidden_dim/config.head_size), (int)(config.head_size * 1.0) * (int)(config.hidden_dim/config.head_size)}, // 1
        {(int)(config.head_size * 0.8) * (int)(config.hidden_dim/config.head_size), (int)(config.head_size * 1.0) * (int)(config.hidden_dim/config.head_size)}, // 2
        {(int)(config.head_size * 0.7) * (int)(config.hidden_dim/config.head_size), (int)(config.head_size * 1.0) * (int)(config.hidden_dim/config.head_size)}, // 3
        {(int)(config.head_size * 0.6) * (int)(config.hidden_dim/config.head_size), (int)(config.head_size * 1.0) * (int)(config.hidden_dim/config.head_size)}, // 4
        {(int)(config.head_size * 0.5) * (int)(config.hidden_dim/config.head_size), (int)(config.head_size * 1.0) * (int)(config.hidden_dim/config.head_size)}, // 5
        {(int)(config.head_size * 0.4) * (int)(config.hidden_dim/config.head_size), (int)(config.head_size * 1.0) * (int)(config.hidden_dim/config.head_size)}, // 6
        {(int)(config.head_size * 0.3) * (int)(config.hidden_dim/config.head_size), (int)(config.head_size * 1.0) * (int)(config.hidden_dim/config.head_size)}, // 7
        {(int)(config.head_size * 0.2) * (int)(config.hidden_dim/config.head_size), (int)(config.head_size * 1.0) * (int)(config.hidden_dim/config.head_size)}, // 8
        {(int)(config.head_size * 0.1) * (int)(config.hidden_dim/config.head_size), (int)(config.head_size * 1.0) * (int)(config.hidden_dim/config.head_size)}, // 9
    };

    // std::cout<<Elastilm::submodel_attn_hidden_dims

    Elastilm::submodel_mlp_hidden_dims = {
        {(int)(config.ffn_hidden * 1.0), (int)(config.ffn_hidden * 1.0)}, // 0 level
        {(int)(config.ffn_hidden * 0.9), (int)(config.ffn_hidden * 1.0)}, // 1
        {(int)(config.ffn_hidden * 0.8), (int)(config.ffn_hidden * 1.0)}, // 2
        {(int)(config.ffn_hidden * 0.7), (int)(config.ffn_hidden * 1.0)}, // 3
        {(int)(config.ffn_hidden * 0.6), (int)(config.ffn_hidden * 1.0)}, // 4
        {(int)(config.ffn_hidden * 0.5), (int)(config.ffn_hidden * 1.0)}, // 5
        {(int)(config.ffn_hidden * 0.4), (int)(config.ffn_hidden * 1.0)}, // 6
        {(int)(config.ffn_hidden * 0.3), (int)(config.ffn_hidden * 1.0)}, // 7
        {(int)(config.ffn_hidden * 0.2), (int)(config.ffn_hidden * 1.0)}, // 8
        {(int)(config.ffn_hidden * 0.1), (int)(config.ffn_hidden * 1.0)}, // 9
    };

    Elastilm::submodel_lora_scale = 2.0;

    Elastilm::anchor_layers = {0, 1, 12, 14};
    Elastilm::layers_order = {};
    for(int id = 0; id < config.block_num; id++) {
        if (Elastilm::anchor_layers.find(id) != Elastilm::anchor_layers.end()) {
            continue;
        } else {
            Elastilm::layers_order.push_back(id);
        }
    }
    Elastilm::layers_order.insert(Elastilm::layers_order.end(), Elastilm::anchor_layers.begin(), Elastilm::anchor_layers.end());

    auto model = ElasticLLaMAModelWithLoRA(config);
    model.load_from_multifiles({model_path, lora_path});


    Elastilm::LEVEL = 9;
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
        input_tensor.printData<float>();

        Elastilm::inner_rank_buffer = Tensor(Backend::global_backends[MLLM_CPU]);
        Elastilm::inner_rank_buffer.setName("lora_inner_rank_buffer");
        Elastilm::inner_rank_buffer.reshape(1, 1, input_tensor.sequence(), Elastilm::RANK);
        Elastilm::inner_rank_buffer.setDtype(MLLM_TYPE_F32);
        Elastilm::inner_rank_buffer.alloc();

        for (int step = 0; step < 1; step++) {
            float ratio = float(Elastilm::SUBMODEL_NUM - Elastilm::LEVEL) / 10.0f;
            vector<vector<int>> activate_dims = {};
            for (int layer_id = 0; layer_id < config.block_num; layer_id++) {
                if (Elastilm::anchor_layers.find(layer_id) == Elastilm::anchor_layers.end()) {
                    // std::cout<<ratio<<"\n";
                    activate_dims.push_back({(int)(config.head_size * ratio), (int)(config.ffn_hidden * ratio)});
                } else {
                    activate_dims.push_back({(int)(config.head_size * 1.0), (int)(config.ffn_hidden * 1.0)});
                }
            }
            auto result = model({input_tensor}, activate_dims);
            result[0].printDataTorchLike<float>();
            auto [out_string, out_token] = tokenizer.detokenize(result[0]);
            auto [not_end, output_string] = tokenizer.postprocess(out_string);
            if (!not_end) { break; }
            std::cout << output_string << std::flush;
            chatPostProcessing(out_token, input_tensor, {});
        }
        printf("\n");
        model.clear_kvcache();
    
    }

    return 0;
}