//
// Created by Rongjie Yi on 2024/1/26 0026.
//

#include <iostream>
#include "cmdline.h"
#include "models/llama/modeling_elastic_llama.hpp"
#include "models/llama/tokenization_llama.hpp"
#include "processor/PostProcess.hpp"
#include <memory/MemInspect.hpp>

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/llama2_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/llama-2-7b-chat-q4_0_4_4.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = LLaMATokenizer(vocab_path);

    LLaMAConfig config(tokens_limit, "3B", HFHUBROPE);
    auto model = ElasticLLaMAModel(config);
    model.load(model_path);

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
        for (int step = 0; step < 1; step++) {
            float ratio = 0.6; // 0.25; //0.5;
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