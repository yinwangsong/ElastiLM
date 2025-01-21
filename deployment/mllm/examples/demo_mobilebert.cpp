//
// Created by yws on 25-1-12.
//
#include "models/mobilebert/configuration_mobilebert.hpp"
#include "models/mobilebert/modeling_mobilebert.hpp"
#include "models/mobilebert/tokenization_mobilebert.hpp"
#include "cmdline.h"
#include <vector>
#include <memory/MemInspect.hpp>

/*
 * an intent to support gte-small BertModel to do text embedding
 * current implementation is just a very basic example with a simple WordPiece tokenizer and a simple BertModel
 * not support batch embedding
 * */

int main(int argc, char *argv[]) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/mobilebert-uncased-fp16.mllm");
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/gte_vocab.mllm");
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string model_path = cmdParser.get<string>("model");
    string vocab_path = cmdParser.get<string>("vocab");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    MobileBertTokenizer tokenizer(vocab_path, true);
    auto config = MobileBertConfig();
    auto model = MobileBertModel(config);
    model.load(model_path);
    // model.setNoLoadWeightsDtype(MLLM_TYPE_F32);

    string text = "hello and what is your name aaa ?";
    vector<string> texts = {text};
    for (auto &text : texts) {
        auto inputs = tokenizer.tokenizes(text);
        auto res = model({inputs[0], inputs[1], inputs[2]})[0];
        res.printDataTorchLike<float>();
    }

    return 0;
}