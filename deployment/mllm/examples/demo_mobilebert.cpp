//
// Created by yws on 25-1-12.
//
#include "models/mobilebert/configuration_mobilebert.hpp"
// #include "models/mobilebert/modeling_mobilebert.hpp"
#include "models/mobilebert/modeling_dual_head_TLM.hpp"
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

    std::vector<std::string> PREFILL_SLO = {
        "[02]",
        "[03]",
        "[04]",
        "[05]",
        "[06]",
        "[07]",
        "[08]",
        "[09]",
        // "[10]"
    };

    std::vector<std::string> DECODE_SLO = {
        "<05>",
        "<06>",
        "<07>",
        "<08>",
        "<09>",
        "<10>"
    };


    MobileBertTokenizer tokenizer(vocab_path, true);
    auto config = MobileBertConfig();
    auto model = MobileBertModel(config);
    model.load(model_path);
    // model.setNoLoadWeightsDtype(MLLM_TYPE_F32);

    tokenizer.add_special_tokens_into_vocab(PREFILL_SLO);
    tokenizer.add_special_tokens_into_vocab(DECODE_SLO);

    string SYS_PROMPT = "";
    SYS_PROMPT += "You are a smart assistant that helps human sloving problems. You help them by answering questions.\n";
    SYS_PROMPT += "Examples: \nQuestion: Which factor will most likely cause a person to develop a fever? Answer: a bacterial population in the bloodstream.";
    SYS_PROMPT += "\nQuestion: Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship? Answer: food.";
    SYS_PROMPT += "\nQuestion: When a switch is used in an electrical circuit, the switch can Answer: stop and start the flow of current.";
    SYS_PROMPT += "\nQuestion: Which of the following is an example of an human is is is is";
    // SYS_PROMPT += "\nQuestion: Rocks are classified as igneous, metamorphic, or sedimentary according to Answer: how they formed.";

    string text = "[SEP] <10>" + SYS_PROMPT;
    vector<string> texts = {text};
    for (auto &text : texts) {
        auto inputs = tokenizer.tokenizes(text);
        inputs[0].printData<float>();
        auto res = model({inputs[0], inputs[1], inputs[2]});
        res[0].printDataTorchLike<float>();
        res[1].printDataTorchLike<float>();
        res[2].printDataTorchLike<float>();
    }

    return 0;
}