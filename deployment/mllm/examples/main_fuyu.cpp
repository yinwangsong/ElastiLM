#include <vector>
#include <iostream>
#include <valarray>
#include <csignal>
#include "cmdline.h"
#include "Net.hpp"
#include "Executor.hpp"
#include "express/Express.hpp"
#include "tokenizers/BPE/Bpe.hpp"
#include "tokenizers/Unigram/Unigram.hpp"
#include "processor/FuyuPreProcess.hpp"

using namespace std;

void fullTensor(shared_ptr<Tensor> input_tensor, Net &net, vector<int> shape) {
    input_tensor->setBackend(net.backends()[BackendType::MLLM_CPU].get());
    input_tensor->reshape(shape[0], shape[1], shape[2], shape[3]);
    input_tensor->setDtype(MLLM_TYPE_F32);
    input_tensor->alloc();
    input_tensor->fullData<float>(1);
}

void patches2Tensor(shared_ptr<Tensor> input_tensor, Net &net, vector<vector<vector<float>>> image_patches) {
    if (image_patches.empty()) {
        fullTensor(input_tensor, net, {0, 0, 0, 0});
        return;
    }
    const int batch = image_patches.size();
    const int seq = image_patches[0].size();
    const int dims = image_patches[0][0].size();
    input_tensor->setBackend(net.backends()[BackendType::MLLM_CPU].get());
    input_tensor->reshape(batch, 1, seq, dims);
    input_tensor->setDtype(MLLM_TYPE_F32);
    input_tensor->alloc();
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < seq; ++j) {
            for (int k = 0; k < dims; ++k) {
                input_tensor->setDataAt<float>(i, 0, j, k, image_patches[i][j][k]);
            }
        }
    }
}

void patchIdx2Tensor(shared_ptr<Tensor> input_tensor, Net &net, vector<vector<int>> image_patches_indices) {
    if (image_patches_indices.empty()) {
        fullTensor(input_tensor, net, {0, 0, 0, 0});
        return;
    }
    const int batch = image_patches_indices.size();
    const int seq = image_patches_indices[0].size();
    input_tensor->setBackend(net.backends()[BackendType::MLLM_CPU].get());
    input_tensor->reshape(batch, 1, seq, 1);
    input_tensor->setDtype(MLLM_TYPE_F32);
    input_tensor->alloc();
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < seq; ++j) {
            input_tensor->setDataAt<float>(i, 0, j, 0, image_patches_indices[i][j]);
        }
    }
}

unsigned int argmax(const std::vector<float> &scores) {
    if (scores.empty()) {
        throw std::invalid_argument("Input vector is empty");
    }
    unsigned int maxIndex = 0;
    float maxValue = scores[0];
    for (size_t i = 1; i < scores.size(); ++i) {
        if (scores[i] > maxValue) {
            maxIndex = i;
            maxValue = scores[i];
        }
    }
    return maxIndex;
}
unsigned int postProcessing(shared_ptr<Tensor> result, shared_ptr<Tensor> &out_result) {
    assert(result->batch() == 1);
    assert(result->head() == 1);
    out_result->reshape(1, 1, 1, 1);
    out_result->alloc();
    vector<float> scores;
    for (int i = 0; i < result->dimension(); ++i) {
        auto value = result->dataAt<float>(0, 0, result->sequence() - 1, i);
        scores.push_back(value);
    }
    auto token_idx = argmax(scores);
    out_result->setDataAt<float>(0, 0, 0, 0, token_idx);
    return token_idx;
}

NetTensor *Attention(NetTensor *x, int embedding_size, int hidden_size, int head_size, int cache_max, string name) {
    x = _Linear({x}, embedding_size, hidden_size * head_size * 3, true, name + ".query_key_value");
    auto skv = _Split({x}, 3, Chl::D_HD, head_size, name + ".split");
    auto *q = skv[0];
    auto *k = skv[1];
    auto *v = skv[2];
    q = _LayerNorm({q}, hidden_size, true, 1e-6, name + ".q_layernorm");
    k = _LayerNorm({k}, hidden_size, true, 1e-6, name + ".k_layernorm");
    q = _RoPE({q}, PERSIMMONROPE, name + ".q_rope");
    k = _RoPE({k}, PERSIMMONROPE, name + ".k_rope");
    k = _KVCache({k}, cache_max, name + ".k_cache");
    v = _KVCache({v}, cache_max, name + ".v_cache");
    auto *qk = _Matmul({q, k}, false, true, name + ".qk");
    qk = _Scale({qk}, 1.0F / std::sqrt(head_size), 0.0F, false, name + ".scale");
    // qk = _Causalmask({qk}, name + ".mask");
    qk = _Softmax({qk}, DIMENSION, true, name + ".softmax");
    auto *o = _Matmul({qk, v}, false, false, name + ".qkv");
    o = o->view(-1, 1, -1, hidden_size * head_size);
    o = _Linear({o}, hidden_size * head_size, embedding_size, true, name + ".dense");
    return o;
}
NetTensor *MLP(NetTensor *i, int hidden_dim, int ffn_hidden_dim, string name) {
    auto *x = _Linear({i}, hidden_dim, ffn_hidden_dim, true, name + ".dense_h_to_4h");
    x = _ReLUSquaredActivation({x}, name + ".relu2");
    x = _Linear({x}, ffn_hidden_dim, hidden_dim, true, name + ".dense_4h_to_h");
    return x;
}
NetTensor *Persimmon(Context *c, NetTensor *i, int hidden_dim = 4096, int ffn_hidden_dim = 4096 * 4, int mutil_head_size = 64, int cache_max = 500, string name = "language_model.model") {
    // loop
    for (int layer = 0; layer < 36; ++layer) {
        auto *x = _LayerNorm({i}, hidden_dim, true, 1e-6, name + (string) ".layers." + std::to_string(layer) + ".input_layernorm");
        x = Attention(x, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, cache_max, name + (string) ".layers." + std::to_string(layer) + ".self_attn");
        i = _Add({x, i}, name + (string) ".layers." + std::to_string(layer) + ".add_attn");
        x = _LayerNorm({i}, hidden_dim, true, 1e-6, name + (string) ".layers." + std::to_string(layer) + ".post_attention_layernorm");
        x = MLP(x, hidden_dim, ffn_hidden_dim, name + (string) ".layers." + std::to_string(layer) + ".mlp");
        i = _Add({x, i}, name + (string) ".layers." + std::to_string(layer) + ".add_mlp");
        _SubgraphBegin(c);
    }
    // end loop
    i = _LayerNorm({i}, hidden_dim, true, 1e-6, name + (string) ".final_layernorm");
    return i;
}
void Fuyu(Context *c, int vocab_size = 262144, int patch_size = 30, int cnl_size = 3, int hidden_dim = 4096, int ffn_hidden_dim = 4096 * 4, int mutil_head_size = 32, int cache_max = 500) {
    auto *i = _Input(c, {}, "input_ids");
    i = _Embedding({i}, vocab_size, hidden_dim, (string) "language_model.model.embed_tokens");
    auto *p = _Input(c, {}, "image_patches");
    p = _Linear({p}, patch_size * patch_size * cnl_size, hidden_dim, true, "vision_embed_tokens");
    auto *id = _Input(c, {}, "image_patches_indices");
    i = _Gather({i, p, id}, "gather");
    i = Persimmon(c, i, hidden_dim, ffn_hidden_dim, mutil_head_size, cache_max, "language_model.model");
    i = _Linear({i}, hidden_dim, vocab_size, false, "language_model.lm_head");
}
int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/fuyu_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/fuyu-8b-q4_k.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 500);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    int thread_num = cmdParser.get<int>("thread");

    auto tokenizer = UnigramTokenizer(vocab_path);

    int vocab_size = 262144;
    int hidden_dim = 4096;
    int ffn_hidden_dim = 4096 * 4;
    int mutil_head_size = 64;
    int patch_size = 30;

    std::unique_ptr<Context> c_ptr(new Context());
    auto *c = c_ptr.get();
    Fuyu(c, vocab_size, patch_size, 3, hidden_dim, ffn_hidden_dim, mutil_head_size, tokens_limit);

    BackendConfig bn;
    Net net(bn);
    net.convert(c->sub_param_, BackendType::MLLM_CPU, thread_num);
    ParamLoader param_loader(model_path);
    Executor ex(&param_loader);
    ex.setup(&net);

    std::vector<vector<string>> in_imgs = {
        {"../assets/bus.png"},
        {"../assets/two_cats.jpg"}};
    vector<string> in_strs = {
        "Generate a coco-style caption.\n",
        "What's this?\n"};
    shared_ptr<Tensor> input_seq = std::make_shared<Tensor>();
    shared_ptr<Tensor> img_patch = std::make_shared<Tensor>();
    shared_ptr<Tensor> img_patch_id = std::make_shared<Tensor>();
    for (int inId = 0; inId < in_strs.size(); ++inId) {
        auto in_str = in_strs[inId];
        auto in_img = in_imgs[inId];
        auto preprocessor = FuyuPreProcess(&tokenizer);
        preprocessor.images_.clear();
        preprocessor.image_input_ids_.clear();
        preprocessor.image_patches_indices_.clear();
        preprocessor.image_patches_.clear();
        preprocessor.PreProcessImages(in_img);
        preprocessor.Process(in_str);
        auto input_ids = preprocessor.image_input_ids_;
        auto image_patches_indices = preprocessor.image_patches_indices_;
        auto image_patches = preprocessor.image_patches_;
        if (input_ids.empty()) {
            input_ids = preprocessor.text_ids_;
        }
        UnigramTokenizer::token2Tensor(&net, input_ids[0], input_seq);
        patches2Tensor(img_patch, net, image_patches);
        patchIdx2Tensor(img_patch_id, net, image_patches_indices);
        std::cout << "[Q] [";
        if (!in_img.empty()) {
            std::cout << in_img[0];
        }
        std::cout << "]" << in_str << std::endl;
        std::cout << "[A] " << std::flush;
        for (int step = 0; step < 50; step++) {
            ex.run(&net, {input_seq, img_patch, img_patch_id});
            auto result = ex.result();
            auto token_idx = postProcessing(result[0], input_seq);
            //        std::cout << token_idx << std::endl;
            if (token_idx == 71013) {
                break;
            }
            fullTensor(img_patch, net, {0, 0, 0, 0});
            fullTensor(img_patch_id, net, {0, 0, 0, 0});
            auto out_token = tokenizer.detokenize({token_idx});
            std::cout << out_token << std::flush;
        }
        printf("\n");
    }

    ex.perf();

    // free memory
    for (auto *op : c->net_ops) {
        delete op;
    }
    for (auto *tensor : c->net_tensors) {
        delete tensor;
    }
    return 0;
}