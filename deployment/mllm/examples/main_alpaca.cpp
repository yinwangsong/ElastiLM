#include <iostream>
#include <valarray>
#include <csignal>
#include "cmdline.h"
#include "Net.hpp"
#include "Executor.hpp"
#include "express/Express.hpp"
#include "tokenizers/BPE/Bpe.hpp"
using namespace mllm;

unsigned int argmax(const std::vector<float>& scores) {
    if(scores.empty()) {
        throw std::invalid_argument("Input vector is empty");
    }
    unsigned int maxIndex = 0;
    float maxValue = scores[0];
    for(size_t i = 1; i < scores.size(); ++i) {
        if(scores[i] > maxValue) {
            maxIndex = i;
            maxValue = scores[i];
        }
    }
    return maxIndex;
}
unsigned int postProcessing(shared_ptr<Tensor> result, shared_ptr<Tensor>& out_result){
    assert(result->batch() == 1);
    assert(result->head() ==  1);
    out_result->reshape(1, 1, 1, 1);
    out_result->alloc();
    vector<float> scores;
    for (int i = 0; i < result->dimension(); ++i) {
        auto value = result->dataAt<float>(0, 0, result->sequence()-1, i);
        scores.push_back(value);
    }
    auto token_idx =  argmax(scores);
    out_result->setDataAt<float>(0, 0, 0, 0, token_idx);
    return token_idx;
}


NetTensor *Attention( NetTensor * x, int embedding_size, int hidden_size, int head_size, int cache_max, string name){
    auto *q =_Linear({x}, embedding_size, hidden_size * head_size, false, name + ".q_proj");
    auto *k =_Linear({x}, embedding_size, hidden_size * head_size, false, name + ".k_proj");
    auto *v =_Linear({x}, embedding_size, hidden_size * head_size, false, name + ".v_proj");
    q = q->view(-1, head_size, -1, hidden_size);
    k = k->view(-1, head_size, -1, hidden_size);
    v = v->view(-1, head_size, -1, hidden_size);
    q = _RoPE( {q}, HFHUBROPE, name + ".q_rope");
    k = _RoPE( {k}, HFHUBROPE, name + ".k_rope");
    k = _KVCache( {k}, cache_max,  name + ".k_cache");
    v = _KVCache( {v}, cache_max, name + ".v_cache");
    auto *qk = _Matmul( {q, k}, false, true, name + ".qk");
    qk = *qk/std::sqrt(hidden_size);
    // qk = _Causalmask( {qk}, name + ".mask");
    qk = _Softmax( {qk}, DIMENSION, true, name + ".softmax");
    auto *o = _Matmul( {qk, v}, false, false, name + ".qkv");
    o = o->view(-1, 1, -1, hidden_size * head_size);
    o = _Linear( {o}, hidden_size * head_size, embedding_size, false, name + ".o_proj");
    return o;
}
NetTensor *FFN( NetTensor * i, int hidden_dim, int ffn_hidden_dim, string name){
    auto *x = _Linear( {i}, hidden_dim, ffn_hidden_dim, false, name+".gate_proj");
    x = _SiLU( {x}, name+".silu");
    auto *y = _Linear( {i}, hidden_dim, ffn_hidden_dim, false, name+".up_proj");
    x = *x*y;// x = _Mul( {x, y}, name+".dot");
    x = _Linear( {x}, ffn_hidden_dim, hidden_dim, false, name+".down_proj");
    return x;
}
void llama(Context* c, int vocab_size= 55296, int hidden_dim= 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32, int cache_max = 200){
    auto *i = _Input(c);
    i = _Embedding( {i}, vocab_size, hidden_dim, "model.embed_tokens");
    // loop
    for(int layer=0; layer<32; ++layer) {
        auto *x = _RMSNorm( {i}, hidden_dim, 1e-6, "model.layers."+std::to_string(layer)+".input_layernorm");
        i = *Attention( x, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, cache_max, "model.layers."+std::to_string(layer)+".self_attn") +i;
        x = _RMSNorm( {i}, hidden_dim, 1e-6, "model.layers."+std::to_string(layer)+".post_attention_layernorm");
        i = *FFN( x, hidden_dim, ffn_hidden_dim, "model.layers."+std::to_string(layer) +".mlp") +i;
        //_SubgraphBegin(c);
    }
    // end loop
    i = _RMSNorm( {i}, hidden_dim, 1e-6, "model.norm");
    i = _Linear( {i}, hidden_dim, vocab_size, false, "lm_head");
}
int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/chinese-alpaca_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/chinese-alpaca-7b-q4_k.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    // string in_str = cmdParser.get<string>("input");
    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int token_limit = cmdParser.get<int>("limits");
    int thread_num = cmdParser.get<int>("thread");

    auto tokenizer = BPETokenizer(vocab_path);

    int vocab_size = 55296;
    int hidden_dim = 4096;
    int ffn_hidden_dim = 11008;
    int mutil_head_size = 32;

    std::unique_ptr<Context> c_ptr(new Context());
    auto *c = c_ptr.get();
    llama(c, vocab_size, hidden_dim, ffn_hidden_dim, mutil_head_size, token_limit);

    BackendConfig bn;
    Net net(bn);
    net.convert(c->sub_param_, BackendType::MLLM_CPU, thread_num);

    ParamLoader param_loader(model_path);
    Executor ex(&param_loader);
    ex.setup(&net);


    vector<string> in_strs = {
        " 介绍北京邮电大学。",
        " 介绍北京。",
    };
    shared_ptr<Tensor> input = std::make_shared<Tensor>();
    for (int str_i = 0; str_i < in_strs.size(); ++str_i)
    {
        auto in_str = in_strs[str_i];
        if(in_str[0] != ' '){
            in_str = ' '+ in_str;
        }
        auto tokens_id = vector<token_id_t>();
        tokenizer.tokenize(in_str, tokens_id, true);
        if(str_i > 0) {
            tokens_id[0] = 13;
        }
        BPETokenizer::token2Tensor( &net, tokens_id, input);
        std::cout <<"[Q] "<< in_str << std::endl;
        std::cout <<"[A] "<< std::flush;
        for(int step = 0; step<100; step++) {
            ex.run(&net, {input});
            auto result = ex.result();
            auto token_idx = postProcessing(result[0], input);
            if(token_idx == 2){// "</s>"
                break;
            }
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
