//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#ifndef MODELING_LLAMA_HPP
#define MODELING_LLAMA_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_llama.hpp"
#include <cassert>
#include <vector>
#include <set>

namespace Elastilm {

    int RANK = 8;
    int SUBMODEL_NUM = 10;
    vector<vector<int>> submodel_attn_hidden_dims; // SUBMODEL_NUM * 2
    vector<vector<int>> submodel_mlp_hidden_dims;
    vector<float> submodel_lora_scales; // SUBMODEL_NUM
    set<int> anchor_layers;
    vector<int> layers_order;


    int LEVEL = 0;
    int CUR_LAYER_ID = 0;
    int IS_ANCHOR_LAYER = 0;

    bool DO_LOAD = true;
};


using namespace mllm;

class ElasticMultiHeadAttention final : public Module {
    ElasticLinear q_proj;
    ElasticLinear k_proj;
    ElasticLinear v_proj;
    RoPE q_rope;
    RoPE k_rope;
    KVCache k_cache;
    KVCache v_cache;
    Softmax softmax;
    ElasticLinear o_proj;
    int head_size_{};
    int kv_head_size_{};
    int attn_hidden_dim_{};

    vector<Layer> lora_q_a;
    vector<Layer> lora_q_b;
    vector<Layer> lora_k_a;
    vector<Layer> lora_k_b;
    vector<Layer> lora_v_a;
    vector<Layer> lora_v_b;
    vector<Layer> lora_o_a;
    vector<Layer> lora_o_b;


public:
    ElasticMultiHeadAttention() = default;
    ElasticMultiHeadAttention(int hidden_dim, int head_size, int kv_head_size, int attn_hidden_dim,
                              RoPEType RoPE_type, int cache_limit, bool do_mask, bool bias, 
                              const TransformerNameConfig &names, const string &base_name) {
        assert(kv_head_size_ == head_size_);
        attn_hidden_dim_ = attn_hidden_dim;
        head_size_ = head_size;
        kv_head_size_ = kv_head_size;
        q_proj = ElasticLinear(hidden_dim, head_size * attn_hidden_dim, bias, base_name + names._q_proj_name);
        k_proj = ElasticLinear(hidden_dim, kv_head_size * attn_hidden_dim, bias, base_name + names._k_proj_name);
        v_proj = ElasticLinear(hidden_dim, kv_head_size * attn_hidden_dim, bias, base_name + names._v_proj_name);

        if (RoPE_type > 0) {
            q_rope = RoPE(RoPE_type, base_name + "q_rope");
            k_rope = RoPE(RoPE_type, base_name + "k_rope");
        }
        if (cache_limit > 0) {
            k_cache = KVCache(head_size / kv_head_size, cache_limit, base_name + "k_cache");
            v_cache = KVCache(head_size / kv_head_size, cache_limit, base_name + "v_cache");
        }
        softmax = Softmax(DIMENSION, do_mask, base_name + "softmax");
        o_proj = ElasticLinear(head_size * attn_hidden_dim, hidden_dim, bias, base_name + names._o_proj_name);

        for(int i = 0; i < Elastilm::SUBMODEL_NUM; i++) {
            auto lora_q_a_ = Linear(hidden_dim, Elastilm::RANK, false, base_name + "lora_q_a_level_" + std::to_string(i));
            auto lora_q_b_ = Linear(Elastilm::RANK, Elastilm::submodel_attn_hidden_dims[i][Elastilm::IS_ANCHOR_LAYER], false, base_name + "lora_q_b_level_" + std::to_string(i));
            auto lora_k_a_ = Linear(hidden_dim, Elastilm::RANK, false, base_name + "lora_k_a_level_" + std::to_string(i));
            auto lora_k_b_ = Linear(Elastilm::RANK, Elastilm::submodel_attn_hidden_dims[i][Elastilm::IS_ANCHOR_LAYER], false, base_name + "lora_k_b_level_" + std::to_string(i));
            auto lora_v_a_ = Linear(hidden_dim, Elastilm::RANK, false, base_name + "lora_v_a_level_" + std::to_string(i));
            auto lora_v_b_ = Linear(Elastilm::RANK, Elastilm::submodel_attn_hidden_dims[i][Elastilm::IS_ANCHOR_LAYER], false, base_name + "lora_v_b_level_" + std::to_string(i));
            auto lora_o_a_ = Linear(Elastilm::submodel_attn_hidden_dims[i][Elastilm::IS_ANCHOR_LAYER], Elastilm::RANK, false, base_name + "lora_o_a_level_" + std::to_string(i));
            auto lora_o_b_ = Linear(Elastilm::RANK, hidden_dim, false, base_name + "lora_o_b_level_" + std::to_string(i));

            lora_q_a.push_back(lora_q_a_);
            lora_q_b.push_back(lora_q_b_);
            lora_k_a.push_back(lora_k_a_);
            lora_k_b.push_back(lora_k_b_);
            lora_v_a.push_back(lora_v_a_);
            lora_v_b.push_back(lora_v_b_);
            lora_o_a.push_back(lora_o_a_);
            lora_o_b.push_back(lora_o_b_);
        }

    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        vector<int> activate_head_dims = std::any_cast<vector<int>>(args[0]);
        int activate_head_dim = activate_head_dims[0];
        activate_head_dim = (activate_head_dim == -1) ? kv_head_size_ : (activate_head_dim);
        Tensor q, k, v;
        q = q_proj(inputs[0], -1, activate_head_dim * attn_hidden_dim_);
        // q.printDataTorchLike<float>();
        // inputs[0].printDataTorchLike<float>();
        // if (Elastilm::DO_LOAD) {
        //     for (int level = 0; level < Elastilm::SUBMODEL_NUM; level++) {
        //         auto tmp_q = lora_q_a[level](inputs[0]);
        //         tmp_q = lora_q_b[level](tmp_q);
        //         tmp_q = tmp_q * Elastilm::submodel_lora_scales[level];
        //         q = q + tmp_q;
        //     }
        // } else {
        //     // auto tmp_q = lora_q_a[Elastilm::LEVEL](inputs[0]);
        //     // tmp_q = lora_q_b[Elastilm::LEVEL](tmp_q);
        //     // tmp_q = tmp_q * Elastilm::submodel_lora_scales[Elastilm::LEVEL];
        //     // q = q + tmp_q;
        // }

        auto tmp_q = lora_q_a[Elastilm::LEVEL](inputs[0]);
        // tmp_q.printDataTorchLike<float>();
        tmp_q = lora_q_b[Elastilm::LEVEL](tmp_q);
        tmp_q = tmp_q * Elastilm::submodel_lora_scales[Elastilm::LEVEL];
        // tmp_q.printDataTorchLike<float>();
        q = q + tmp_q;

        k = k_proj(inputs[1], -1, activate_head_dim * attn_hidden_dim_);
        // if (Elastilm::DO_LOAD) {
        //     for (int level = 0; level < Elastilm::SUBMODEL_NUM; level++) {
        //         auto tmp_k = lora_k_a[level](inputs[0]);
        //         tmp_k = lora_k_b[level](tmp_k);
        //         tmp_k = tmp_k * Elastilm::submodel_lora_scales[level];
        //         k = k + tmp_k;
        //     }
        // } else {
        //     // auto tmp_k = lora_k_a[Elastilm::LEVEL](inputs[1]);
        //     // tmp_k = lora_k_b[Elastilm::LEVEL](tmp_k);
        //     // tmp_k = tmp_k * Elastilm::submodel_lora_scales[Elastilm::LEVEL];
        //     // k = k + tmp_k;
        // }

        auto tmp_k = lora_k_a[Elastilm::LEVEL](inputs[1]);
        tmp_k = lora_k_b[Elastilm::LEVEL](tmp_k);
        tmp_k = tmp_k * Elastilm::submodel_lora_scales[Elastilm::LEVEL];
        k = k + tmp_k;


        v = v_proj(inputs[2], -1, activate_head_dim * attn_hidden_dim_);


        // if (Elastilm::DO_LOAD) {
        //     for (int level = 0; level < Elastilm::SUBMODEL_NUM; level++) {
        //         auto tmp_v = lora_v_a[level](inputs[0]);
        //         tmp_v = lora_v_b[level](tmp_v);
        //         tmp_v = tmp_v * Elastilm::submodel_lora_scales[level];
        //         v = v + tmp_v;
        //     }
        // } else {
        //     // auto tmp_v = lora_v_a[Elastilm::LEVEL](inputs[2]);
        //     // tmp_v = lora_v_b[Elastilm::LEVEL](tmp_v);
        //     // tmp_v = tmp_v * Elastilm::submodel_lora_scales[Elastilm::LEVEL];
        //     // v = v + tmp_v;
        // }

        auto tmp_v = lora_v_a[Elastilm::LEVEL](inputs[2]);
        tmp_v = lora_v_b[Elastilm::LEVEL](tmp_v);
        tmp_v = tmp_v * Elastilm::submodel_lora_scales[Elastilm::LEVEL];

        if(Tensor::tensor_status == TENSOR_STATIC_READY) {
            v.printDataTorchLike<float>();
            tmp_v.printDataTorchLike<float>();
        }

        v = v + tmp_v;

        if(Tensor::tensor_status == TENSOR_STATIC_READY) {
            v.printDataTorchLike<float>();
            exit(0);
        }

        q = q.view(-1, activate_head_dim, -1, attn_hidden_dim_);
        k = k.view(-1, activate_head_dim, -1, attn_hidden_dim_);
        v = v.view(-1, activate_head_dim, -1, attn_hidden_dim_);
        if (q_rope.ready() && k_rope.ready()) {
            q = q_rope(q);
            k = k_rope(k);
        }
        if (k_cache.ready() && v_cache.ready()) {
            k = k_cache(k);
            v = v_cache(v);
        }
        k = k.transpose(SEQUENCE, DIMENSION);
        auto qk = Tensor::mm(q, k);
        qk = qk / std::sqrt(attn_hidden_dim_); // attn_hidden_dim_
        if (k_cache.ready() && v_cache.ready()) {
            qk = softmax(qk, k_cache.getCacheSeqLen());
        } else {
            qk = softmax(qk);
        }
        auto o = Tensor::mm(qk, v);
        o = o.view(-1, 1, -1, attn_hidden_dim_ * activate_head_dim);
        auto o_raw = o;

        o = o_proj(o, activate_head_dim * attn_hidden_dim_, -1);
        // if (Elastilm::DO_LOAD) {
        //     for (int level = 0; level < Elastilm::SUBMODEL_NUM; level++) {
        //         auto tmp_o = lora_o_a[level](inputs[0]);
        //         tmp_o = lora_o_b[level](tmp_o);
        //         tmp_o = tmp_o * Elastilm::submodel_lora_scales[level];
        //         o = o + tmp_o;
        //     }
        // } else {
        //     // auto tmp_o = lora_o_a[Elastilm::LEVEL](o_raw);
        //     // tmp_o = lora_o_b[Elastilm::LEVEL](tmp_o);
        //     // tmp_o = tmp_o * Elastilm::submodel_lora_scales[Elastilm::LEVEL];
        //     // o = o + tmp_o;
        // }

        auto tmp_o = lora_o_a[Elastilm::LEVEL](o_raw);
        tmp_o = lora_o_b[Elastilm::LEVEL](tmp_o);
        tmp_o = tmp_o * Elastilm::submodel_lora_scales[Elastilm::LEVEL];
        o = o + tmp_o;

        return {o};
    }
    vector<KVCache *> get_cache() {
        return {&k_cache, &v_cache};
    }
    vector<RoPE *> get_rope() {
        return {&q_rope, &k_rope};
    }
};

class ElasticLLaMAMLP final : public Module {
    ElasticLinear gate_proj;
    Layer silu;
    ElasticLinear up_proj;
    ElasticLinear down_proj;

    vector<Layer> lora_gate_a;
    vector<Layer> lora_gate_b;
    vector<Layer> lora_up_a;
    vector<Layer> lora_up_b;
    vector<Layer> lora_down_a;
    vector<Layer> lora_down_b;


public:
    ElasticLLaMAMLP() = default;
    ElasticLLaMAMLP(int hidden_dim, int ffn_hidden, const LLaMANameConfig &names, const string &base_name) {
        gate_proj = ElasticLinear(hidden_dim, ffn_hidden, false, base_name + names._gate_proj_name);
        silu = SiLU(base_name + "act");
        up_proj = ElasticLinear(hidden_dim, ffn_hidden, false, base_name + names._up_proj_name);
        down_proj = ElasticLinear(ffn_hidden, hidden_dim, false, base_name + names._down_proj_name);

        for(int i = 0; i < Elastilm::SUBMODEL_NUM; i++) {
            auto lora_gate_a_ = Linear(hidden_dim, Elastilm::RANK, false, base_name + "lora_gate_a_level_" + std::to_string(i));
            auto lora_gate_b_ = Linear(Elastilm::RANK, Elastilm::submodel_mlp_hidden_dims[i][Elastilm::IS_ANCHOR_LAYER], false, base_name + "lora_gate_b_level_" + std::to_string(i));
            auto lora_up_a_ = Linear(hidden_dim, Elastilm::RANK, false, base_name + "lora_up_a_level_" + std::to_string(i));
            auto lora_up_b_ = Linear(Elastilm::RANK, Elastilm::submodel_mlp_hidden_dims[i][Elastilm::IS_ANCHOR_LAYER], false, base_name + "lora_up_b_level_" + std::to_string(i));
            auto lora_down_a_ = Linear(hidden_dim, Elastilm::RANK, false, base_name + "lora_down_a_level_" + std::to_string(i));
            auto lora_down_b_ = Linear(Elastilm::RANK, Elastilm::submodel_mlp_hidden_dims[i][Elastilm::IS_ANCHOR_LAYER], false, base_name + "lora_down_b_level_" + std::to_string(i));

            lora_gate_a.push_back(lora_gate_a_);
            lora_gate_b.push_back(lora_gate_b_);
            lora_up_a.push_back(lora_up_a_);
            lora_up_b.push_back(lora_up_b_);
            lora_down_a.push_back(lora_down_a_);
            lora_down_b.push_back(lora_down_b_);
        }


    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        vector<int> activate_dims = std::any_cast<vector<int>>(args[0]);
        int activate_dim = activate_dims[0];
        auto x = gate_proj(inputs[0], -1, activate_dim);

        // if (Elastilm::DO_LOAD) {
        //     for (int level = 0; level < Elastilm::SUBMODEL_NUM; level++) {
        //         auto tmp_gate = lora_gate_a[level](inputs[0]);
        //         tmp_gate = lora_gate_b[level](tmp_gate);
        //         tmp_gate = tmp_gate * Elastilm::submodel_lora_scales[level];
        //         x = x + tmp_gate;
        //     }
        // } else {
        //     // auto tmp_gate = lora_gate_a[Elastilm::LEVEL](inputs[0]);
        //     // tmp_gate = lora_gate_b[Elastilm::LEVEL](tmp_gate);
        //     // tmp_gate = tmp_gate * Elastilm::submodel_lora_scales[Elastilm::LEVEL];
        //     // x = x + tmp_gate;
        // }

        auto tmp_gate = lora_gate_a[Elastilm::LEVEL](inputs[0]);
        tmp_gate = lora_gate_b[Elastilm::LEVEL](tmp_gate);
        tmp_gate = tmp_gate * Elastilm::submodel_lora_scales[Elastilm::LEVEL];
        x = x + tmp_gate;

        x = silu(x);

        auto y = up_proj(inputs[0], -1, activate_dim);
        // if (Elastilm::DO_LOAD) {
        //     for (int level = 0; level < Elastilm::SUBMODEL_NUM; level++) {
        //         auto tmp_up = lora_up_a[level](inputs[0]);
        //         tmp_up = lora_up_b[level](tmp_up);
        //         tmp_up = tmp_up * Elastilm::submodel_lora_scales[level];
        //         x = x + tmp_up;
        //     }
        // } else {
        //     // auto tmp_up = lora_up_a[Elastilm::LEVEL](inputs[0]);
        //     // tmp_up = lora_up_b[Elastilm::LEVEL](tmp_up);
        //     // tmp_up = tmp_up * Elastilm::submodel_lora_scales[Elastilm::LEVEL];
        //     // x = x + tmp_up;
        // }

        auto tmp_up = lora_up_a[Elastilm::LEVEL](inputs[0]);
        tmp_up = lora_up_b[Elastilm::LEVEL](tmp_up);
        tmp_up = tmp_up * Elastilm::submodel_lora_scales[Elastilm::LEVEL];
        x = x + tmp_up;

        x = x * y;
        x = down_proj(x, activate_dim, -1);
        // if (Elastilm::DO_LOAD) {
        //     for (int level = 0; level < Elastilm::SUBMODEL_NUM; level++) {
        //         auto tmp_down = lora_down_a[level](inputs[0]);
        //         tmp_down = lora_down_b[level](tmp_down);
        //         tmp_down = tmp_down * Elastilm::submodel_lora_scales[level];
        //         x = x + tmp_down;
        //     }
        // } else {
        //     // auto tmp_down = lora_down_a[Elastilm::LEVEL](inputs[0]);
        //     // tmp_down = lora_down_b[Elastilm::LEVEL](tmp_down);
        //     // tmp_down = tmp_down * Elastilm::submodel_lora_scales[Elastilm::LEVEL];
        //     // x = x + tmp_down;
        // }

        auto tmp_down = lora_down_a[Elastilm::LEVEL](inputs[0]);
        tmp_down = lora_down_b[Elastilm::LEVEL](tmp_down);
        tmp_down = tmp_down * Elastilm::submodel_lora_scales[Elastilm::LEVEL];
        x = x + tmp_down;

        return {x};
    }
};

class ElasticLLaMABlock final : public Module {
    ElasticMultiHeadAttention attention;
    ElasticLLaMAMLP mlp;
    Layer norm1;
    Layer norm2;

public:
    ElasticLLaMABlock() = default;
    ElasticLLaMABlock(int hidden_dim, int head_size, int ffn_hidden, RoPEType RoPE_type, int cache_limit, const LLaMANameConfig &names, const string &base_name) {

        size_t lastPos = base_name.rfind('.');
        size_t secondLastPos = base_name.rfind('.', lastPos - 1);
        std::string result = base_name.substr(0, secondLastPos+1);

        result = result + std::to_string(Elastilm::layers_order[Elastilm::CUR_LAYER_ID]) + ".";
        attention = ElasticMultiHeadAttention(hidden_dim, head_size, head_size, hidden_dim / head_size,
                                              RoPE_type, cache_limit, true, false, names, result + names._attn_base_name);
        mlp = ElasticLLaMAMLP(hidden_dim, ffn_hidden, names, result + names._ffn_base_name);
        norm1 = RMSNorm(hidden_dim, 1e-6, result + names._attn_norm_name);
        norm2 = RMSNorm(hidden_dim, 1e-6, result + names._ffn_norm_name);
        Elastilm::CUR_LAYER_ID += 1;
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        vector<int> activate_dims = std::any_cast<vector<int>>(args[0]);
        vector<int> dim_attns = {activate_dims[0]};
        vector<int> dim_mlps = {activate_dims[1]};
        auto x = norm1(inputs[0]);
        x = attention({x, x, x}, dim_attns)[0];
        auto tmp = x + inputs[0];
        x = norm2(tmp);
        x = mlp({x}, dim_mlps)[0];
        x = x + tmp;
        return {x};
    }
    ElasticMultiHeadAttention &get_attention() {
        return attention;
    }
};

class ElasticLLaMAModelWithLoRA final : public Module {
    Layer embedding;
    vector<ElasticLLaMABlock> blocks;
    vector<ElasticLLaMABlock> blocks_anchor;
    Layer norm;
    Layer lm_head;
    int num_layer_size;

public:
    explicit ElasticLLaMAModelWithLoRA(const LLaMAConfig &config) :
        ElasticLLaMAModelWithLoRA(config.vocab_size, config.hidden_dim, config.head_size, config.ffn_hidden, config.block_num, config.RoPE_type, config.cache_limit,
                          config.names_config, config.names_config.blk_name) {
    }
    ElasticLLaMAModelWithLoRA(int vocab_size, int hidden_dim, int head_size, int ffn_hidden, int block_num, RoPEType RoPE_type, int cache_limit,
                      const LLaMANameConfig &names, const string &base_name) {

        Elastilm::CUR_LAYER_ID = 0;

        embedding = Embedding(vocab_size, hidden_dim, names.token_embd_name);
        Elastilm::IS_ANCHOR_LAYER = 0;
        blocks = List<ElasticLLaMABlock>(block_num - Elastilm::anchor_layers.size(), hidden_dim, head_size, ffn_hidden, RoPE_type, cache_limit, names, base_name);
        Elastilm::IS_ANCHOR_LAYER = 1;
        blocks_anchor = List<ElasticLLaMABlock>(Elastilm::anchor_layers.size(), hidden_dim, head_size, ffn_hidden, RoPE_type, cache_limit, names, base_name);
        norm = RMSNorm(hidden_dim, 1e-6, names.post_norm_name);
        lm_head = Linear(hidden_dim, vocab_size, false, names.lm_head_name);
        num_layer_size = block_num;
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        vector<vector<int>> activate_dims = std::any_cast<vector<vector<int>>>(args[0]);
        assert(activate_dims.size() == num_layer_size);
        auto x = embedding(inputs[0]);

        x.printDataTorchLike<float>();

        int cur_block = 0;
        int cur_anchor_block = 0;
        for (int id = 0; id < num_layer_size; id++) {
            if(Elastilm::anchor_layers.find(id) != Elastilm::anchor_layers.end()) {
                x = blocks_anchor[cur_anchor_block]({x}, activate_dims[id])[0];
                cur_anchor_block += 1;
            } else {
                x = blocks[cur_block]({x}, activate_dims[id])[0];
                cur_block += 1;
            }
        }
        x = norm(x);
        x = lm_head(x);
        return {x};
    }

    void clear_kvcache() override {
        for (auto &block : blocks) {
            auto kvcache = block.get_attention().get_cache();
            for (auto &cache : kvcache) { cache->clearCache(); }
            auto ropes = block.get_attention().get_rope();
            for (auto &rope : ropes) { rope->clearCache(); }
        }
    }
};

#endif // MODELING_LLAMA_HPP