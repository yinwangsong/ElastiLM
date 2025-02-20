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

using namespace mllm;

class ElasticMultiHeadAttention final : public Module {
    ElasticLinearWithLoRA q_proj;
    ElasticLinearWithLoRA k_proj;
    ElasticLinearWithLoRA v_proj;
    RoPE q_rope;
    RoPE k_rope;
    KVCache k_cache;
    KVCache v_cache;
    Softmax softmax;
    ElasticLinearWithLoRA o_proj;
    int head_size_{};
    int kv_head_size_{};
    int attn_hidden_dim_{};

public:
    ElasticMultiHeadAttention() = default;
    ElasticMultiHeadAttention(int hidden_dim, int head_size, int kv_head_size, int attn_hidden_dim,
                              RoPEType RoPE_type, int cache_limit, bool do_mask, bool bias,
                              const TransformerNameConfig &names, const string &base_name) {
        assert(kv_head_size_ == head_size_);
        attn_hidden_dim_ = attn_hidden_dim;
        head_size_ = head_size;
        kv_head_size_ = kv_head_size;
        q_proj = ElasticLinearWithLoRA(hidden_dim, head_size * attn_hidden_dim, bias, base_name + names._q_proj_name);
        k_proj = ElasticLinearWithLoRA(hidden_dim, kv_head_size * attn_hidden_dim, bias, base_name + names._k_proj_name);
        v_proj = ElasticLinearWithLoRA(hidden_dim, kv_head_size * attn_hidden_dim, bias, base_name + names._v_proj_name);

        if (RoPE_type > 0) {
            q_rope = RoPE(RoPE_type, base_name + "q_rope");
            k_rope = RoPE(RoPE_type, base_name + "k_rope");
        }
        if (cache_limit > 0) {
            k_cache = KVCache(head_size / kv_head_size, cache_limit, base_name + "k_cache");
            v_cache = KVCache(head_size / kv_head_size, cache_limit, base_name + "v_cache");
            // std::cout<<Elastilm::CUR_LAYER_ID<<"\n";
            Elastilm::CUR_LAYER_ID+=1;
        }
        softmax = Softmax(DIMENSION, do_mask, base_name + "softmax");
        o_proj = ElasticLinearWithLoRA(head_size * attn_hidden_dim, hidden_dim, bias, base_name + names._o_proj_name);

        // lora_q_a = List<Linear>
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        vector<int> activate_head_dims = std::any_cast<vector<int>>(args[0]);
        int activate_head_dim = activate_head_dims[0];
        activate_head_dim = (activate_head_dim == -1) ? kv_head_size_ : (activate_head_dim);
        Tensor q, k, v;
        q = q_proj(inputs[0], -1, activate_head_dim * attn_hidden_dim_);
        k = k_proj(inputs[1], -1, activate_head_dim * attn_hidden_dim_);
        v = v_proj(inputs[2], -1, activate_head_dim * attn_hidden_dim_);

        // if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        //     q.printDataTorchLike<float>();
        //     k.printDataTorchLike<float>();
        //     v.printDataTorchLike<float>();
        // }
        // if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        //     exit(-1);
        // }
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
        // if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        //     qk.printDataTorchLike<float>();
        // }
        qk = qk / std::sqrt(attn_hidden_dim_); // attn_hidden_dim_
        // if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        //     qk.printDataTorchLike<float>();
        // }
        if (k_cache.ready() && v_cache.ready()) {
            qk = softmax(qk, k_cache.getCacheSeqLen());
        } else {
            qk = softmax(qk);
        }
        // if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        //     qk.printDataTorchLike<float>();
        // }
        auto o = Tensor::mm(qk, v);
        o = o.view(-1, 1, -1, attn_hidden_dim_ * activate_head_dim);
        o = o_proj(o, activate_head_dim * attn_hidden_dim_, -1);
        // if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        //     o.printDataTorchLike<float>();
        // }

        // if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        //     inputs[0].printDataTorchLike<float>();
        // }
        // printActivations<float>();
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
    ElasticLinearWithLoRA gate_proj;
    Layer silu;
    ElasticLinearWithLoRA up_proj;
    ElasticLinearWithLoRA down_proj;

public:
    ElasticLLaMAMLP() = default;
    ElasticLLaMAMLP(int hidden_dim, int ffn_hidden, const LLaMANameConfig &names, const string &base_name) {
        gate_proj = ElasticLinearWithLoRA(hidden_dim, ffn_hidden, false, base_name + names._gate_proj_name);
        silu = SiLU(base_name + "act");
        up_proj = ElasticLinearWithLoRA(hidden_dim, ffn_hidden, false, base_name + names._up_proj_name);
        down_proj = ElasticLinearWithLoRA(ffn_hidden, hidden_dim, false, base_name + names._down_proj_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        vector<int> activate_dims = std::any_cast<vector<int>>(args[0]);
        int activate_dim = activate_dims[0];
        // if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        //     inputs[0].printDataTorchLike<float>();
        // }
        auto x = gate_proj(inputs[0], -1, activate_dim);
        // if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        //     std::cout<<123123<<std::endl;
        //     x.printDataTorchLike<float>();
        // }
        x = silu(x);
        // if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        //     x.printDataTorchLike<float>();
        // }
        auto y = up_proj(inputs[0], -1, activate_dim);
        // if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        //     y.printDataTorchLike<float>();
        // }
        x = x * y;
        x = down_proj(x, activate_dim, -1);
        // if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        //     y.printDataTorchLike<float>();
        // }
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
        attention = ElasticMultiHeadAttention(hidden_dim, head_size, head_size, hidden_dim / head_size,
                                              RoPE_type, cache_limit, true, false, names, base_name + names._attn_base_name);
        mlp = ElasticLLaMAMLP(hidden_dim, ffn_hidden, names, base_name + names._ffn_base_name);
        norm1 = RMSNorm(hidden_dim, 1e-6, base_name + names._attn_norm_name);
        norm2 = RMSNorm(hidden_dim, 1e-6, base_name + names._ffn_norm_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        vector<int> activate_dims = std::any_cast<vector<int>>(args[0]);
        vector<int> dim_attns = {activate_dims[0]};
        vector<int> dim_mlps = {activate_dims[1]};
        // inputs[0].printDataTorchLike<float>();
        // Tensor input_raw = inputs[0];
        auto x = norm1(inputs[0]);
        // inputs[0].printDataTorchLike<float>();
        x = attention({x, x, x}, dim_attns)[0];
        // attention.printActivations<float>();
        // x.printDataTorchLike<float>();
        // std::cout<<123<<std::endl;
        // if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        //     inputs[0].printData0<float>();
        // }
        // inputs[0].printDataTorchLike<float>();
        auto tmp = x + inputs[0];
        // tmp.printDataTorchLike<float>();
        x = norm2(tmp);
        // x.printDataTorchLike<float>();
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
        embedding = Embedding(vocab_size, hidden_dim, names.token_embd_name);
        blocks = List<ElasticLLaMABlock>(block_num - Elastilm::anchor_layers.size(), hidden_dim, head_size, ffn_hidden, RoPE_type, cache_limit, names, base_name);
        blocks_anchor = List<ElasticLLaMABlock>(Elastilm::anchor_layers.size(), hidden_dim, head_size, ffn_hidden, RoPE_type, cache_limit, names, base_name + "anchor.");
        norm = RMSNorm(hidden_dim, 1e-6, names.post_norm_name);
        lm_head = Linear(hidden_dim, vocab_size, false, names.lm_head_name);
        num_layer_size = block_num;
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        vector<vector<int>> activate_dims = std::any_cast<vector<vector<int>>>(args[0]);
        assert(activate_dims.size() == num_layer_size);
        auto x = embedding(inputs[0]);

        // x.printDataTorchLike<float>();
        int cur_block = 0;
        int cur_anchor_block = 0;
        for (int id = 0; id < blocks.size() + blocks_anchor.size(); id++) {
            // std::cout<<id<<std::endl;
            if(Elastilm::anchor_layers.find(id) != Elastilm::anchor_layers.end()) {
                Elastilm::IS_ANCHOR_LAYER = 1;
                x = blocks_anchor[cur_anchor_block]({x}, activate_dims[id])[0];
                // if (Tensor::tensor_status == TENSOR_STATIC_READY) {
                //     x.printDataTorchLike<float>();
                //     exit(-1);
                // }
                cur_anchor_block += 1;
            } else {
                Elastilm::IS_ANCHOR_LAYER = 0;
                // x.printDataTorchLike<float>();
                x = blocks[cur_block]({x}, activate_dims[id])[0];
                // x.printDataTorchLike<float>();
                cur_block += 1;
            }
            // x.printDataTorchLike<float>();
            // if (Tensor::tensor_status == TENSOR_STATIC_READY) {
            //     exit(-1);
            // }
        }
        // std::cout<<123<<std::endl;
        x = norm(x);
        x = lm_head(x);
        return {x};
    }

    void revert_kvcache_to_pos(int pos) {

        for (auto &block : blocks) {
            auto kvcache = block.get_attention().get_cache();
            for (auto &cache : kvcache) { cache->revertCache(pos); }
            auto ropes = block.get_attention().get_rope();
            for (auto &rope : ropes) { rope->revertCache(pos); }
        }

        for (auto &block : blocks_anchor) {
            auto kvcache = block.get_attention().get_cache();
            for (auto &cache : kvcache) { cache->revertCache(pos); }
            auto ropes = block.get_attention().get_rope();
            for (auto &rope : ropes) { rope->revertCache(pos); }
        }

    }

    void clear_kvcache() override {
        for (auto &block : blocks) {
            auto kvcache = block.get_attention().get_cache();
            for (auto &cache : kvcache) { cache->clearCache(); }
            auto ropes = block.get_attention().get_rope();
            for (auto &rope : ropes) { rope->clearCache(); }
        }

        for (auto &block : blocks_anchor) {
            auto kvcache = block.get_attention().get_cache();
            for (auto &cache : kvcache) { cache->clearCache(); }
            auto ropes = block.get_attention().get_rope();
            for (auto &rope : ropes) { rope->clearCache(); }
        }
    }
};

#endif // MODELING_LLAMA_HPP