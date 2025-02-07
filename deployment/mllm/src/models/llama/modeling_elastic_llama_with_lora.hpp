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

    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {

        // std::cout<<inputs.size()<<std::endl;

        vector<int> activate_head_dims = std::any_cast<vector<int>>(args[0]);
        int activate_head_dim = activate_head_dims[0];
        activate_head_dim = (activate_head_dim == -1) ? kv_head_size_ : (activate_head_dim);

        vector<float> lora_scales = std::any_cast<vector<float>>(args[1]);

        Tensor q_delta = inputs[0];
        Tensor k_delta = inputs[1];
        Tensor v_delta = inputs[2];
        Tensor o_delta;

        Tensor q, k, v;
        Tensor lora_q_a = inputs[3];
        Tensor lora_q_b = inputs[4];
        Tensor lora_k_a = inputs[5];
        Tensor lora_k_b = inputs[6];
        Tensor lora_v_a = inputs[7];
        Tensor lora_v_b = inputs[8];
        Tensor lora_o_a = inputs[9];
        Tensor lora_o_b = inputs[10];


        float lora_q_scale = lora_scales[0];
        float lora_k_scale = lora_scales[1];
        float lora_v_scale = lora_scales[2];
        float lora_o_scale = lora_scales[3];


        q = q_proj(inputs[0], -1, activate_head_dim * attn_hidden_dim_);
        q_delta.printDataTorchLike<float>();
        lora_q_a.printDataTorchLike<float>();
        q_delta = Tensor::mm(q_delta, lora_q_a);
        q_delta.printDataTorchLike<float>();
        q_delta = Tensor::mm(q_delta, lora_q_b);
        q_delta.printDataTorchLike<float>();
        q_delta = q_delta*lora_q_scale;
        q = q + q_delta;

        k = k_proj(inputs[1], -1, activate_head_dim * attn_hidden_dim_);
        k_delta = Tensor::mm(k_delta, lora_k_a);
        k_delta = Tensor::mm(k_delta, lora_k_b);
        k_delta = k_delta*lora_k_scale;
        k = k + k_delta;

        v = v_proj(inputs[2], -1, activate_head_dim * attn_hidden_dim_);
        v_delta = Tensor::mm(v_delta, lora_v_a);
        v_delta = Tensor::mm(v_delta, lora_v_b);
        v_delta = v_delta*lora_v_scale;
        v = v + v_delta;

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

        o_delta = o;
        o = o_proj(o, activate_head_dim * attn_hidden_dim_, -1);
        o_delta = Tensor::mm(o_delta, lora_o_a);
        o_delta = Tensor::mm(o_delta, lora_o_b);
        o_delta = o_delta*lora_o_scale;
        o = o + o_delta;

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

public:
    ElasticLLaMAMLP() = default;
    ElasticLLaMAMLP(int hidden_dim, int ffn_hidden, const LLaMANameConfig &names, const string &base_name) {
        gate_proj = ElasticLinear(hidden_dim, ffn_hidden, false, base_name + names._gate_proj_name);
        silu = SiLU(base_name + "act");
        up_proj = ElasticLinear(hidden_dim, ffn_hidden, false, base_name + names._up_proj_name);
        down_proj = ElasticLinear(ffn_hidden, hidden_dim, false, base_name + names._down_proj_name);

        5组

        

    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        vector<int> activate_dims = std::any_cast<vector<int>>(args[0]);
        int activate_dim = activate_dims[0];

        vector<float> lora_scales = std::any_cast<vector<float>>(args[1]);

        Tensor gate_delta = inputs[0];
        Tensor up_delta = inputs[0];
        Tensor down_delta;

        Tensor lora_gate_a = inputs[1];
        Tensor lora_gate_b = inputs[2];
        Tensor lora_up_a = inputs[3];
        Tensor lora_up_b = inputs[4];
        Tensor lora_down_a = inputs[5];
        Tensor lora_down_b = inputs[6];
        float lora_gate_scale = lora_scales[0];
        float lora_up_scale = lora_scales[1];
        float lora_down_scale = lora_scales[2];

        auto x = gate_proj(inputs[0], -1, activate_dim);
        gate_delta = Tensor::mm(gate_delta, lora_gate_a);
        gate_delta = Tensor::mm(gate_delta, lora_gate_b);
        gate_delta = gate_delta * lora_gate_scale;
        x = x + gate_delta;
        
        x = silu(x);

        auto y = up_proj(inputs[0], -1, activate_dim);
        up_delta = Tensor::mm(up_delta, lora_up_a);
        up_delta = Tensor::mm(up_delta, lora_up_b);
        up_delta = up_delta * lora_up_scale;
        y = y + up_delta;

        x = x * y;
        down_delta = x;
        x = down_proj(x, activate_dim, -1);
        down_delta = Tensor::mm(down_delta, lora_down_a);
        down_delta = Tensor::mm(down_delta, lora_down_b);
        down_delta = down_delta * lora_down_scale;
        x = x + down_delta;
        return {x};
    }
};

class ElasticLLaMABlock final : public Module {
    ElasticMultiHeadAttention attention;
    ElasticLLaMAMLP mlp;
    Layer norm1;
    Layer norm2;

public:
    int allocated = 1;
    ElasticLLaMABlock() = default;
    ElasticLLaMABlock(int hidden_dim, int head_size, int ffn_hidden, RoPEType RoPE_type, int cache_limit, const LLaMANameConfig &names, const string &base_name) {
        attention = ElasticMultiHeadAttention(hidden_dim, head_size, head_size, hidden_dim / head_size,
                                              RoPE_type, cache_limit, true, false, names, base_name + names._attn_base_name);
        mlp = ElasticLLaMAMLP(hidden_dim, ffn_hidden, names, base_name + names._ffn_base_name);
        norm1 = RMSNorm(hidden_dim, 1e-6, base_name + names._attn_norm_name);
        norm2 = RMSNorm(hidden_dim, 1e-6, base_name + names._ffn_norm_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        // std::cout<<"place-1\n";
        vector<int> activate_dims = std::any_cast<vector<int>>(args[0]);
        vector<int> dim_attns = {activate_dims[0]};
        vector<int> dim_mlps = {activate_dims[1]};


        // std::cout<<"place0\n";
        vector<Tensor> lora_attn = {};
        for (int i = 0; i < 4*2; i++) {
            lora_attn.push_back(inputs[i+1]);
        }
        // std::cout<<"place1\n";
        vector<Tensor> lora_mlp = {};
        for (int i = 4*2; i < 7*2; i++) {
            lora_mlp.push_back(inputs[i+1]);
        }

        // std::cout<<"place2\n";
        // 7*1*layernum scales
        vector<float> scales = std::any_cast<vector<float>>(args[1]);
        vector<float> scales_attn = {scales[0], scales[1], scales[2], scales[3]};
        vector<float> scales_mlp = {scales[4], scales[5], scales[6]};

        std::cout<<123<<std::endl;
        inputs[0].printDataTorchLike<float>();
        auto x = norm1(inputs[0]);
        x.printDataTorchLike<float>();
        vector<Tensor> attn_input = {x, x, x};
        attn_input.insert(attn_input.end(), lora_attn.begin(), lora_attn.end());
        // std::cout<<"begin attn\n";
        // for(auto i : dim_attns){
        //     std::cout<<i<<std::endl;
        // }
        // for(auto j : scales_attn){
        //     std::cout<<j<<std::endl;
        // }
        x = attention(attn_input, dim_attns, scales_attn)[0];
        auto tmp = x + inputs[0];
        x = norm2(tmp);
        vector<Tensor> mlp_input = {x};
        mlp_input.insert(mlp_input.end(), lora_mlp.begin(), lora_mlp.end());
        x = mlp(mlp_input, dim_mlps, scales_mlp)[0];
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
        blocks = List<ElasticLLaMABlock>(block_num, hidden_dim, head_size, ffn_hidden, RoPE_type, cache_limit, names, base_name);
        norm = RMSNorm(hidden_dim, 1e-6, names.post_norm_name);
        lm_head = Linear(hidden_dim, vocab_size, false, names.lm_head_name);
        num_layer_size = block_num;
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        // std::cout<<"forwarding "<<std::any_cast<vector<vector<int>>>(args[0]).size()<<std::endl;
        vector<vector<int>> activate_dims = std::any_cast<vector<vector<int>>>(args[0]);
        // for (auto i : activate_dims){
        //     std::cout<<"aaa"<<std::endl;
        //     for (auto j : i) {
        //         std::cout<<j<<std::endl;
        //     }
        // }
        // std::cout<<activate_dims.size()<<std::endl;
        assert(activate_dims.size() == num_layer_size);
        
        // Wq/Wk/Wv/Wo/Wgate/Wup/Wdown
        // 7*2*layernum
        vector<vector<Tensor>> LoRAs = {};
        for (int i = 0; i < blocks.size(); i++) {
            vector<Tensor> lora = {};
            for (int j = 0; j < 7*2; j++) {
                lora.push_back(inputs[1 + i*14 + j]);
            }
            LoRAs.push_back(lora);
        }

        // 7*1*layernum scales
        vector<vector<float>> scales = std::any_cast<vector<vector<float>>>(args[1]);

        // std::cout<<scales.size()<<std::endl;
        // inputs[0].printDataTorchLike<float>();
        auto x = embedding(inputs[0]);
        x.printDataTorchLike<float>();
        for (int id = 0; id < blocks.size(); id++) {
            vector<Tensor> block_in = {x};
            block_in.insert(block_in.end(), LoRAs[id].begin(), LoRAs[id].end());
            // std::cout<<block_in.size()<<std::endl;
            // std::cout<<activate_dims[id].size()<<std::endl;
            // std::cout<<scales[id].size()<<std::endl;
            // std::cout<<blocks.size()<<std::endl;

            // inputs[0].printDataTorchLike<float>();
            // for (Tensor& i : block_in) {
            //     i.printDataTorchLike<float>();
            // }
            // for (auto i : activate_dims[id]) {
            //     std::cout<<i<<std::endl;
            // }
            // std::cout<<std::endl;
            // for (auto j : scales[id]) {
            //     std::cout<<j<<std::endl;
            // }
            // std::cout<<blocks[id].allocated<<std::endl;
            auto tmp1 = activate_dims[id];
            // std::cout<<"tmp1"<<std::endl;
            auto tmp2 = scales[id];
            // std::cout<<"tmp2"<<std::endl;
            auto block_ = blocks[id];
            // std::cout<<"block"<<std::endl;
            block_in[0].printDataTorchLike<float>();
            auto tmpx = block_(block_in, tmp1, tmp2);
            x = tmpx[0];
        }
        x = norm(x);
        x = lm_head(x);
        return {x};
    }

    void clear_kvcache() override {
        std::cout<<"780\n";
        for (auto &block : blocks) {
            std::cout<<"456\n";
            auto kvcache = block.get_attention().get_cache();
            std::cout<<"123\n";
            for (auto &cache : kvcache) { cache->clearCache(); }
            auto ropes = block.get_attention().get_rope();
            for (auto &rope : ropes) { rope->clearCache(); }
        }
    }

    void load(string path, const LLaMAConfig &config) {
        // create global loader and save to llm_model_ptr.loader as QNNBackend needs to load weights in runtime
        // std::cout<<"start init\n";
        loader = new ParamLoader(std::move(path));
        // std::cout<<"start2 init\n";
        load(*loader, config);
    }

    void load(AbstructLoader &param_loader, const LLaMAConfig &config) {
        Tensor::tensor_status = TENSOR_STATIC_INIT;
        mllm_time_init();

        loader = &param_loader;
        doLoad = true;
        vector<Tensor> tmps;
        int max_in_size = 1 + 7*2*config.block_num + 10;
        // std::cout<<"start init\n";
        for (int i = 0; i < max_in_size; ++i) {
            Tensor t(Backend::global_backends[MLLM_CPU]);
            // std::cout<<"setting name\n";
            t.setName("input" + std::to_string(i));
            t.reshape(1, 1, 1, 10);
            // std::cout<<i<<" start alloc\n";
            t.alloc();
            // std::cout<<"initing\n";
            t.setModule(this);
            tmps.push_back(t);
            // t.printDataTorchLike<float>();
            // std::cout<<"inited\n";
        }
        // std::cout<<"finish init\n";
        llm_model_ptr = this;
        // vector<std::any> alternate_args = {
        //     // {},
        //     // vector<int>{0, 0},
        //     // {},
        //     // {},
        //     std::vector<int>{0, 0}
        //     // std::vector<std::vector<int>>(1, std::vector<int>(2)),
        //     // std::vector<std::vector<float>>(32, std::vector<float>(7))
        // };
        uint64_t time_start = 0;
        // for (auto args : alternate_args) {
        time_start = mllm_time_us();
        auto args0 = std::vector<std::vector<int>>(config.block_num, std::vector<int>(2));
        auto args1 = std::vector<std::vector<float>>(config.block_num, std::vector<float>(7));
        try {
            operator()(tmps, args0, args1);
        } catch (const std::exception &e) {
#if not defined(__ARM_NEON)
            if (std::string("bad any_cast") != e.what()) {
                MLLM_LOG_ERROR_STREAM << e.what() << std::endl;
                exit(0);
            }
#endif
        } catch (...) {
            MLLM_LOG_ERROR_STREAM << "load error" << std::endl;
            exit(0);
        }
        // }
        uint64_t time_end = mllm_time_us();
        load_time_ = (time_end - time_start) / 1000.0F; // ms
        doLoad = false;
    }
};

#endif // MODELING_LLAMA_HPP