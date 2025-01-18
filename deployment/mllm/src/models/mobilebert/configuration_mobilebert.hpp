#ifndef CONFIG_MOBILEBERT_HPP
#define CONFIG_MOBILEBERT_HPP
#include "Types.hpp"
#include "models/transformer/configuration_transformer.hpp"
#include <cctype>
#include <iterator>

using namespace mllm;

class MobileBertNameConfig : public TransformerNameConfig {
public:
    void init() {
        // modules
        embedding_module = "embeddings.";
        encoder_module = "encoder.";
        layer_module = "layer.";
        attn_module = "attention.";
        self_module = "self.";
        intermediate_module = "intermediate.";
        output_module = "output.";
        input_module = "input.";
        bottleneck_module = "bottleneck.";
        ffn_module = "ffn.";
        head_module = "head.";

        //layers
        _q_proj_name = "query";
        _k_proj_name = "key";
        _v_proj_name = "value";
        _dense_name = "dense";
        _layer_norm_name = "LayerNorm";
        _softmax_name = "softmax";
        _act_name = "act";
        _head_name = "head";
    }
    std::string embedding_module;
    std::string encoder_module;
    std::string layer_module;
    std::string attn_module;
    std::string self_module;
    std::string intermediate_module;
    std::string output_module;
    std::string bottleneck_module;
    std::string ffn_module;
    std::string input_module;
    std::string head_module;

    std::string _q_proj_name;
    std::string _k_proj_name;
    std::string _v_proj_name;
    std::string _dense_name;
    std::string _layer_norm_name;
    std::string _softmax_name;
    std::string _act_name;
    std::string _head_name;
};

struct MobileBertConfig : public TransformerConfig {
    explicit MobileBertConfig() {
        names_config.init();
    };

    int intra_bottleneck_size = 128;
    int embedding_size = 128;
    int true_hidden_size = 128;
    bool use_bottleneck = true;
    int num_feedforward_networks = 3;
    int type_vocab_size = 2;
    float layer_norm_eps = 1e-12;
    std::string hidden_act = "ReLU";
    int hidden_size = 512;
    int intermediate_size = 512;
    int max_position_embeddings = 512;
    int num_attention_heads = 4;
    int num_hidden_layers = 24;
    int num_classes = 2;
    int vocab_size = 30522;

    MobileBertNameConfig names_config;
};

#endif //! CONFIG_BERT_HPP