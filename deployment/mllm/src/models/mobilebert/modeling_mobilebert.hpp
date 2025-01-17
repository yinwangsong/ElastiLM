#ifndef MODELING_MOBILEBERT_HPP
#define MODELING_MOBILEBERT_HPP

#include "Backend.hpp"
#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "configuration_mobilebert.hpp"
#include "models/transformer/modeling_transformer.hpp"
using namespace mllm;

class MobileBertEmbeddings : public Module {
public:
    MobileBertEmbeddings() = default;
    MobileBertEmbeddings(MobileBertConfig &config, const string &base_name) {
        embedding_size = config.hidden_size;
        hidden_size = config.hidden_size;

        word_embeddings = Embedding(config.vocab_size, config.embedding_size, base_name + "word_embeddings");
        token_type_embeddings = Embedding(config.type_vocab_size, config.hidden_size, base_name + "token_type_embeddings");
        position_embeddings = Embedding(config.max_position_embeddings, hidden_size, base_name + "position_embeddings");
        layer_norm = NoNorm(config.hidden_size, true,  base_name + config.names_config._layer_norm_name);

        // mobilebert_conv_embed = MBBertConvEmbed(base_name + "mobilebert_conv_embed"); // currently hard coded inside the op
        
        int embedded_input_size = config.embedding_size * 3;
        embedding_transformation = Linear(embedded_input_size, config.hidden_size, true, base_name + "embedding_transformation");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        Tensor inputs_embeds = word_embeddings(inputs[0]);
        // inputs_embeds = mobilebert_conv_embed(inputs_embeds);
        inputs_embeds = Tensor::mobilebert_conv_embed(inputs_embeds);
        inputs_embeds = embedding_transformation(inputs_embeds);
        auto position_embeds = position_embeddings(inputs[1]);
        auto type_embeds = token_type_embeddings(inputs[2]);
        auto embeddings = inputs_embeds + type_embeds + position_embeds;
        embeddings = layer_norm(embeddings);
        return {embeddings};
    }

private:
    Layer word_embeddings;
    Layer token_type_embeddings;
    Layer position_embeddings;
    Layer layer_norm;
    Layer embedding_transformation;
    Layer mobilebert_conv_embed;

    int embedding_size;
    int hidden_size;
    int embed_dim_multiplier;
};

class MobileBertSelfAttention : public Module {
    public:
        MobileBertSelfAttention() = default;
        MobileBertSelfAttention(MobileBertConfig &config, const string &base_name) {
            num_attention_heads = config.num_attention_heads;
            attention_head_size = int(config.true_hidden_size / config.num_attention_heads);
            all_head_size = num_attention_heads * attention_head_size;

            query_proj = Linear(config.true_hidden_size, all_head_size, true, base_name + config.names_config._q_proj_name);
            key_proj = Linear(config.true_hidden_size, all_head_size, true, base_name + config.names_config._k_proj_name);
            value_proj = Linear(config.hidden_size, all_head_size, true, base_name + config.names_config._v_proj_name);

            softmax = Softmax(DIMENSION, false, base_name + "softmax");
        }

        std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
            Tensor q = query_proj(inputs[0]);
            Tensor k = key_proj(inputs[1]);
            Tensor v = value_proj(inputs[2]);

            q = q.view(-1, num_attention_heads, -1, attention_head_size);
            k = k.view(-1, num_attention_heads, -1, attention_head_size);
            v = v.view(-1, num_attention_heads, -1, attention_head_size);
            
            k = k.transpose(SEQUENCE, DIMENSION);
            Tensor attention_scores = Tensor::mm(q, k);
            attention_scores = attention_scores / std::sqrt(attention_head_size);
            attention_scores = softmax(attention_scores);
            Tensor o = Tensor::mm(attention_scores, v);

            o = o.view(-1, 1, -1, all_head_size);
            return {o};       
        }

    private:
        int num_attention_heads;
        int attention_head_size;
        int all_head_size;

        Layer query_proj;
        Layer key_proj;
        Layer value_proj;
        Layer softmax;
};


class MobileBertSelfOutput : public Module {
    public:
        MobileBertSelfOutput() = default;
        MobileBertSelfOutput(MobileBertConfig &config, const string &base_name) {
            use_bottleneck = config.use_bottleneck;
            dense = Linear(config.true_hidden_size, config.true_hidden_size, true, base_name + config.names_config._dense_name);
            layer_norm = NoNorm(config.true_hidden_size, true,  base_name + config.names_config._layer_norm_name);
        }
        std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
            Tensor layer_outputs = dense(inputs[0]);
            layer_outputs = layer_norm(layer_outputs + inputs[1]);
            return {layer_outputs};
        }
    private:
        bool use_bottleneck;
        Layer dense;
        Layer layer_norm;
};

class MobileBertAttention : public Module {
    public:
        MobileBertAttention() = default;
        MobileBertAttention(MobileBertConfig &config, const string &base_name) {
            self = MobileBertSelfAttention(config, base_name+config.names_config.self_module);
            output = MobileBertSelfOutput(config, base_name+config.names_config.output_module);
        }
        std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
            std::vector<Tensor> self_outputs = self({inputs[0], inputs[1], inputs[2]});
            std::vector<Tensor> attention_output = output({self_outputs[0], inputs[3]});
            return attention_output;
        }

    private:
       MobileBertSelfOutput output;
       MobileBertSelfAttention self; 
};

class MobileBertIntermediate : public Module {
    public:
        MobileBertIntermediate() = default;
        MobileBertIntermediate(MobileBertConfig &config, const string &base_name) {
            dense = Linear(config.true_hidden_size, config.intermediate_size, true, base_name + config.names_config._dense_name);
            intermediate_act_fn = ACT_FN[config.hidden_act](base_name + config.names_config._act_name);
        }
        std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
            Tensor hidden_states = inputs[0];
            hidden_states = dense(hidden_states);
            hidden_states = intermediate_act_fn(hidden_states);
            return {hidden_states};
        }
    private:
        Layer dense;
        Layer intermediate_act_fn;
};

class OutputBottleneck : public Module {
    public:
        OutputBottleneck() = default;
        OutputBottleneck(MobileBertConfig &config, const string &base_name) {
            dense = Linear(config.true_hidden_size, config.hidden_size, true, base_name + config.names_config._dense_name);
            layer_norm = NoNorm(config.hidden_size, true, base_name + config.names_config._layer_norm_name);
        }
        std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
            Tensor layer_outputs = dense(inputs[0]);
            layer_outputs = layer_norm(layer_outputs + inputs[1]);
            return {layer_outputs};
        }
    private:
        Layer dense;
        Layer layer_norm;
};

class MobileBertOutput : public Module {
    public:
        MobileBertOutput() = default;
        MobileBertOutput(MobileBertConfig &config, const string &base_name) {
            dense = Linear(config.intermediate_size, config.true_hidden_size, true, base_name + config.names_config._dense_name);
            layer_norm = NoNorm(config.true_hidden_size, true,  base_name + config.names_config._layer_norm_name);
            bottleneck = OutputBottleneck(config, base_name + config.names_config.bottleneck_module);
        }
        std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
            Tensor layer_output = dense(inputs[0]);
            layer_output = layer_norm(layer_output + inputs[1]);
            return bottleneck({layer_output, inputs[2]});
        }

    private:
        Layer dense;
        Layer layer_norm;
        OutputBottleneck bottleneck;
};

class BottleneckLayer : public Module {
    public:
        BottleneckLayer() = default;
        BottleneckLayer(MobileBertConfig &config, const string &base_name) {
            dense = Linear(config.hidden_size, config.intra_bottleneck_size, true, base_name + config.names_config._dense_name);
            layer_norm = NoNorm(config.intra_bottleneck_size, true,  base_name + config.names_config._layer_norm_name);

        }
        std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
            Tensor layer_input = dense(inputs[0]);
            layer_input = layer_norm(layer_input);
            return {layer_input};
        }

    private:
        Layer dense;
        Layer layer_norm;
};

class Bottleneck : public Module {
    public:
        Bottleneck() = default;
        Bottleneck(MobileBertConfig &config, const string &base_name) {
            input = BottleneckLayer(config, base_name + config.names_config.input_module);
            attention = BottleneckLayer(config, base_name + config.names_config.attn_module);
        }
        std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
            std::vector<Tensor> bottlenecked_hidden_states = input(inputs);
            std::vector<Tensor> shared_attention_input = attention(inputs);
            return {shared_attention_input[0], shared_attention_input[0], inputs[0], bottlenecked_hidden_states[0]};
        }
    private:
        BottleneckLayer input;
        BottleneckLayer attention;

};

class FFNOutput : public Module {
    public:
        FFNOutput() = default;
        FFNOutput(MobileBertConfig &config, const string &base_name) {
            dense = Linear(config.intermediate_size, config.true_hidden_size, true, base_name + config.names_config._dense_name);
            layer_norm = NoNorm(config.true_hidden_size, true,  base_name + config.names_config._layer_norm_name);
        }
        std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
            Tensor layer_outputs = dense(inputs[0]);
            layer_outputs = layer_norm(layer_outputs + inputs[1]);
            return {layer_outputs};
        }
    
    private:
        Layer dense;
        Layer layer_norm;
};

class FFNLayer : public Module {
    public:
        FFNLayer() = default;
        FFNLayer(MobileBertConfig &config, const string &base_name) {
            intermediate = MobileBertIntermediate(config, base_name + config.names_config.intermediate_module);
            output = FFNOutput(config, base_name + config.names_config.output_module);
        }
        std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
            std::vector<Tensor> intermediate_output = intermediate(inputs);
            return output({intermediate_output[0], inputs[0]});
        }

    private:
        MobileBertIntermediate intermediate;
        FFNOutput output;
};

class MobileBertLayer : public Module {
public:
    MobileBertLayer() = default;
    MobileBertLayer(MobileBertConfig &config, const string &base_name) {

        num_feedforward_networks = config.num_feedforward_networks;

        attention = MobileBertAttention(config, base_name + config.names_config.attn_module);
        intermediate = MobileBertIntermediate(config, base_name + config.names_config.intermediate_module);
        output = MobileBertOutput(config, base_name + config.names_config.output_module);

        bottleneck = Bottleneck(config, base_name + config.names_config.bottleneck_module);
        ffn = List<FFNLayer>(num_feedforward_networks, config, base_name + config.names_config.ffn_module);

    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        Tensor hidden_states = inputs[0];
        std::vector<Tensor> tmp = bottleneck({hidden_states});
        Tensor query_tensor = tmp[0];
        Tensor key_tensor = tmp[1];
        Tensor value_tensor = tmp[2];
        Tensor layer_input = tmp[3];

        std::vector<Tensor> self_attention_outputs = attention({query_tensor, key_tensor, value_tensor, layer_input});
        Tensor attention_output = self_attention_outputs[0];
        for(FFNLayer& ffn_module: ffn){
            attention_output = ffn_module({attention_output})[0];
        }
        std::vector<Tensor> intermediate_output = intermediate({attention_output});
        std::vector<Tensor> layer_output = output({intermediate_output[0], attention_output, hidden_states});

        return {layer_output[0]};
    }

private:
    bool use_bottleneck;
    int num_feedforward_networks;
    MobileBertAttention attention;
    MobileBertIntermediate intermediate;
    MobileBertOutput output;
    Bottleneck bottleneck;
    std::vector<FFNLayer> ffn;
};

class MobileBertEncoder : public Module {
public:
    MobileBertEncoder() = default;
    MobileBertEncoder(MobileBertConfig &config, const string &base_name) {
        layer = List<MobileBertLayer>(config.num_hidden_layers, config,  base_name + config.names_config.layer_module);
    }
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        Tensor hidden_states = inputs[0];
        for(MobileBertLayer& layer_module : layer){
            std::vector<Tensor> layer_outputs = layer_module({hidden_states});
            hidden_states = layer_outputs[0];
        }
        return {hidden_states};
    }
private:
    std::vector<MobileBertLayer> layer;
};

class MobileBertPooler : public Module {
public:
    MobileBertPooler() = default;
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = inputs[0];
        x = x.clip({}, {}, {0}, {});
        return {x};
    }
};

class MobileBertHead : public Module {
public:
    MobileBertHead() = default;
    MobileBertHead(MobileBertConfig &config, const string &base_name) {
        head = Linear(config.hidden_size, config.num_classes, true, base_name + config.names_config._head_name);
    }
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        Tensor x = head(inputs[0]);
        return {x};
    }
private:
    Layer head;
};


class MobileBertModel : public Module {
public:
    MobileBertModel(MobileBertConfig &config) {
        embeddings = MobileBertEmbeddings(config, config.names_config.embedding_module);
        encoder = MobileBertEncoder(config, config.names_config.encoder_module);
        pooler = MobileBertPooler();
        // head = MobileBertHead(config, config.names_config.head_module);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = embeddings(inputs)[0];
        x = encoder({x})[0];
        x = pooler({x})[0];
        // x = head({x})[0];
        return {x};
    }

private:
    MobileBertEmbeddings embeddings;
    MobileBertEncoder encoder;
    MobileBertPooler pooler;
    MobileBertHead head;
};

#endif //! MODELING_MOBILEBERT_HPP