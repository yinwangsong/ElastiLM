#ifndef MODELING_ELASTICMLP_HPP
#define MODELING_ELASTICMLP_HPP

#include "Backend.hpp"
#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"

#include "../llama/configuration_llama.hpp"
using namespace mllm;

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
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        vector<int> activate_dims = std::any_cast<vector<int>>(args[0]);
        int activate_dim = activate_dims[0];
        auto x = gate_proj(inputs[0], -1, activate_dim);
        x = silu(x);
        auto y = up_proj(inputs[0], -1, activate_dim);
        x = x * y;
        x = down_proj(x, activate_dim, -1);
        return {x};
    }
};


#endif //! MODELING_ELASTICMLP_HPP