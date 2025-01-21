#ifndef MODELING_MLP_HPP
#define MODELING_MLP_HPP

#include "Backend.hpp"
#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
using namespace mllm;

class MLP : public Module {
public:
    MLP() = default;
    MLP(const string &base_name) {
        mlp = ElasticLinear(128, 123, true, base_name + "mlp");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        Tensor x = mlp(inputs[0]);
        return {x};
    }

private:
    Layer mlp;
};

#endif //! MODELING_MLP_HPP