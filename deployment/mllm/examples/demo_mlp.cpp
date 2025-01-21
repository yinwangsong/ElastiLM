#include "models/mlp/modeling_mlp.hpp"
#include "cmdline.h"
#include <vector>


int main(int argc, char *argv[]) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/mlp-fp32.mllm");
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string model_path = cmdParser.get<string>("model");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto model = MLP("");
    model.load(model_path);
    // model.setNoLoadWeightsDtype(MLLM_TYPE_F32);

    Tensor x(1, 1, 128, 128, Backend::global_backends[MLLM_CPU], true);
    x.setName("input");
    Tensor::tensor_status = TENSOR_STATIC_INIT;
    x.setTtype(INPUT_TENSOR);
    for (int i = 0; i < 128; ++i) {
        for (int j = 0; j < 123; ++j) {
            x.setDataAt<float>(0, 0, i, j, 1.0);
        }
    }

    Tensor y = model({x})[0];
    y.printDataTorchLike<float>();
    return 0;
}
