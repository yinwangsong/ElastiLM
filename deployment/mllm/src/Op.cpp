
#include "Op.hpp"


#include <vector>
#include <set>
#include <Tensor.hpp>

namespace Elastilm {

    int RANK = 8;
    int SUBMODEL_NUM = 10;
    std::vector<std::vector<int>> submodel_attn_hidden_dims; // SUBMODEL_NUM * 2
    std::vector<std::vector<int>> submodel_mlp_hidden_dims;
    float submodel_lora_scale;
    std::set<int> anchor_layers;
    std::vector<int> layers_order;

    mllm::Tensor inner_rank_buffer;

    int LEVEL = 0;
    int CUR_LAYER_ID = 0;
    int IS_ANCHOR_LAYER = 0;

};

namespace mllm {

DataType Op::no_load_weights_dtype_ = MLLM_TYPE_F32;

}