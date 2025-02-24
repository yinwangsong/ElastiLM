CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/Contextual_sparsity/generate_c4_mlp_data.py --model "llama"
CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/Contextual_sparsity/train_llama.py --model "llama"


CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/Contextual_sparsity/generate_c4_mlp_data.py --model "llama3"
CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/Contextual_sparsity/train_llama3.py --model "llama3"

CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/Contextual_sparsity/generate_c4_mlp_data.py --model "llama3_instruct"
CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/Contextual_sparsity/train_llama3_instruct.py --model "llama3_instruct"

CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/Contextual_sparsity/generate_c4_mlp_data.py --model "vicuna"
CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/Contextual_sparsity/train_vicuna.py --model "vicuna"

CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/Contextual_sparsity/generate_c4_mlp_data.py --model "orcamini"
CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/Contextual_sparsity/train_orcamini.py --model "orcamini"

