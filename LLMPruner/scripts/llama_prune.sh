
prune_ckpt_path='llama_0.2'
echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=0 python ./LLMPruner/hf_prune.py --prune_pth $prune_ckpt_path --pruning_ratio 0.9 --device cpu  --eval_device cuda --block_wise --anchor_layers 7 4 1 0 5 31 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --taylor param_first --save_model
echo "[FINISH] - Finish Pruning Model"

prune_ckpt_path='llama_0.3'
echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=0 python ./LLMPruner/hf_prune.py --prune_pth $prune_ckpt_path --pruning_ratio 0.875 --device cpu  --eval_device cuda --block_wise --anchor_layers 7 4 1 0 5 31 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --taylor param_first --save_model
echo "[FINISH] - Finish Pruning Model"

prune_ckpt_path='llama_0.4'
echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=0 python ./LLMPruner/hf_prune.py --prune_pth $prune_ckpt_path --pruning_ratio 0.75 --device cpu  --eval_device cuda --block_wise --anchor_layers 7 4 1 0 5 31 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --taylor param_first --save_model
echo "[FINISH] - Finish Pruning Model"

prune_ckpt_path='llama_0.5'
echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=0 python ./LLMPruner/hf_prune.py --prune_pth $prune_ckpt_path --pruning_ratio 0.625 --device cpu  --eval_device cuda --block_wise --anchor_layers 7 4 1 0 5 31 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --taylor param_first --save_model
echo "[FINISH] - Finish Pruning Model"

prune_ckpt_path='llama_0.6'
echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=0 python ./LLMPruner/hf_prune.py --prune_pth $prune_ckpt_path --pruning_ratio 0.5 --device cpu  --eval_device cuda --block_wise --anchor_layers 7 4 1 0 5 31 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --taylor param_first --save_model
echo "[FINISH] - Finish Pruning Model"

prune_ckpt_path='llama_0.7'
echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=0 python ./LLMPruner/hf_prune.py --prune_pth $prune_ckpt_path --pruning_ratio 0.375 --device cpu  --eval_device cuda --block_wise --anchor_layers 7 4 1 0 5 31 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --taylor param_first --save_model
echo "[FINISH] - Finish Pruning Model"

prune_ckpt_path='llama_0.8'
echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=0 python ./LLMPruner/hf_prune.py --prune_pth $prune_ckpt_path --pruning_ratio 0.25 --device cpu  --eval_device cuda --block_wise --anchor_layers 7 4 1 0 5 31 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --taylor param_first --save_model
echo "[FINISH] - Finish Pruning Model"

prune_ckpt_path='llama_0.9'
echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=0 python ./LLMPruner/hf_prune.py --prune_pth $prune_ckpt_path --pruning_ratio 0.125 --device cpu  --eval_device cuda --block_wise --anchor_layers 7 4 1 0 5 31 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --taylor param_first --save_model
echo "[FINISH] - Finish Pruning Model"