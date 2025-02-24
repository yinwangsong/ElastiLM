tune_ckpt_path='llama_0.4'
prune_ckpt_path='llama_0.4'

# export HF_HOME=/data/share
# export HF_ENDPOINT=https://hf-mirror.com

echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=$1 python3 ./LLMPruner/hf_prune.py --base_model "huggyllama/llama-7b" --prune_pth $prune_ckpt_path --pruning_ratio 0.75 --device cpu  --eval_device cuda --block_wise --anchor_layers 7 4 1 0 5 31 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --taylor param_first --save_model
echo "[FINISH] - Finish Pruning Model"

echo "[START] - Start Tuning"
CUDA_VISIBLE_DEVICES=$1 python3 ./LLMPruner/post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path --wandb_project llama_tune --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64
echo "[FINISH] - Finish Prune and Post-Training."
echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

echo "You can use the command:"
echo "       python3 generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path"
echo "to use the pruned model"