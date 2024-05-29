# Training
## Prerequisite
Make sure you have created the environment and downloaded the data according to [README](../README.md).

## Llama-2

### Pre-Training

```bash
# prepare 2B data (packing texts from the same source to form sequences of 8K length)
# you only need to run this command once
python -m main.pretrain_data --output_dir data/pretrain/llama-8K_2B --num_token 8192:2b --model_name_or_path meta-llama/Llama-2-7b-chat-hf

output_name=ultragist-llama2-7b-chat-pt

torchrun --nproc_per_node 8 -m main.train \
--output_dir data/outputs/$output_name \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--train_data data/pretrain/llama-8K_2B \
--enable_ultragist \
--ultragist_window 1024 \
--ultragist_stride 1024 \
--ultragist_attn step-expansion \
--ultragist_attend_prev True \
--ultragist_sink_size 1 \
--ultragist_ratio 2 4 8 16 32 \
--ultragist_ratio_mix step-random \
--ultragist_param q k v o \
--gradient_checkpointing \
--use_reentrant False \
--save_only_model \
--save_strategy steps \
--evaluation_strategy steps \
--num_train_epochs 1 \
--save_steps 0.49 \
--logging_steps 50 \
--bf16 \
--deepspeed data/deepspeed/stage2.json
```

### Fine-Tuning
```bash
# prepare 100M data that are evenly distributed across all domains to prevent forgetting during fine-tuning
python -m main.pretrain_data --output_dir data/pretrain/llama-8K_100M-even --num_token 8192:100m --config data/config/even.json

output_name=ultragist-llama2-7b-chat-ft

torchrun --nproc_per_node 8 -m main.train \
--output_dir data/outputs/$output_name \
--model_name_or_path data/outputs/ultragist-llama2-7b-chat-pt/checkpoint-xxxxx \
--train_data ultragist:gpt/one_detail_book.train.8K.json ultragist:gpt/one_detail_paper.train.8K.json ultragist:longalpaca/train.json ultragist:booksum/train.8K.json ultragist:needle/train.8K.json  data/pretrain/llama-8K_100M-even[5000] \
--max_length 10240 \
--min_length 7200 \
--group_by_stride strict \
--enable_ultragist \
--ultragist_window 1024 \
--ultragist_stride 1024 \
--ultragist_attn step-expansion \
--ultragist_attend_prev True \
--ultragist_sink_size 1 \
--ultragist_ratio 2 4 8 \
--ultragist_ratio_mix step-random \
--ultragist_param q k v o \
--learning_rate 1e-5 \
--gradient_checkpointing \
--use_reentrant False \
--save_only_model \
--num_train_epochs 1 \
--save_strategy epoch \
--logging_steps 50 \
--bf16 \
--chat_template llama-2 \
--deepspeed data/deepspeed/stage2.json
```

## Mistral
### Pre-Training
```bash
# prepare 2B data (packing texts from the same source to form sequences of 16K length)
# you only need to run this command once
python -m main.pretrain_data --output_dir data/pretrain/mistral-16K_2B/ --num_token 16384:2b --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2

output_name=ultragist-llama2-7b-chat-pt

torchrun --nproc_per_node 8 -m main.train \
--output_dir data/outputs/$output_name \
--model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
--train_data data/pretrain/mistral-2B-16K/ \
--enable_ultragist \
--ultragist_window 2048 \
--ultragist_stride 2048 \
--ultragist_attn step-expansion \
--ultragist_attend_prev False \
--ultragist_sink_size 1 \
--ultragist_ratio 2 4 8 16 32 \
--ultragist_ratio_mix step-random \
--ultragist_param q k v o \
--gradient_checkpointing \
--use_reentrant False \
--save_only_model \
--num_train_epochs 1 \
--save_strategy steps \
--save_steps 0.49 \
--logging_steps 50 \
--bf16 \
--deepspeed data/deepspeed/stage2.json
```

### Fine-Tuning
```bash
# prepare 100M data that are evenly distributed across all domains to prevent forgetting during fine-tuning
python -m main.pretrain_data --output_dir data/pretrain/mistral-16K_100M-even --num_token 16384:100m --config data/config/even.json --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2

output_name=ultragist-mistral-7b-inst-ft

torchrun --nproc_per_node 8 -m main.train \
--output_dir data/outputs/$output_name \
--model_name_or_path data/outputs/ultragist-mistral-7b-inst-pt/checkpoint-xxxxx \
--train_data ultragist:gpt/one_detail_book.train.16K.json ultragist:gpt/one_detail_paper.train.16K.json ultragist:longalpaca/train.json ultragist:booksum/train.16K.json ultragist:needle/train.16K.json data/pretrain/mistral-16K_100M-even[5000] \
--max_length 20480 \
--min_length 7200 \
--group_by_stride strict \
--enable_ultragist \
--ultragist_window 2048 \
--ultragist_stride 2048 \
--ultragist_attn step-expansion \
--ultragist_attend_prev False \
--ultragist_sink_size 1 \
--ultragist_ratio 2 4 8 \
--ultragist_ratio_mix step-random \
--ultragist_param q k v o \
--learning_rate 1e-5 \
--gradient_checkpointing \
--use_reentrant False \
--save_only_model \
--num_train_epochs 1 \
--save_strategy epoch \
--logging_steps 50 \
--bf16 \
--chat_template mistral \
--deepspeed data/deepspeed/stage2.json
```