# Evaluation
## Prerequisite
Make sure you have created the environment and downloaded the data according to [README](../README.md).

## Llama
```bash
model_id=namespace-Pt/ultragist-llama2-7b-chat
# you can also evaluate your models
# model_id=data/outputs/ultragist-llama2-7b-chat-ft/checkpoint-xxxx

########### Topic Retrieval ##########
for ratio in 4 8 16 24
do
# the default window 1024 cannot be evenly divided by 24, so we change it to 960
if [[ $ratio == "24" ]]; then
    torchrun --nproc_per_node 8 -m main.eval_topic --ultragist_ratio $ratio --model_name_or_path $model_id --enable_ultragist --num_topic 1 2 3 10 --ultragist_window 960 --ultragist_stride 960
else
    torchrun --nproc_per_node 8 -m main.eval_topic --ultragist_ratio $ratio --model_name_or_path $model_id --enable_ultragist --num_topic 1 2 3 10
fi
done

########### MSC ##########
for ratio in 4 8 16 24
do
# the default window 1024 cannot be evenly divided by 24, so we change it to 960
if [[ $ratio == "24" ]]; then
    torchrun --nproc_per_node 8 -m main.eval_msc --ultragist_ratio $ratio --model_name_or_path $model_id --enable_ultragist --chat_template no --ultragist_window 960 --ultragist_stride 960
else
    torchrun --nproc_per_node 8 -m main.eval_msc --ultragist_ratio $ratio --model_name_or_path $model_id --enable_ultragist --chat_template no
fi
done

########### Long-Context Tasks ##########
torchrun --nproc_per_node 8 -m main.eval_longbench --model_name_or_path $model_id --enable_ultragist --ultragist_ratio 0 2 4 8 16 32

########### Needle-In-A-Haystack ##########
torchrun --nproc_per_node 8 -m main.eval_needle --model_name_or_path $model_id --enable_ultragist --max_length 32000 --ultragist_ratio 2 4 8 --ultragist_ratio_mix adapt-1024 --rope_method dynamic --rope_factor 2

# by default, we evaluate with ROUGE-L (R), you can specify OPENAI_API_KEY to use gpt-3.5 as evaluator
# OPENAI_API_KEY="<you_api_key>" torchrun --nproc_per_node 8 -m main.eval_needle --model_name_or_path $model_id --enable_ultragist --max_length 32000 --ultragist_ratio 2 4 8 --ultragist_ratio_mix adapt-1024 --rope_method dynamic --rope_factor 2 --gpt_eval

########### ShareGPT ##########
for turn in 1 2 3
do
torchrun --nproc_per_node 8 -m main.eval_multiturn --model_name_or_path $model_id --enable_ultragist --ultragist_ratio 8 --ultragist_window 512 --ultragist_stride 512 --num_turn $turn
done
```

## Mistral
```bash
model_id=namespace-Pt/ultragist-mistral-7b-inst
# you can also evaluate your models
# model_id=data/outputs/ultragist-mistral-7b-inst-ft/checkpoint-xxxx

########### Topic Retrieval ##########
for ratio in 4 8 16 24
do
# the default window 2048 cannot be evenly divided by 24, so we change it to 960
if [[ $ratio == "24" ]]; then
    torchrun --nproc_per_node 8 -m main.eval_topic --ultragist_ratio $ratio --model_name_or_path $model_id --enable_ultragist --num_topic 1 2 3 10 --ultragist_window 960 --ultragist_stride 960 --chat_template mistral
else
    torchrun --nproc_per_node 8 -m main.eval_topic --ultragist_ratio $ratio --model_name_or_path $model_id --enable_ultragist --num_topic 1 2 3 10 --chat_template mistral
fi
done

########### MSC ##########
for ratio in 4 8 16 24
do
# the default window 2048 cannot be evenly divided by 24, so we change it to 960
if [[ $ratio == "24" ]]; then
    torchrun --nproc_per_node 8 -m main.eval_msc --ultragist_ratio $ratio --model_name_or_path $model_id --enable_ultragist --chat_template no --ultragist_window 960 --ultragist_stride 960
else
    torchrun --nproc_per_node 8 -m main.eval_msc --ultragist_ratio $ratio --model_name_or_path $model_id --enable_ultragist --chat_template no
fi
done

########### Long-Context Tasks ##########
torchrun --nproc_per_node 8 -m main.eval_longbench --model_name_or_path $model_id --enable_ultragist --ultragist_ratio 2 4 8 16 32 --chat_template mistral --beacon_ratio_mix adapt-1024

########### Needle-In-A-Haystack ##########
torchrun --nproc_per_node 8 -m main.eval_needle --model_name_or_path $model_id --enable_ultragist --max_length 128000 --chat_template mistral --ultragist_ratio 2 4 8 --beacon_ratio_mix adapt-1024 --rope_method dynamic --rope_factor 2

# by default, we evaluate with ROUGE-L (R), you can specify OPENAI_API_KEY to use gpt-3.5 as evaluator
# OPENAI_API_KEY="<you_api_key>" torchrun --nproc_per_node 8 -m main.eval_needle --model_name_or_path $model_id --enable_ultragist --max_length 128000 --chat_template mistral --ultragist_ratio 2 4 8 --beacon_ratio_mix adapt-1024 --rope_method dynamic --rope_factor 2 --gpt_eval

########### ShareGPT ##########
for turn in 1 2 3
do
torchrun --nproc_per_node 8 -m main.eval_multiturn --model_name_or_path $model_id --enable_ultragist --ultragist_ratio 8 --ultragist_window 512 --ultragist_stride 512 --num_turn $turn --chat_template mistral
done
```
