# ChessGPT - Bridging Policy Learning and Language Modeling
The official code for the paper: [ChessGPT - Bridging Policy Learning and Language Modeling](https://arxiv.org/abs/2306.09200).

## Contribution
We welcome any contribution, especially on chess related dataset towards the development of the next-generation model, ChessGPT-V2. For related matters, please contact xidong.feng.20@ucl.ac.uk.

## Model and Dataset
We open source our three models [ChessGPT-Base](https://huggingface.co/Waterhorse/chessgpt-base-v1), [ChessGPT-Chat](https://huggingface.co/Waterhorse/chessgpt-chat-v1) and [ChessCLIP](https://huggingface.co/Waterhorse/ChessCLIP), and our [Chess dataset](https://huggingface.co/datasets/Waterhorse/chess_data).

## Installation
1. Create a Python virtual environment with the method of your choice. The locked dependencies are generated for Python 3.8.10, but python beyond 3.8 like python 3.9/3.10 are also available.
2. Install dependencies by running `pip install -r requirements/dev.txt`.
3. Setup environment variables with the following commands.
```bash
export PYTHONPATH=$PWD:$PWD/third_party/chessclip/src
```
## Visualization
We offer a ChessCLIP visualization [demo](https://github.com/waterhorse1/ChessGPT/blob/main/chessclip_demo.ipynb) to show its capability.

## Training

### ChessCLIP 
We adopt the code from [open_clip-v2.9.3](https://github.com/mlfoundations/open_clip) for our training code of ChessCLIP. To reproduce our training, here are two procedures:
#### Generate dataset using tfds
Run tfds build for `pathtochessmastery`, `pgnlib`, `gameknot` and `lichess_studies`. Note that the processing of pgnlib needs fasttext's model, you can download it from their [official website](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin) and modify the [path](https://github.com/waterhorse1/ChessGPT/blob/dc99b5b1b9977d266828809b51316f9b961d22ff/chess_ai/datasets/tfds/pgn_base.py#L44). These 4 sources are free dataset and you can also add the source of `megabase` and `chess_publishing` if you buy them.
```bash
tfds build --imports chess_ai.datasets.tfds --overwrite pathtochessmastery --manual_dir ./chessclip_data/annotated_pgn \
--register_checksums '--beam_pipeline_options=runner=DirectRunner,direct_num_workers=8,direct_running_mode=multi_processing'
```

#### ChessCLIP training
After tfds building for all 4 sources, run the following code to train ChessCLIP:
```bash
cd chessclip/open_clip/src
torchrun --nproc_per_node 8 -m training.main_chess --model chessclip-quickgelu
```

### ChessGPT
#### Base-training
After downloading data from chessgpt_data, run the following code for conducting tokenization on all datasets,
```bash
cd chessgpt/data
python3 interleave_dataset.py --tokenizer_path ./tokenizer_path --data_path ./data_path --save_path ./save_path --max_seq_length 1024
```
After preparing the tokenized dataset, modify the hyperparameters and run base training:
```bash
cd chessgpt/train/clm_training
sh chess_ai/train/clm_traning/finetune_pp_peft_trainer.sh
```

#### Instruction-tuning
After running the base training, we can conduct further instruction-tuning based on instruction data or conversation data.
1. For dataset preparation, merge all data sources to one jsonl file.
2. After the dataset preparation, run chess_ai/train/sft_traning/train.sh.
   
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=20001 ./train.py \
    --model_name_or_path your_chess_base_model_path  \
    --data_path the_aggregated_one_jsonl_file_path \
    --bf16 True \
    --output_dir output_file \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'GPTNeoXLayer' \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
```
## Evaluation
Refer to `./eval` for evaluation dataset and code of ChessCLIP and ChessGPT.

## License
The code of ChessGPT/CLIP are released under the Apache License, Version 2.0.

## Citation
If you find ChessGPT useful, please cite it in your publications.

```bibtex
@article{feng2023chessgpt,
  title={ChessGPT: Bridging Policy Learning and Language Modeling},
  author={Feng, Xidong and Luo, Yicheng and Wang, Ziyan and Tang, Hongrui and Yang, Mengyue and Shao, Kun and Mguni, David and Du, Yali and Wang, Jun},
  journal={arXiv preprint arXiv:2306.09200},
  year={2023}
}
```
