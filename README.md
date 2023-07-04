# ChessGPT - Bridging Policy Learning and Language Modeling
The official code for the paper: [ChessGPT - Bridging Policy Learning and Language Modeling](https://arxiv.org/abs/2306.09200).

## Model and Dataset
We open source our three models [ChessGPT-Base](https://huggingface.co/Waterhorse/chessgpt-base-v1), [ChessGPT-Chat](https://huggingface.co/Waterhorse/chessgpt-chat-v1) and [ChessCLIP](https://huggingface.co/Waterhorse/ChessCLIP), and our [Chess dataset](https://huggingface.co/datasets/Waterhorse/chess_data).

## Installation
1. Create a Python virtual environment with the method of your choice. The locked dependencies are generated for Python 3.8.10, but python beyond 3.8 like python 3.9/3.10 are also available.
2. Install dependencies by running `pip install -r requirements/dev.txt`.
3. Setup environment variables with the following commands.
```bash
export PYTHONPATH=$PWD:$PWD/third_party/chessclip/src
```

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
1. Firstly Run all pipleines in chess_ai/data/pipelines to form all formatted Jsonl files.
2. Run dataset tokenization and merging/shuffling to form a fully tokenized dataset. Check out the README there.
3. For basemodel finetuning, run chess_ai/train/clm_traning/finetune_pp_peft_trainer.sh.

#### Instruction-tuning
After running the base-training, we can conduct further instrustion-tuning based one instruction data or conversation data.
1. For dataset preparation, Check chess_ai/data/sft_data/README.md for prepare conversation data from different sources.
2. After the dataset preparation, run chess_ai/train/sft_traning/train.sh.

## Evaluation
Refer to `./eval` for evaluation dataset and code of ChessCLIP and ChessGPT.

## License
ChessGPT/CLIP are released under the Apache License, Version 2.0.

## Citation
@article{feng2023chessgpt,
  title={ChessGPT: Bridging Policy Learning and Language Modeling},
  author={Feng, Xidong and Luo, Yicheng and Wang, Ziyan and Tang, Hongrui and Yang, Mengyue and Shao, Kun and Mguni, David and Du, Yali and Wang, Jun},
  journal={arXiv preprint arXiv:2306.09200},
  year={2023}
}
