# ChessGPT - Bridging Policy Learning and Language Modeling

## Model and Dataset
We open source our three models [ChessGPT-Base](https://huggingface.co/Waterhorse/chessgpt-base-v1), [ChessGPT-Chat](https://huggingface.co/Waterhorse/chessgpt-chat-v1) and [ChessCLIP](https://huggingface.co/Waterhorse/ChessCLIP), and our [Chess dataset](https://huggingface.co/Waterhorse/ChessCLIP).

## Installation
1. Create a Python virtual environment with the method of your choice. The locked dependences is generated for Python 3.8.10, but python beyond 3.8 like python 3.9/3.10 are also available.
2. Install dependencies by running `pip install -r requirements/dev.txt`.
3. Setup environment variables with the following commands.
```bash
export PYTHONPATH=$PWD:$PWD/third_party/chessclip/src
```

## Training

### ChessCLIP 
We adopt the code from [open_clip-v2.9.3](https://github.com/mlfoundations/open_clip) for our training code of ChessCLIP. To reproduce our training, here are two procedures:
#### Generate dataset using tfds

#### ChessCLIP training
```bash
cd chessclip/open_clip/src
torchrun --nproc_per_node 8 -m training.main_chess --model chessclip-quickgelu
```

### ChessGPT
#### Base-training
1. Firstly Run all pipleines in chess_ai/data/pipelines to form all formatted Jsonl files.
2. Run dataset tokenization and merging/shuffling to form a fully tokenized dataset. Checkout the README there.
3. For basemodel finetuning, run chess_ai/train/clm_traning/finetune_pp_peft_trainer.sh.

#### Instruction-tuning
After running the base-training, we can conduct further instrustion-tuning based one instruction data or conversation data.
1. For dataset preparation, Check chess_ai/data/sft_data/README.md for prepare conversation data from different sources.
2. After the dataset preparation, run chess_ai/train/sft_traning/train.sh.

## Evaluation

## License

## Citation
Will be updated soon.
