# chess-project

## Installation
1. Create a Python virtual environment with the method of your choice.
The locked dependences is generated for Python 3.8.10.
2. Install dependencies by running `pip install -r requirements/dev.txt`.
3. Setup environment variables with the following commands.
```
export PYTHONPATH=$PWD:$PWD/third_party:$PWD/third_party/open_clip/src
```
5. Run tests with `./run_test.sh`

## Run CLIP
```
cd third_party/open_clip/src
torchrun --nproc_per_node 8 -m training.main_chess --model ViT-B-32-quickgelu-chess
```

## Download datasets
To download the raw datasets, run `bash scripts/download_all_data.sh <path to download dir>.`
This will download all PGN datasets crawled by us and put it into the correct folder.
To download a specific subset, refer to `scripts/download_*.sh` scripts.

## ChessGPT
### Base-training
1. Firstly Run all pipleines in chess_ai/data/pipelines to form all formatted Jsonl files.
2. Run dataset tokenization and merging/shuffling to form a fully tokenized dataset. Checkout the README there.
3. For basemodel finetuning, run chess_ai/train/clm_traning/finetune_pp_peft_trainer.sh.

### Instruction-tuning
After running the base-training, we can conduct further instrustion-tuning based one instruction data or conversation data.
1. For dataset preparation, Check chess_ai/data/sft_data/README.md for prepare conversation data from different sources.
2. After the dataset preparation, run chess_ai/train/sft_traning/train.sh.
