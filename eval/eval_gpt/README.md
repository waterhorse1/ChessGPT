# Chess evaluation

## Pre-requisites
Our evlaution based on the opensorce project [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).

Please install it first:
```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

Then, replace the bigbench evaluation file.
```bash
cd ChessGPT/eval/eval_gpt/
cp -f bigbench.py lm-evaluation-harness/lm_eval/tasks/
```
Finally, copy the test file into the bigbench tasks folder.
```bash
cp test.json lm-evaluation-harness/lm_eval/datasets/bigbench_resources/
```

Test the installation:
```bash
cd lm-evaluation-harness
python main.py     --model hf-causal   \
--model_args pretrained=/path/to/your/llm-model/ \
--tasks bigbench_test  \
--device cuda:0
```

## Chess State Tracking 
Copy evaluation json files to bigbench tasks folder.
```bash
cp -r ../eval_task/chess_state_tracking/* lm-evaluation-harness/lm_eval/datasets/bigbench_resources/
```

There are 6 tasks for chess state tracking:
- bigbench_real_long_task
- bigbench_real_med_task
- bigbench_real_short_task
- bigbench_synthetic_long_task
- bigbench_synthetic_med_task
- bigbench_synthetic_short_task

Run the evluation, for example:
```bash
cd lm-evaluation-harness
python main.py     --model hf-causal   \
--model_args pretrained=/path/to/your/llm-model/ \
--tasks bigbench_real_long_task  \
--device cuda:0
```

## Board State Tracking

Copy evaluation json files to bigbench tasks folder.
```bash
cp -r ../eval_task/board_state_tracking/* lm-evaluation-harness/lm_eval/datasets/bigbench_resources/
```
There are 6 tasks for board state tracking:
- bigbench_real_long_task_u2f
- bigbench_real_med_task_u2f
- bigbench_real_short_task_u2f
- bigbench_real_long_task_p2f
- bigbench_real_med_task_p2f
- bigbench_real_short_task_p2f

Run the evaluation, for example:
```bash
cd lm-evaluation-harness
python main.py     --model hf-causal   \
--model_args pretrained=/path/to/your/llm-model/ \
--tasks bigbench_real_long_task_u2f  \
--device cuda:0
```
## Chess state value

Copy evaluation json files to bigbench tasks folder.
```bash
cp -r ../eval_task/chess_state_value/* lm-evaluation-harness/lm_eval/datasets/bigbench_resources/
```

There are several variants:
- chess_state_value_multi_choice_2.json (prompts with { suffix, multi-choice for base model)
- chess_state_value_multi_choice_2_nob.json (prompts without { suffix, multi-choice for base model)
- chess_state_value_multi_choice_sft.json (prompts with { suffix, exact multi-choice for ChessGPT-Chat)
- chess_state_value_multi_choice_sft_nob.json (prompts without { suffix, multi-choice for ChessGPT-Chat)

Run the evaluation, for example:
```bash
cd lm-evaluation-harness
python main.py     --model hf-causal   \
--model_args pretrained=/path/to/your/llm-model/ \
--tasks bigbench_chess_state_value_multi_choice_2  \
--device cuda:0
```
## Chess annotation

Copy evaluation json files to bigbench tasks folder.
```bash
cp -r ../eval_task/chess_annotation/* lm-evaluation-harness/lm_eval/datasets/bigbench_resources/
```
The variants are the same with Chess state value variations.

## Chess opening

Copy evaluation json files to bigbench tasks folder.
```bash
cp -r ../eval_task/chess_opening/* lm-evaluation-harness/lm_eval/datasets/bigbench_resources/
```

There are several variants:
- chess_opening_generation.json (Opening2PGN for base model)
- chess_opening_generation_sft.json (Opening2PGN for ChessGPT-Chat)
- chess_opening_mc.json (PGN2Opening for base model)
- chess_opening_mc_sft.json (PGN2Opening for ChessGPT-Chat)

## Checkmate in one

Copy evaluation json files to bigbench tasks folder.
```bash
cp -r ../eval_task/checkmate_in_one/* lm-evaluation-harness/lm_eval/datasets/bigbench_resources/
```

There are several variants:
- checkmate_in_one_comment_esm/mc.json (prompts with { suffix, exact string match/multi-choice for base model)
- checkmate_in_one_esm/mc.json (prompts without { suffix, exact string match/multi-choice for base model)
- checkmate_in_one_comment_sft_esm/mc.json (prompts with { suffix, exact string match/multi-choice for ChessGPT-Chat)
- checkmate_in_one_sft_esm/mc.json (prompts without { suffix, exact string match/multi-choice for ChessGPT-Chat)

Run the evaluation, for example:
```bash
cd lm-evaluation-harness
python main.py     --model hf-causal   \
--model_args pretrained=/path/to/your/llm-model/ \
--tasks bigbench_checkmate_in_one_comment_esm  \
--device cuda:0
```
## General Policy Evaluation 

Copy evaluation json files to bigbench tasks folder.
```bash
cp -r ../eval_task/general_policy/* lm-evaluation-harness/lm_eval/datasets/bigbench_resources/
```

There are 8 tasks for general policy:
- bigbench_chess_win_rate_white_gap0
- bigbench_chess_win_rate_white_gap500
- bigbench_chess_win_rate_white_gap1000
- bigbench_chess_win_rate_white_gap2000
- bigbench_chess_win_rate_black_gap0
- bigbench_chess_win_rate_black_gap500
- bigbench_chess_win_rate_black_gap1000
- bigbench_chess_win_rate_black_gap2000

Run the evaluation, for example:
```bash
cd lm-evaluation-harness
python main.py     --model hf-causal   \
--model_args pretrained=/path/to/your/llm-model/ \
--tasks bigbench_chess_win_rate_white_gap1000  \
--device cuda:0
```
