# ChessCLIP evaluation

## Value/knowledge judgement evaluation
Modify the dataset loaded in eval_multi_choice.py file:
- ../eval_task/chess_opening_generation.json for Opening2PGN task.
- ../eval_task/chess_opening_mc.json for PGN2Opening task.
- ../eval_task/chess_annotation_nob.json for Annotation multi-choice task.
- ../eval_task/chess_state_value_multi_choice_2_nob.json for State value multi-choice task.

Then run it:
```bash
python3 eval_multi_choice.py
```

## Checkmate in one
- ../eval_task/chess_opening_generation.json for Opening2PGN

Then run it:
```bash
python3 eval_clip_checkmate_in_one.py
```
