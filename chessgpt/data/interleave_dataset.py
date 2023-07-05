from datasets import load_dataset, concatenate_datasets, load_from_disk
import os
import transformers
from itertools import chain
import argparse
import random
import shutil
import copy
import warnings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str,default='/RedPajama-INCITE-Base-3B-v1/')
    parser.add_argument("--data_path", type=str,default='./json_data/')
    parser.add_argument("--save_path", type=str,default='./training_tokenized/')
    parser.add_argument("--max_seq_length", type=int, default=1024)

    args = parser.parse_args()

    # This can control the ratio of each subset
    sub_dir_dict = {
        'pro_player': 1,
        'annotated_pgn': 1,
        'chess_book': 1,
        'c4': 1, 
        'ccrl': 1,
        'chess_specific_crawl': 1,
        'medium': 1,
        'forum': 1,
        'chess_puzzle': 1,
        #'oscar': 1,
        'stackexchange': 1,
        'wikipedia': 1,
        'pile': 1,
        'lichess_db_37': 1,
        'redpajama': 1,
        "chess_modeling": 1
    }
    main_dir = args.data_path
    all_dataset_list = []
    for sd in sub_dir_dict.keys():
        print(f"Working on Subset {sd}")
        if not sd in os.listdir(main_dir):
            warnings.warn(f"Cannot find subset {sd}", UserWarning)
            continue
        jsonl_path = os.path.join(main_dir, sd) + '/*.jsonl*'
        dataset = load_dataset('json', data_files={'train': jsonl_path}).remove_columns("metadata")

        dataset = dataset['train'].shuffle(seed=42)

        tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)
        column_names = list(dataset.features)
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            modified_text = ['<|endoftext|>' + element + '<|endoftext|>' for element in examples['text']]
            output = tokenizer(modified_text)
            return output
        
        tokenized_datasets = dataset.map(
                        tokenize_function,
                        batched=True,
                        num_proc=32,
                        remove_columns=column_names,
                        keep_in_memory=True,
                        desc="Running tokenizer on dataset",
                    )

        block_size = args.max_seq_length
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            #print(total_length)
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            return result

        lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=32,
                    keep_in_memory=True,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
        for _ in range(sub_dir_dict[sd]):
            all_dataset_list.append(copy.deepcopy(lm_datasets))
    
    lm_datasets = concatenate_datasets(all_dataset_list)
    lm_datasets = lm_datasets.shuffle(seed=42)
    lm_datasets.save_to_disk(args.save_path)
    print('finish ' + args.save_path)

if __name__ == '__main__':
    main()
