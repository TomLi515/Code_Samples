import glob
import logging
import math
import json
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List, Any, Dict, Optional, Union, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
import datasets
from datasets import load_dataset, IterableDataset, set_caching_enabled
from datasets.iterable_dataset import _BaseExamplesIterable, HasNextIterator, deepcopy, IterableDataset, Features, DatasetInfo
import torch.distributed as dist
set_caching_enabled(False)

import numpy as np
import transformers
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForTokenClassification,
    set_seed,
)
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import get_last_checkpoint, PredictionOutput, TrainOutput
from typing import Iterator, List, Optional
import numpy as np
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class MyArguments:
    data_path: str = field(
        default="",
        metadata={
            "help": (
                "data_path"
            )
        },
    )

    predict_output_path: str = field(
        default="",
        metadata={
            "help": (
                "predict_output_path"
            )
        },
    )

    max_eval_dataset_size: int = field(
        default=16384, 
        metadata={"help": "max_eval_dataset_size"}
    )

    eval_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to eval"}
    )

    eval_print_gen_example_count: int = field(
        default=8, 
        metadata={"help": "eval_print_gen_example_count"}
    )

    model_name: str = field(
        default="OpenMEDLab/PULSE-20bv5",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    tokenizer_name: str = field(
        default="OpenMEDLab/PULSE-20bv5",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    adapter_size: int = field(
        default=128, 
        metadata={"help": "adapter_size"}
    )

    rope_alpha: int = field(
        default=1, 
        metadata={"help": "rope_alpha"}
    )

    train_max_len: int = field(
        default=2048, 
        metadata={"help": "train_max_len"}
    )

    gen_max_len: int = field(
        default=512, 
        metadata={"help": "gen_max_len"}
    )

    pretrain_cut_step: int = field(
        default=1536, 
        metadata={"help": "gen_max_len"}
    )


    def __post_init__(self):
        pass



    def __post_init__(self):
        pass


class MyTrainer(Seq2SeqTrainer):


    def __init__(self, my_args: MyArguments, args: Seq2SeqTrainingArguments, **kwargs):

        from transformers import LlamaTokenizer

        self.train_dataset: IterableDataset
        self.eval_dataset: IterableDataset
        self.args: Seq2SeqTrainingArguments

        tokenizer = LlamaTokenizer.from_pretrained(
            my_args.tokenizer_name, 
            padding_side='left',
        )

        self.my_args = my_args

        def model_init():
            # from adapters.models.bloom.modeling_bloom import BloomForCausalLM
            from modeling_llama import LlamaForCausalLM
            # model
            model = LlamaForCausalLM.from_pretrained(
                my_args.model_name,
                pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
                torch_dtype=torch.bfloat16,
                adapter_size=my_args.adapter_size,
                rope_alpha=my_args.rope_alpha,
            )

            # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
            # on a small vocab and want a smaller embedding size, remove this test.
            embedding_size = model.config.vocab_size
            logger.warning(("resize_token_embeddings", len(tokenizer), embedding_size))

            if len(tokenizer) > embedding_size:
                model.resize_token_embeddings(len(tokenizer))

            # lock other prams except for fp
            # for name, p in model.named_parameters():
            #     if "adapter_" in name:
            #         p.requires_grad = True
            #     else:
            #         p.requires_grad = False

            if is_deepspeed_zero3_enabled():
                n_params = sum(dict((p.ds_id, p.ds_numel) for p in model.parameters()).values())
                trainable_n_params = sum(dict((p.ds_id, p.ds_numel) for p in model.parameters() if p.requires_grad).values())
            else:
                n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
                trainable_n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

            logger.info(f"Training new model from scratch - Trainable size={trainable_n_params/2**20:.2f}M params - Total size={n_params/2**20:.2f}M params")

            return model


        return super(MyTrainer, self).__init__(
            args=args,
            model=model_init(), 
            tokenizer=tokenizer,
            # train_dataset=raw_train_datasets.shuffle(seed=args.seed, buffer_size=100000),
            # eval_dataset=raw_eval_dataset.take(my_args.max_eval_dataset_size),
            data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer, padding="longest"),
            **kwargs
        )


    # make loss map
    def make_loss_map(self):
        tokenizer = self.tokenizer
        train_max_len = self.my_args.train_max_len
        prompt = 'Instructions: You are PULSE, a large language model of Transformer architecture trained by OpenMedLab. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-06-28'

        def predict_loss_map(batch):
            batch_input_ids = []
            batch_labels = []

            for doc, question, answer in zip(batch['doc'], batch['question'], batch['answer']):
                doc = doc[:train_max_len]

                input_ids = tokenizer(f"<|im_start|>{prompt}<|im_end|>").input_ids
                
                #Prompt for PULSE, Here's the English Translation:
                #"""
                #Goal:Respond appropriately based on the chat history. The response should be very detailed and presented in Markdown format.
                #Restrictions:
                #1.	When responding, refer to the document for information using the format No..
                #2.	If no relevant information is found in the document or the document contains “None,” 
                # ignore it and respond using GPT’s knowledge without mentioning having seen the document.
                #```{doc}```
                #Chat History:
                
                input_text = f'''目标：
                根据聊天记录回复合适的内容，回复需要非常详尽，回复采用Markdown格式。
                限制：
                1. 学习文档进行回复，可以使用[No.]的格式引用文档中的相关内容。
                2. 如果文档中没有相关信息或者为None，忽略文档并用GPT的知识回答问题，不要透露自己看到过文档。
                文档：

                [1]
                ```
                {doc}
                ```

                聊天记录:'''
                
                for item in question:
                    if item['speaker'] == "User":
                        input_text += f"\nUser: {item['text']}"
                    else:
                        input_text += f"\nHelper: {item['text']}"

                input_ids += tokenizer(f"<|im_start|>User: {input_text}<|im_end|>", add_special_tokens=False).input_ids

                labels = [-100] * len(input_ids)

                answer_ids = tokenizer(f"<|im_start|>Helper: {answer}<|im_end|>", add_special_tokens=False).input_ids

                input_ids += answer_ids
                labels += answer_ids

                batch_input_ids.append(input_ids[-train_max_len:])
                batch_labels.append(labels[-train_max_len:])

            return {
                "input_ids": batch_input_ids,
                "labels": batch_labels,
            }
        
        return predict_loss_map



    # Generative Prediction
    # test_dataset must contain 'dig', and the last conversation must be from the patient.
    # Generates the next sentence, which should be the doctor's response.
    def loss_predict(
        self, 
        test_dataset: IterableDataset, 
        ignore_keys: Optional[List[str]] = None, 
        metric_key_prefix: str = "test",
        dry_run=False,
        **gen_kwargs
    ) -> PredictionOutput:

        deal_test_dataset = test_dataset.map(
            self.make_loss_map(),
            batched=True,
            # num_proc=32,
            remove_columns=['id', "question", "answer", "doc"],
            # desc="Running " + metric_key_prefix + " - predict_map",
        )
        
        # Adjust parameters to enable loss mode.    
        self.args.predict_with_generate = False

        if dry_run:
            return PredictionOutput(predictions=tuple(), label_ids=tuple(), metrics={})

        tmp_predict_output = super(MyTrainer, self).predict(
            test_dataset=deal_test_dataset, 
            ignore_keys=ignore_keys, 
            metric_key_prefix=metric_key_prefix,
        )

        # only return metrics
        return PredictionOutput(predictions=tmp_predict_output.predictions, label_ids=None, metrics=None)



def main():

    parser = HfArgumentParser((MyArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        tmp_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        tmp_args = parser.parse_args_into_dataclasses()

    my_args: MyArguments = tmp_args[0]
    training_args: Seq2SeqTrainingArguments = tmp_args[1]

    os.makedirs(training_args.logging_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(filename=os.path.join(training_args.logging_dir, "train.log"), encoding="utf8")
        ],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.add_handler(
        logging.FileHandler(filename=os.path.join(training_args.logging_dir, "train.log"), encoding="utf8")
    )
    transformers.utils.logging.enable_explicit_format()

    # deepspeed logger
    from deepspeed.utils.logging import logger as deepspeed_logger
    deepspeed_logger.setLevel(log_level)
    deepspeed_logger.addHandler(
        logging.FileHandler(filename=os.path.join(training_args.logging_dir, "train.log"), encoding="utf8")
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info("CUDA_VISIBLE_DEVICES = " + str(os.environ.get("CUDA_VISIBLE_DEVICES")))

    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"My parameters {my_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize our Trainer
    trainer = MyTrainer(
        my_args=my_args,
        args=training_args,
    )

    for test_file_path in sorted(glob.glob(os.path.join(my_args.data_path, "**/*.jsonl"), recursive=True)):
        predict_file_path = test_file_path.replace(my_args.data_path, my_args.predict_output_path)
        logger.info(f"run eval on {test_file_path}")
        logger.info(f"save eval on {predict_file_path}")

        if os.path.exists(predict_file_path) == True:
            logger.info(f"{predict_file_path} is finish, continue")
            continue

        test_dataset = load_dataset(
            "json", 
            data_files=test_file_path, 
            split="train",
        )

        predict_output = trainer.loss_predict(
            test_dataset=test_dataset,
            dry_run=False,
        )

        if trainer.is_world_process_zero():
            os.makedirs(os.path.dirname(predict_file_path), exist_ok=True)

            assert len(test_dataset) == len(predict_output.predictions)

            with open(predict_file_path, "w", encoding="utf8") as f:
                for test_dataset_item, predict_output_item in zip(test_dataset, predict_output.predictions):
                    f.write(json.dumps({
                        "id": test_dataset_item["id"],
                        "question": test_dataset_item["question"],
                        "answer": test_dataset_item["answer"],
                        "doc": test_dataset_item["doc"],
                        "predict_answer": float(predict_output_item),
                    }, ensure_ascii=False) + "\n")

        dist.barrier()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()