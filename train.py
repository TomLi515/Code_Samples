"""
Demonstration of Coding ability:
This is the code from one of the experiments in my project “M-MeD: A New Dataset and Graph-Based Explorations for Enhancing RAG Retrieval in Medical Care,” \
which has been submitted to AAAI-2025. After the preliminary process, the input of the task includes a series of unranked and ranked retrieved medical documents from a medical database, \
as part of the Retrieval Augmented Generation (RAG) process. In this code, I aim to design a framework that optimizes the retrieved documents by learning from the ranked documents. \
Each multi-turn dialogue has ten retrieved documents. I have innovatively designed a combination of RoBERTa and graph models like GCN and GAT to derive multi-turn dialogue embeddings. \
Document embeddings, derived from RoBERTa, calculate the similarity with the multi-turn dialogue embeddings. A Soft Pairwise Loss, specially tailored for retrieval and ranking tasks, is applied during training.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import List, Optional, Tuple, Union
from transformers import (
  BertTokenizerFast,
  AutoModel,
)
import datasets
import numpy as np
import torch
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from datasets import load_dataset
import transformers
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers import AutoModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import PaddingStrategy, send_example_telemetry
from transformers import Trainer, EarlyStoppingCallback

from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GCNConv



logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    q_model_name_or_path: str = field(
        default="", metadata={"help": "q_model_name_or_path"}
    )
    q_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "q_tokenizer_name"}
    )
    q_model_type: str = field(
        default="gcn", metadata={"help": "q_model_type"}
    )

    d_model_name_or_path: str = field(
        default="", metadata={"help": "d_model_name_or_path"}
    )
    d_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "d_tokenizer_name"}
    )

    score_temperature: float = field(
        default=64.0,
        metadata={"help": "score_temperature"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "The input test data file (a text file)."})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    
    q_max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If passed, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    d_max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If passed, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to the maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json", "jsonl"], "`train_file` should be a csv or a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "jsonl"], "`validation_file` should be a csv or a json file."
        if self.test_file is not None:
            extension = self.test_file.split(".")[-1]
            assert extension in ["csv", "json", "jsonl"], "`test_file` should be a csv or a json file."

@dataclass
class CustomTrainingArguments(TrainingArguments):
    do_test: bool = field(default=False, metadata={"help": "Whether to run testing."})
    early_stopping: bool = field(default=False, metadata={"help": "Enable early stopping."})
    early_stopping_patience: int = field(default=3, metadata={"help": "Number of evaluations to wait before stopping if no improvement."})
    early_stopping_threshold: float = field(default=0.0, metadata={"help": "Minimum improvement to qualify as an improvement."})


@dataclass
class DataCollatorForQD:
    q_tokenizer: PreTrainedTokenizerBase
    d_tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        import torch
        from torch_geometric.data import Data
        from networkx.readwrite import json_graph

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features]

        q_graphs = []

        for feature in features:
            input_ids = torch.tensor(feature['q_graphs']['input_ids'], dtype=torch.long)
            attention_masks = torch.tensor(feature['q_graphs']['attention_mask'], dtype=torch.long)
            graph = feature['q_graphs']['graph']

            node_mapping = {node['id']: idx for idx, node in enumerate(graph['nodes'])}
            edges = [(node_mapping[edge['source']], node_mapping[edge['target']]) for edge in graph['links']]
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

            q_graphs.append(Data(x=input_ids, edge_index=edge_index, attention_mask=attention_masks))



        batch_size = len(features)
        num_choices = len(features[0]["d_input_ids"])

        d_batch = self.d_tokenizer.pad(
            {
                "input_ids": list(chain(*[feature['d_input_ids'] for feature in features])),
                "attention_mask": list(chain(*[feature['d_attention_mask'] for feature in features])),
                "token_type_ids": list(chain(*[feature['d_token_type_ids'] for feature in features])),
            },
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {
            "q_graphs": q_graphs,
            "d_input_ids": d_batch['input_ids'].view((batch_size, num_choices, -1)),
            "d_attention_mask": d_batch['attention_mask'].view((batch_size, num_choices, -1)),
            "d_token_type_ids": d_batch['token_type_ids'].view((batch_size, num_choices, -1)),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        return batch


#Evaluation Metrics  
def dense_rank(values):
    unique_values = sorted(set(values))
    ranks = {value: rank for rank, value in enumerate(unique_values, start=0)}
    return [ranks[value] for value in values]

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

def mrr_at_k(r, k):
    r = np.asarray(r)[:k]  # 获取前k个结果
    ranks = np.argwhere(r) + 1  # 获取相关结果的排名（位置）
    if ranks.size:
        return (1. / ranks[0]).item()
    return 0

def hit_rate_at_k(r, k):
    r = np.asarray(r)[:k]
    if np.any(r):
        return 1.0
    return 0.0

def average_precision_at_k(r, k):
    r = np.asarray(r)[:k]
    relevant = np.argwhere(r).flatten()
    if relevant.size == 0:
        return 0.0
    precisions = [np.mean(r[:i+1]) for i in relevant]  # Precision at each relevant hit
    return np.mean(precisions)

# Muli-turn Dialogue Embedding - GCN + RoBERTa
class QMeanModel(nn.Module):
    def __init__(self, model_args,bert_model_name='uer/roberta-base-wwm-chinese-cluecorpussmall'):
        super(QMeanModel, self).__init__()
        self.bert_model = AutoModel.from_pretrained(bert_model_name, add_pooling_layer=False)
        self.input_dim = 768  
        self.hidden_dim = 32  
        self.output_dim = 768  

        self.conv1 = GCNConv(self.input_dim, self.hidden_dim)
        self.dropout1 = nn.Dropout(0.1) 
        self.conv2 = GCNConv(self.hidden_dim, self.output_dim)

    def forward(self, q_graphs):
        all_input_ids = []
        all_attention_masks = []
        node_count_per_graph = []

        # Combination
        for graph in q_graphs:
            all_input_ids.append(graph.x)
            all_attention_masks.append(graph.attention_mask)
            node_count_per_graph.append(graph.x.size(0))
    
        all_input_ids = torch.cat(all_input_ids, dim=0)
        all_attention_masks = torch.cat(all_attention_masks, dim=0)

        # RoBERTa
        bert_outputs = self.bert_model(all_input_ids, attention_mask=all_attention_masks)
        all_node_features = bert_outputs.last_hidden_state

        # Split
        all_graph_features = []
        start_idx = 0
        for i, graph_size in enumerate(node_count_per_graph):
            node_features = all_node_features[start_idx:start_idx + graph_size]
            attention_mask = all_attention_masks[start_idx:start_idx + graph_size][:, :, None]
            node_features = torch.sum(node_features * attention_mask, dim=-2) / torch.sum(attention_mask, dim=-2)
            
            #GCN
            edge_index = q_graphs[i].edge_index

            x = self.conv1(node_features, edge_index)
            x = F.relu(x)
            x = self.dropout1(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = x.mean(dim=0)
            
            all_graph_features.append(x)
            start_idx += graph_size

        all_graph_features = torch.stack(all_graph_features, dim=0)
        return all_graph_features 
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        return self.bert_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        return self.bert_model.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self.bert_model.is_gradient_checkpointing


# Document Embeddings - RoBERTa
class DModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(model_args.d_model_name_or_path, add_pooling_layer=False)

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        batch_size = input_ids.size(0)
        d_count = input_ids.size(1)

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        last_hidden_state = self.bert_model(
            flat_input_ids, 
            flat_position_ids, 
            flat_token_type_ids,
            flat_attention_mask,
            return_dict=True
        ).last_hidden_state

        flat_attention_mask = flat_attention_mask[:,:,None]

        o = torch.sum(last_hidden_state * flat_attention_mask, dim=-2) / torch.sum(flat_attention_mask, dim=-2)
        o = o.view((batch_size, d_count, o.size(-1)))

        return o
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        return self.bert_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        return self.bert_model.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self.bert_model.is_gradient_checkpointing


class QDModel(nn.Module):

    def __init__(self, model_args):
        super().__init__()
        self.q_model = QMeanModel(model_args)  
        
        self.d_model = DModel(model_args)  

        self.score_temperature: float = model_args.score_temperature

    def forward(
        self, 
        q_graphs,  
        d_input_ids: torch.Tensor,
        d_attention_mask: torch.Tensor,
        d_token_type_ids: Optional[torch.Tensor] = None,
        d_position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Process the graph data for the queries
        q_h = self.q_model(q_graphs)
        q_h = F.normalize(q_h, p=2, dim=-1)

        # Process the document data
        d_h = self.d_model(
            d_input_ids,
            d_attention_mask,
            d_token_type_ids,
            d_position_ids,
        )
        d_h = F.normalize(d_h, p=2, dim=-1)

        # Compute the cosine similarity score
        score = -torch.einsum('bj,bdj->bd', q_h, d_h)
       
        #Soft Pairwise Loss
        score_sub = (score[:,:,None] - score[:,None,:]).type(torch.float32) / self.score_temperature
        label_sub = ((labels[:,:,None] - labels[:,None,:]) < 0).type_as(score_sub)
        score_sub = score_sub.view((-1, 1))
        loss = torch.logsumexp(torch.cat([
            torch.zeros_like(score_sub),
            score_sub,
        ], dim=-1), dim=-1)
        

        label_sub = label_sub.view((-1))

        loss = torch.sum(loss * label_sub) / torch.sum(label_sub)

        return (loss, score)


    #Gradient Checkpointing
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.q_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        self.d_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.q_model.gradient_checkpointing_disable()
        self.d_model.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self.q_model.is_gradient_checkpointing and self.d_model.is_gradient_checkpointing


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: CustomTrainingArguments

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_swag", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters data_args={model_args}")
    logger.info(f"Training/evaluation parameters data_args={data_args}")
    logger.info(f"Training/evaluation parameters training_args={training_args}")

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

    if data_args.train_file is not None or data_args.validation_file is not None:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]

        if extension == "jsonl":
            extension = "json"
        
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
        )
    else:
        assert "need train_file validation_file"

    #Load RoBERTa tokenizer
    q_tokenizer = AutoTokenizer.from_pretrained(
        model_args.q_tokenizer_name if model_args.q_tokenizer_name else model_args.q_model_name_or_path,
    )

    d_tokenizer = AutoTokenizer.from_pretrained(
        model_args.d_tokenizer_name if model_args.d_tokenizer_name else model_args.d_model_name_or_path,
    )


    model = QDModel(model_args)
    if data_args.q_max_seq_length is None:
        q_max_seq_length = q_tokenizer.model_max_length
        if q_max_seq_length > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            q_max_seq_length = 1024
    else:
        if data_args.q_max_seq_length > q_tokenizer.model_max_length:
            logger.warning(
                f"The q_max_seq_length passed ({data_args.q_max_seq_length}) is larger than the maximum length for the "
                f"model ({q_tokenizer.model_max_length}). Using q_max_seq_length={q_tokenizer.model_max_length}."
            )
        q_max_seq_length = min(data_args.q_max_seq_length, q_tokenizer.model_max_length)


    if data_args.d_max_seq_length is None:
        d_max_seq_length = q_tokenizer.model_max_length
        if d_max_seq_length > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            d_max_seq_length = 1024
    else:
        if data_args.d_max_seq_length > d_tokenizer.model_max_length:
            logger.warning(
                f"The d_max_seq_length passed ({data_args.d_max_seq_length}) is larger than the maximum length for the "
                f"model ({d_tokenizer.model_max_length}). Using d_max_seq_length={d_tokenizer.model_max_length}."
            )
        d_max_seq_length = min(data_args.d_max_seq_length, d_tokenizer.model_max_length)

    pad_to_max_length = data_args.pad_to_max_length

    logger.info(f"d_max_seq_length={d_max_seq_length}")
    logger.info(f"pad_to_max_length={pad_to_max_length}")


    
    # Preprocessing the datasets.
    def preprocess_function(examples):
        import networkx as nx
        from networkx.readwrite import json_graph
        import json
        
        #Graph Construction
        def load_graph_from_triplets(triples):
            G = nx.DiGraph()
            for head, tail, relation in triples:
                G.add_edge(head, tail, relation=relation)
                G.add_node(head, token=head)  # Add node attribute
                G.add_node(tail, token=tail)  # Add node attribute
            return G

        q_graphs = []
        for triples in examples['triple']:
            graph = load_graph_from_triplets(triples)
            nodes = [graph.nodes[node]['token'] for node in graph.nodes()]
            tokenized_nodes = q_tokenizer(
            nodes, 
            truncation=True, 
            max_length=10, 
            padding="max_length",
            return_tensors="pt"
        ) 
        

            graph_data = {
            'graph': json_graph.node_link_data(graph),
            'input_ids': tokenized_nodes['input_ids'],
            'attention_mask': tokenized_nodes['attention_mask'],
        }
            q_graphs.append(graph_data)



        d_input_ids = []
        d_attention_mask = []
        d_token_type_ids = []

        for doc in examples['doc']:
            tokenized_d = d_tokenizer(
            doc,
            truncation=True,
            max_length=d_max_seq_length,
            padding="max_length" if pad_to_max_length else False,
        )

            d_input_ids.append(tokenized_d['input_ids'])
            d_attention_mask.append(tokenized_d['attention_mask'])
            d_token_type_ids.append(tokenized_d['token_type_ids'])

        output = {
        "q_graphs": q_graphs,
        "d_input_ids": d_input_ids,
        "d_attention_mask": d_attention_mask,
        "d_token_type_ids": d_token_type_ids,
        "labels": [dense_rank(item) for item in examples['scores']],
    }

        return output



    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
 
        train_dataset = train_dataset.select(range(len(train_dataset)))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=False,
                
            )
            train_dataset = train_dataset.shuffle(training_args.seed)
            #print(train_dataset[0].keys)

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]

        eval_dataset = eval_dataset.select(range(len(eval_dataset)))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=False,
            )
            
    if training_args.do_test and "test" in raw_datasets:
        test_dataset = raw_datasets["test"]
        test_dataset = test_dataset.select(range(len(test_dataset)))
        with training_args.main_process_first(desc="test dataset map pre-processing"):
            test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=False,
        )
            
    # Data collator
    data_collator = DataCollatorForQD(q_tokenizer=q_tokenizer, d_tokenizer=d_tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)


    # Metric
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions

        ndcgs = []
        mrrs = []
        aps = []
        hit_rates = []
        k_value_mrr = 10
        k_value_hit = 1  

        for prediction, label_id in zip(predictions.tolist(), label_ids.tolist()):
            m_label_id = max(label_id)

            prediction_r = [
            m_label_id - item
            for _, item in sorted(zip(prediction, label_id), key=lambda x: x[0], reverse=True)
        ]

            ndcgs.append(ndcg_at_k(prediction_r, k=len(prediction_r)))
            mrrs.append(mrr_at_k(prediction_r, k=k_value_mrr))
            hit_rates.append(hit_rate_at_k(prediction_r, k=k_value_hit))
            aps.append(average_precision_at_k(prediction_r, k=len(prediction_r)))

        mean_ndcg = np.mean(ndcgs).item()
        mean_mrr = np.mean(mrrs).item()
        mean_hit_rate = np.mean(hit_rates).item()
        mean_ap = np.mean(aps).item()
        

        return {"ndcg": mean_ndcg, "mrr": mean_mrr, "hit_rate": mean_hit_rate,"map": mean_ap}

    callbacks = []
    if training_args.early_stopping:
        callbacks.append(EarlyStoppingCallback(
        early_stopping_patience=training_args.early_stopping_patience,
        early_stopping_threshold=training_args.early_stopping_threshold
        ))

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=q_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

       
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    #Test
    if training_args.do_test:
        logger.info("*** Test ***")
        metrics = trainer.evaluate(eval_dataset=test_dataset)
        metrics["test_samples"] = len(test_dataset)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()