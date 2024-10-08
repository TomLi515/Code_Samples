# M-MeD: A New Dataset and Graph-Based Explorations for Enhancing RAG Retrieval in Medical Care

## train.py

This is the code from one of the experiments in my project “M-MeD: A New Dataset and Graph-Based Explorations for Enhancing RAG Retrieval in Medical Care,” which has been submitted to AAAI-2025. \
After the preliminary process, the input of the task includes a series of unranked and ranked retrieved medical documents from a medical database, as part of the Retrieval Augmented Generation (RAG) process. \
In this code, I aim to design a framework that optimizes the retrieved documents by learning from the ranked documents. Each multi-turn dialogue has ten retrieved documents. I have innovatively designed a combination of RoBERTa and graph models like GCN and GAT to derive multi-turn dialogue embeddings. Document embeddings, derived from RoBERTa, calculate the similarity with the multi-turn dialogue embeddings. A Soft Pairwise Loss, specially tailored for retrieval and ranking tasks, is applied during training.

## train_gcn.sh

This is the .sh file that sets up and runs train.py. bf16, gradient checkpointing, and gradient accumulation are applied.

## gcn_neg6_turn_final.log

The sample log file that records the training setups, training process, and training results.
