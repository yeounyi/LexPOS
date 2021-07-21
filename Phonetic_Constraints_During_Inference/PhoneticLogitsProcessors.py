import inspect
import math
from abc import ABC
from typing import Callable, Iterable, List
import numpy as np
import torch
from transformers.file_utils import add_start_docstrings
from transformers.generation_logits_process import LogitsProcessor

class PhoneticLogitsProcessor(LogitsProcessor):
    def __init__(self, alpha: float, score_matrix):
        self.alpha = alpha
        self.score_matrix = score_matrix

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
                 decoder_attentions) -> torch.FloatTensor:
        # input_ids: (batch_size, sequence_length)
        # scores: (batch_size, vocab_size)

        # decoder_attentions: tuple of (batch_size, num_head, seq_len, seq_len)
        # (batch_size, sequence_length)
        # the last layer & last head's decoder attention in latest target token
        decoder_attentions = decoder_attentions[-1][:, -1, -1, :]

        # if sequence is shorter than 3
        if len(decoder_attentions[0]) < 3:
            return scores

        # max_attending_tokens except itself: (batch_size)
        max_attending_tokens = torch.argmax(decoder_attentions[:, :-1], dim=-1)
        max_attending_tokens_idx = [input_ids[batch_idx, t] for batch_idx, t in enumerate(max_attending_tokens)]

        # sim_matrix: (batch_size, vocab_size)
        # each row contains the phonetic cosine similarity score w.r.t corresponding max_attending_tokens
        for i, mati in enumerate(max_attending_tokens_idx):
            if i == 0:
                sim_matrix = self.score_matrix[mati].to(device='cuda')
            else:
                sim_matrix = torch.vstack((sim_matrix, self.score_matrix[mati].to(device='cuda')))

        scores = self.alpha * scores + (1 - self.alpha) * sim_matrix

        return scores