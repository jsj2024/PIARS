#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module defines the PIARSDataset, a PyTorch Dataset for loading and preprocessing
data for training models with the PIARS methodology. It handles various data sources,
including single-turn and multi-turn conversations, and prepares them into specific
formats for "erase," "retain," and "validation" sets.
"""

import json
import os
import random
from typing import Dict, List, Any

import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

# Set a fixed seed for reproducibility
random.seed(0)

class PIARSDataset(Dataset):
    """
    A custom PyTorch Dataset to prepare data for PIARS training.

    This class loads and processes several distinct datasets:
    1.  Erase Set (Boundary Set): Harmful prompts paired with undesirable responses,
        intended to be "forgotten" by the model.
    2.  Retain Set: Benign or safe instruction-response pairs that the model
        should continue to generate correctly.
    3.  Multi-turn Dialogue Set: Preprocessed multi-turn conversations used to evaluate
        and guide the model's behavior in extended interactions.
    4.  Validation Set: A separate set for evaluating the model's performance.

    It dynamically configures chat templates based on the model name and tokenizes
    the data into formats suitable for training.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        num_examples (int): The number of examples to load from the general-purpose retain set.
        lorra_args: A configuration object containing arguments like file paths and flags.
        model_name_or_path (str): The identifier of the pre-trained model to configure templates for.
    """
    
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 num_examples: int,
                 lorra_args: Any, # Use `Any` for argparse-like objects
                 model_name_or_path: str):
        super().__init__()

        self.tokenizer = tokenizer
        self.model_name_or_path = model_name_or_path.lower()
        
        # --- Configuration from lorra_args ---
        self.multi_turn_data_path = lorra_args.multi_turn_data_path
        self.boundary_data_size = lorra_args.boundary_data_size
        self.k_max = getattr(lorra_args, 'k_max', 5)  # Max turns in multi-turn dialogues
        self.use_unlearn_loss = getattr(lorra_args, 'use_unlearn_loss', False)

        # --- Tokenization settings ---
        self.max_length = 1024
        self.cb_max_length = 512 # Max length for circuit breaker components (request/response)

        # --- Configure model-specific chat templates and settings ---
        self._configure_model_template()

        # --- Load and preprocess all required datasets ---
        self.multi_turn_data = self._load_multi_turn_data()
        self.x_boundary_orig = self._load_erase_set()
        self.orig_s_retain = self._load_retain_set(num_examples)
        self.val_orig = self._load_validation_set()


    def _configure_model_template(self):
        """Sets model-specific chat templates, tags, and behavior flags."""
        model_configs = {
            'llama': {"user": "<|start_header_id|>user<|end_header_id|>\n\n", "assistant": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "sep": "", "switch": [0, 1], "refusal": True},
            'mistral': {"user": "[INST] ", "assistant": " [/INST]", "sep": " ", "switch": [0], "refusal": False},
            'qwen': {"user": "<|im_start|>user\n", "assistant": "<|im_end|>\n<|im_start|>assistant\n", "sep": "<|im_end|>", "switch": [0, 1], "refusal": True},
            'gemma': {"user": "<start_of_turn>user\n", "assistant": "<end_of_turn>\n<start_of_turn>model\n", "sep": " ", "switch": [0, 1], "refusal": True},
            'deepseek': {"user": "<|im_start|>user\n", "assistant": "<|im_end|>\n<|im_start|>assistant\n", "sep": "<|im_end|>", "switch": [0, 1], "refusal": True},
        }

        config_key = next((key for key in model_configs if key in self.model_name_or_path), None)

        if not config_key:
            raise NotImplementedError(f"Model config for '{self.model_name_or_path}' not found.")

        print(f"USING {config_key.upper()} TEMPLATE")
        config = model_configs[config_key]
        self.user_tag = config["user"]
        self.assistant_tag = config["assistant"]
        self.sep_token = config["sep"]
        self.switch_select = config["switch"]
        self.use_refusal_retain = config["refusal"]
        
        # Special handling for Mistral's chat template if needed
        if config_key == 'mistral':
            self.tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

    def _load_multi_turn_data(self) -> List[Dict]:
        """Loads and validates preprocessed multi-turn conversation data."""
        if not self.multi_turn_data_path:
            print("No multi-turn data path provided. Skipping.")
            return []

        if not os.path.exists(self.multi_turn_data_path):
            raise FileNotFoundError(f"Multi-turn data file not found at: {self.multi_turn_data_path}")
        
        print(f"Loading multi-turn data from {self.multi_turn_data_path}")
        with open(self.multi_turn_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data:
            print("Multi-turn data file is empty.")
            return []
        
        # Validate data format
        if 'turns' not in data[0]:
            raise ValueError(f"Data in {self.multi_turn_data_path} is not preprocessed. "
                             f"Please run the preprocessing script first.")

        print(f"Loaded {len(data)} preprocessed multi-turn conversations.")
        return data

    def _format_one_shot(self, instruction: str, response: str) -> str:
        """Formats a single instruction-response pair using the model's template."""
        return f"{self.user_tag}{instruction}{self.assistant_tag}<SEPARATOR>{response}"

    def _load_erase_set(self) -> List[Any]:
        """Loads and prepares the erase set (x_boundary) from multiple sources."""
        print("Loading erase set...")
        with open("data/train/circuit_breakers_train_2400.json", 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        with open("data/train/Safe_train_600.json", 'r', encoding='utf-8') as f:
            dataset.extend(json.load(f))
        
        x_boundary = []
        for d in tqdm(dataset, desc="Processing erase set"):
            instruction = d['prompt'] if np.random.choice(self.switch_select) == 0 else ""
            formatted_input = self._format_one_shot(instruction, d['output'])
            x_boundary.append(formatted_input)

        random.shuffle(x_boundary)
        print(f"Initial erase set size: {len(x_boundary)}")
        return x_boundary

    def _load_retain_set(self, num_examples: int) -> List[Any]:
        """Loads and prepares the retain set (orig_s) from multiple sources."""
        print("Loading retain set...")
        # 1. General Safe QA from UltraChat
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        orig_s = []
        for example in tqdm(ds, desc="Processing UltraChat", total=num_examples):
            if len(orig_s) >= num_examples:
                break
            messages = example["messages"]
            if len(messages) < 2:
                continue

            if np.random.choice(self.switch_select) == 0:
                # Apply the model-specific chat template
                template_applied = self.tokenizer.apply_chat_template(messages, tokenize=False)
                if self.tokenizer.bos_token:
                    template_applied = template_applied.replace(self.tokenizer.bos_token, "")
                orig_s.append(template_applied)
            else:
                # Use a simplified template with an empty instruction
                orig_s.append(self._format_one_shot("", messages[1]["content"]))
        
        print(f"Loaded {len(orig_s)} examples from UltraChat.")
        
        # 2. Harmful Question + Safe Refusal Answer
        if self.use_refusal_retain:
            self._add_refusal_retain_data(orig_s)

        random.shuffle(orig_s)
        return orig_s

    def _add_refusal_retain_data(self, retain_set: List[str]):
        """Adds harmful prompts paired with safe refusals to the retain set."""
        print("Adding refusal retain data...")
        with open("data/train/circuit_breakers_train_2400.json", 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Determine which model's safe output to use
        output_key_map = {'qwen': 'qwen_output', 'gemma': 'gemma_output', 'deepseek': 'llama3_output'}
        output_label = next((v for k, v in output_key_map.items() if k in self.model_name_or_path), 'llama3_output')
        print(f"Using '{output_label}' for refusal responses.")

        random.shuffle(dataset)
        dataset = dataset[:2000] # Limit the number of examples

        for d in tqdm(dataset, desc="Processing refusal retain"):
            instruction = d['prompt'] if np.random.choice(self.switch_select) == 0 else ""
            formatted_input = self._format_one_shot(instruction, d[output_label])
            retain_set.append(formatted_input)

    def _load_validation_set(self) -> List[str]:
        """Loads the validation dataset."""
        print("Loading validation set...")
        with open("data/train/circuit_breakers_val.json", 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        val_orig = []
        for d in tqdm(dataset, desc="Processing validation set"):
            val_orig.append(self._format_one_shot(d['prompt'], d['output']))
        return val_orig
    
    def _add_boundary_safe_data(self):
        """
        Injects boundary-safe QA pairs into both the retain and erase sets.
        This feature is currently unused as it was mixed in __init__ but is kept
        here for potential future use. It would be called from __init__ if enabled.
        """
        if self.boundary_data_size <= 0:
            return
            
        print(f"Adding {self.boundary_data_size} boundary-safe examples...")
        with open('data/train/ORbench_retain_set.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Filter for fully compliant examples
            data = [row for row in data if row.get('status') == "1_full_compliance"]
            data = data[:self.boundary_data_size]

        refusal_response = 'I cannot provide information or guidance on illegal or harmful activities. Can I help you with something else?'
        for d in data:
            # Add the safe, compliant pair to the retain set
            retain_formatted = self._format_one_shot(d['prompt'], d['completion'])
            self.orig_s_retain.append(retain_formatted)
            
            # Add the prompt with a generic refusal to the erase (boundary) set
            # This is structured as a dict to be handled specially in __getitem__
            bs_formatted = self._format_one_shot(d['prompt'], refusal_response)
            self.x_boundary_orig.append({"overrefusalQA": bs_formatted, "retainQA": retain_formatted})
            
        random.shuffle(self.x_boundary_orig)
        random.shuffle(self.orig_s_retain)
        print(f"Erase set size after adding boundary data: {len(self.x_boundary_orig)}")
        print(f"Retain set size after adding boundary data: {len(self.orig_s_retain)}")

    def __len__(self) -> int:
        """Returns the size of the dataset, constrained by the smaller of the retain or erase sets."""
        return min(len(self.orig_s_retain), len(self.x_boundary_orig))

    def _sample_multi_turn_data(self) -> Dict:
        """Randomly samples one conversation and extracts its turns."""
        if not self.multi_turn_data:
            return {}
        
        conversation = random.choice(self.multi_turn_data)
        turns_data = {}
        
        # Iterate from the second turn (k=2) up to k_max or the last available turn
        max_turn_in_conv = conversation.get('max_turn', 0)
        for k in range(2, min(self.k_max + 1, max_turn_in_conv + 1)):
            turn_index = k - 1 # list is 0-indexed
            if turn_index < len(conversation['turns']):
                turn = conversation['turns'][turn_index]
                turns_data[k] = {
                    'history': turn.get('history', ''),
                    'safe_alternative': turn.get('safe_alternative', ''),
                    'induction_degree': turn.get('induction_degree', 0.0),
                    'is_harmful': turn.get('is_harmful', False)
                }
        
        return turns_data

    def _tokenize_and_split(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenizes a text string that is split by '<SEPARATOR>' into request and response."""
        parts = text.split('<SEPARATOR>')
        if len(parts) != 2:
            # Fallback for improperly formatted data
            cb_request, cb_response = text, ""
        else:
            cb_request, cb_response = parts

        cb_tokenized_kwargs = dict(max_length=self.cb_max_length, padding='max_length', truncation=True, return_tensors="pt")

        self.tokenizer.padding_side = "left"
        tokenized_request = self.tokenizer(cb_request, **cb_tokenized_kwargs)

        self.tokenizer.padding_side = "right"
        tokenized_response = self.tokenizer(cb_response, add_special_tokens=False, **cb_tokenized_kwargs)

        self.tokenizer.padding_side = "left" # Reset to default

        return {
            "request_ids": tokenized_request["input_ids"],
            "request_mask": tokenized_request["attention_mask"],
            "response_ids": tokenized_response["input_ids"],
            "response_mask": tokenized_response["attention_mask"],
        }
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Retrieves and tokenizes a single data item, including erase, retain, validation,
        and multi-turn samples.
        """
        # --- Get raw data samples ---
        retain_sample = self.orig_s_retain[index]
        boundary_sample = self.x_boundary_orig[index]
        val_sample = self.val_orig[index % len(self.val_orig)] # Use modulo for validation set
        
        # --- General tokenization settings ---
        tokenize_kwargs = dict(max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        # --- Process Erase (Boundary) Sample ---
        is_boundary_pair = isinstance(boundary_sample, dict)
        if is_boundary_pair:
            # This is a special overrefusal case from boundary-safe data
            retain_sample = boundary_sample['retainQA']
            boundary_text = boundary_sample['overrefusalQA']
        else:
            boundary_text = boundary_sample

        boundary_tokens = self._tokenize_and_split(boundary_text)
        combined_ids_boundary = torch.cat([boundary_tokens["request_ids"], boundary_tokens["response_ids"]], dim=1)
        combined_mask_boundary = torch.cat([boundary_tokens["request_mask"], boundary_tokens["response_mask"]], dim=1)
        
        # Label mask for boundary should only cover the request part
        label_mask_boundary = torch.cat([boundary_tokens['request_mask'], torch.zeros_like(boundary_tokens["response_mask"])], dim=1)
        
        # --- Process Retain Sample ---
        tokenized_retain = self.tokenizer(retain_sample.replace('<SEPARATOR>', self.sep_token), **tokenize_kwargs)
        
        # --- Process Validation Sample ---
        tokenized_val = self.tokenizer(val_sample.replace('<SEPARATOR>', self.sep_token), **tokenize_kwargs)

        # --- Process Multi-turn Sample ---
        multi_turn_sample = self._sample_multi_turn_data()
        mt_data = {}
        for k in range(2, self.k_max + 1):
            turn_data = multi_turn_sample.get(k)
            if turn_data:
                history_tok = self.tokenizer(turn_data['history'], **tokenize_kwargs)
                safe_alt_tok = self.tokenizer(turn_data['safe_alternative'], **tokenize_kwargs)
                
                mt_data[f'history_input_ids_{k}'] = history_tok['input_ids']
                mt_data[f'history_attention_mask_{k}'] = history_tok['attention_mask']
                mt_data[f'safe_alt_input_ids_{k}'] = safe_alt_tok['input_ids']
                mt_data[f'safe_alt_attention_mask_{k}'] = safe_alt_tok['attention_mask']
                mt_data[f'induction_degree_{k}'] = torch.tensor(turn_data['induction_degree'], dtype=torch.float)
            else:
                # Pad with zeros if a turn is missing to maintain consistent tensor shapes in the batch
                zeros = torch.zeros((1, self.max_length), dtype=torch.long)
                mt_data[f'history_input_ids_{k}'] = zeros
                mt_data[f'history_attention_mask_{k}'] = zeros
                mt_data[f'safe_alt_input_ids_{k}'] = zeros
                mt_data[f'safe_alt_attention_mask_{k}'] = zeros
                mt_data[f'induction_degree_{k}'] = torch.tensor(0.0, dtype=torch.float)

        # --- Assemble final dictionary ---
        result = {
            "input_ids_x_boundary": combined_ids_boundary.squeeze(0),
            "attention_mask_x_boundary": combined_mask_boundary.squeeze(0),
            "label_mask_x_boundary": label_mask_boundary.squeeze(0),
            
            "input_ids_retain": tokenized_retain['input_ids'].squeeze(0),
            "attention_mask_retain": tokenized_retain['attention_mask'].squeeze(0),
            "label_mask_retain": tokenized_retain['attention_mask'].squeeze(0), # By default, learn from the whole sequence
            
            "input_ids_val": tokenized_val["input_ids"].squeeze(0),
            "attention_mask_val": tokenized_val["attention_mask"].squeeze(0),
            
            "is_boundary": is_boundary_pair
        }
        
        # Add multi-turn data and squeeze tensors to remove batch dimension
        for key, value in mt_data.items():
            result[key] = value.squeeze(0) if isinstance(value, torch.Tensor) else value
            
        return result