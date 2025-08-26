#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script preprocesses multi-turn conversation data to accelerate training for PIARS.

It processes a JSON file containing conversations, extracts relevant information for each
assistant turn, calculates metrics like an "induction degree," and generates a new
structured JSON file suitable for model training.

Key functionalities include:
- Extracting conversation history up to a specific turn.
- Creating a "safe alternative" history by replacing harmful content.
- Calculating an induction degree based on inducement and harmfulness scores.
- Structuring the data by conversation and by turn, including harmfulness flags.
- Providing command-line arguments for specifying input/output files and max turns.
"""

import json
import os
import argparse
from typing import List, Dict, Union

from tqdm import tqdm

# --- Constants for dictionary keys to avoid typos and improve maintainability ---
ROLE = "role"
CONTENT = "content"
SAFE_CONTENT = "safe_content"
USER = "user"
ASSISTANT = "assistant"
DUAL_EVALUATION = "dual_evaluation"
INDUCEMENT_SCORE = "inducement_score"
HARMFULNESS_SCORE = "harmfulness_score"


def _build_conversation_string(conversation: List[Dict],
                               turn_idx: int,
                               use_safe_content: bool = False) -> str:
    """
    A helper function to build a formatted string representation of the conversation history.

    Args:
        conversation (List[Dict]): The list of conversation turns.
        turn_idx (int): The index of the current turn to include in the history.
        use_safe_content (bool): If True, uses 'safe_content' for the assistant's
                                 replies when available. Defaults to False.

    Returns:
        str: A formatted string of the conversation history.
    """
    history_lines = []
    # Iterate through the conversation up to the specified turn index
    for i in range(turn_idx + 1):
        if i < len(conversation):
            turn = conversation[i]
            role = turn.get(ROLE)
            
            if role == USER:
                history_lines.append(f"User: {turn.get(CONTENT, '')}")
            elif role == ASSISTANT:
                # Choose content based on the 'use_safe_content' flag
                if use_safe_content:
                    content = turn.get(SAFE_CONTENT, turn.get(CONTENT, ''))
                else:
                    content = turn.get(CONTENT, '')
                history_lines.append(f"Assistant: {content}")
                
    return "\n".join(history_lines)


def calculate_induction_degree(turn_data: Dict) -> float:
    """
    Calculates the induction degree (ID) by combining inducement and harmfulness scores.
    The formula is: ID = (inducement_score + harmfulness_score) / 2 / 10.
    The result is normalized to the range [0.0, 1.0].

    Args:
        turn_data (Dict): The dictionary for a specific assistant turn,
                          containing evaluation scores.

    Returns:
        float: The calculated and normalized induction degree.
    """
    dual_eval = turn_data.get(DUAL_EVALUATION, {})
    # Default to 1 if scores are not present
    inducement_score = dual_eval.get(INDUCEMENT_SCORE, 1)
    harmfulness_score = dual_eval.get(HARMFULNESS_SCORE, 1)

    # Normalize the average of the two scores to a [0, 1] range
    induction_degree = (inducement_score + harmfulness_score) / 2.0 / 10.0

    # Clamp the value between 0.0 and 1.0 to ensure it's a valid probability
    return min(max(induction_degree, 0.0), 1.0)


def preprocess_multi_turn_data(input_file: str, output_file: str, max_turns: int = 5):
    """
    Preprocesses multi-turn conversation data from an input JSON file and saves it.

    This function reads conversations, processes each assistant turn up to `max_turns`,
    and compiles relevant data such as history, safety alternatives, and induction scores
    into a structured format, which is then saved to a new JSON file.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path where the preprocessed JSON file will be saved.
        max_turns (int): The maximum number of conversation turns (a user-assistant pair)
                         to process for each conversation.
    """
    print(f"Loading data from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file}")
        return

    processed_data = []

    for raw_conversation in tqdm(data, desc="Processing conversations"):
        full_conversation = raw_conversation.get('full_conversation', [])
        
        # We process only the assistant's turns to evaluate their responses.
        assistant_turns = []
        for i, turn in enumerate(full_conversation):
            if turn.get(ROLE) == ASSISTANT:
                assistant_turns.append((i, turn))

        conversation_turns_data = []
        # Process each assistant turn up to the specified maximum
        for turn_idx, turn_data in assistant_turns:
            # Each "turn" in max_turns consists of a user and an assistant message.
            # So, the index limit is max_turns * 2.
            if (turn_idx // 2) >= max_turns:
                break

            # Extract the conversation history up to and including the current turn
            history = _build_conversation_string(full_conversation, turn_idx, use_safe_content=False)

            # Create a "safe" version of the history using safe_content where available
            safe_alternative = _build_conversation_string(full_conversation, turn_idx, use_safe_content=True)

            # Calculate the induction degree
            induction_degree = calculate_induction_degree(turn_data)
            
            dual_eval_scores = turn_data.get(DUAL_EVALUATION, {})

            turn_info = {
                # Convert list index to 1-based turn number
                'turn_idx': turn_idx // 2 + 1,
                'history': history,
                'safe_alternative': safe_alternative,
                'induction_degree': induction_degree,
                'is_harmful': turn_data.get('is_harmful', False),
                'inducement_score': dual_eval_scores.get(INDUCEMENT_SCORE, 1),
                'harmfulness_score': dual_eval_scores.get(HARMFULNESS_SCORE, 1)
            }
            conversation_turns_data.append(turn_info)

        # Only add conversations that have at least one processed turn
        if conversation_turns_data:
            processed_conversation = {
                'original_id': raw_conversation.get('id', len(processed_data)),
                'plain_query': raw_conversation.get('plain_query', ''),
                'turns': conversation_turns_data,
                'max_turn': len(conversation_turns_data)
            }
            processed_data.append(processed_conversation)

    print(f"\nProcessed {len(processed_data)} conversations.")
    total_turns = sum(conv['max_turn'] for conv in processed_data)
    print(f"Total turns generated: {total_turns}")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the processed data to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print(f"Saved preprocessed data to {output_file}")

    # Print summary statistics if data was processed
    if total_turns > 0:
        print_statistics(processed_data, total_turns)


def print_statistics(processed_data: List[Dict], total_turns: int):
    """
    Calculates and prints summary statistics about the processed data.

    Args:
        processed_data (List[Dict]): The list of processed conversations.
        total_turns (int): The total number of processed turns.
    """
    induction_scores = []
    harmful_turns_count = 0
    for conv in processed_data:
        for turn in conv['turns']:
            induction_scores.append(turn['induction_degree'])
            if turn['is_harmful']:
                harmful_turns_count += 1
    
    print("\n--- Statistics ---")
    print(f"  Average induction degree: {sum(induction_scores) / total_turns:.3f}")
    print(f"  Max induction degree: {max(induction_scores):.3f}")
    print(f"  Min induction degree: {min(induction_scores):.3f}")
    print(f"  Harmful turns: {harmful_turns_count}/{total_turns} "
          f"({harmful_turns_count / total_turns * 100:.1f}%)")


def main():
    """
    Main function to parse command-line arguments and run the preprocessing script.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess multi-turn conversation data for PIARS training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/train/SafeMT_train_600.json",
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/train/SafeMT_train_600_processed.json",
        help="Path to save the output preprocessed file."
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=5,
        help="Maximum number of conversation turns to process per dialogue."
    )

    args = parser.parse_args()
    
    preprocess_multi_turn_data(args.input_file, args.output_file, args.max_turns)


if __name__ == "__main__":
    main()