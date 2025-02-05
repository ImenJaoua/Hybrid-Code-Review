
from datasets import load_from_disk

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from typing import List, Union, Dict, Any
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, EarlyStoppingCallback
import torch
import glob
import re
import argparse
from training.config import parse_args
from peft import LoraConfig
# Parse arguments
parser = argparse.ArgumentParser()
args = parse_args(parser)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set environment variables for GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"

def load_dataset():
    # Load the dataset from disk
    loaded_dataset = load_from_disk("processed_dataset_new_prompt")
    # Print sizes of training and validation sets
    train_size = len(loaded_dataset["train"])
    valid_size = len(loaded_dataset["validation"])
    print(f"Filtered training set size: {train_size} examples")
    print(f"Filtered validation set size: {valid_size} examples")
    
    return loaded_dataset

class CustomDataCollatorForCompletionOnlyLM(DataCollatorForCompletionOnlyLM):
    def init(self, args, **kwargs):
        super().init(args, **kwargs)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        # Ensure the last token is the EOS token
        batch["labels"][:, -1] = self.tokenizer.eos_token_id
        return batch

if __name__ == "__main__":
    dataset = load_dataset()
    print(dataset)
    # Set a random seed for reproducibility
    torch.manual_seed(27)

    print(f"Continue from checkpoint: {args['continue_from_checkpoint']}")
    print(f"Checkpoint folder: {args['checkpoint_folder']}")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Model configuration
    model_name = "codellama/CodeLlama-7b-Instruct-hf"
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "torch_dtype": torch.float16
    }
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Early stopping callback
    callbacks = EarlyStoppingCallback(early_stopping_patience=5)
    
    # Response template for the DataCollator
    response_template = "[/INST]"
    collator = CustomDataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    model.gradient_checkpointing_enable()

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./output5',
        gradient_accumulation_steps=8,
        num_train_epochs=args['num_epochs'],
        save_steps=2000,
        logging_steps=200,
        per_device_train_batch_size=1,
        evaluation_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="none"
    )
    
    # LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
        max_seq_length=2048,
        callbacks=[callbacks],
        data_collator=collator
    )

    # Clear GPU memory cache
    torch.cuda.empty_cache()
    # Sorting checkpoints function
    def sort_checkpoints(checkpoint_paths):
        def extract_checkpoint_number(checkpoint_path):
            match = re.search(r'checkpoint-(\d+)', checkpoint_path)
            return int(match.group(1)) if match else float('inf')
        return sorted(checkpoint_paths, key=extract_checkpoint_number)

    # Continue training from the latest checkpoint if provided
    if args['continue_from_checkpoint']:
        checkpoint_path = os.path.join('./output5', "checkpoint-*")
        checkpoints = sort_checkpoints(glob.glob(checkpoint_path))
        
        print(f"Found checkpoints: {checkpoints}")
        
        if checkpoints:
            checkpoint = checkpoints[-1]  # Get the most recent checkpoint
            print(f"Resuming from checkpoint: {checkpoint}")
            trainer.train(resume_from_checkpoint=checkpoint)
        else:
            print("No checkpoint found. Starting training from scratch.")
            trainer.train()
    else:
        print("Starting training from scratch.")
        trainer.train()

