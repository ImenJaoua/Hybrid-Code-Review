# python run_batch_inference.py --test_data hf-datasets/test/ --save_steps 20 --batch_size 2 --continue_from_checkpoint
import os
gpu = 3
os.environ["CUDA_VISIBLE_DEVICES"]=f"{gpu}"
import argparse
from transformers import pipeline, AutoTokenizer, set_seed, AutoModelForCausalLM, BitsAndBytesConfig
import datasets
import torch
import pickle
from tqdm import tqdm
import gc

torch.cuda.empty_cache()
gc.collect()
set_seed(0)

def formatting_prompts(examples, tokenizer):
    system_template = "You are an expert in coding and code peer-reviewing."
    instruction_template = """ ### Code review comment generation
    You will be assisted by outputs from a static code analyzer to generate a review comment given code difference.
    Given the code difference below, generate a review comment that highlights potential issues or suggests improvements.

    ### Static code analyzer output
    {static_analyzer_output}

    ### Code difference
    {code_diff}
    """

    prompts = []
    for i in range(len(examples['union_diff'])):
        chat = [
        {"role": "system", "content": system_template},
        {"role": "user", "content": instruction_template.format(
            code_diff=examples['union_diff'][i][0: min(2048, len(examples['union_diff'][i]))], 
            static_analyzer_output=examples['review2'][i])},
        ]
        text = tokenizer.apply_chat_template(chat, tokenize=False)
        prompts.append(text)
    return prompts


def run_inference(model_name, dataset, batch_size=8,  save_steps=100, continue_from_checkpoint=False, save_file=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "torch_dtype": torch.float16
    }
    q_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                **model_kwargs,
                                                attn_implementation="flash_attention_2",
                                                device_map="auto",
                                                quantization_config=q_config)
    pipe = pipeline(
                "text-generation", 
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map="auto",
                pad_token_id=tokenizer.eos_token_id,
                )
    print('*** Running inference ***')
    steps_done = 0
    steps = 0
    updated_dataset = None
    results = []
    formatted_prompts = []

    if continue_from_checkpoint:
        with open(save_file, 'rb') as f:
            steps_done, results, formatted_prompts = pickle.load(f)
            assert len(results) == len(formatted_prompts) == steps_done
        print(f'*** Continue from step {steps_done} ***')

    def process_batch(batch):
        nonlocal steps
        nonlocal steps_done
        nonlocal results
        nonlocal updated_dataset
        nonlocal formatted_prompts
        prompts = batch["formatted_prompt"]
        if continue_from_checkpoint:
            if steps < steps_done:
                steps += len(prompts)
                return {"generated_response": results[steps-len(prompts):steps]}
        batch_results = pipe(prompts, do_sample=True,
                             temperature=0.2,
                             top_p=0.9,
                             num_return_sequences=3,
                            #  max_length=2048,
                            return_full_text=False,
                             max_new_tokens=512,
                             truncation=True)
        responses = [[r["generated_text"] for r in res] for res in batch_results]
        results.extend(responses)
        formatted_prompts.extend(prompts)
        steps += len(prompts)
        if steps % save_steps == 0 or steps == len(dataset):
            with open(save_file, 'wb') as f:
                pickle.dump((steps, results, formatted_prompts), f)
            print(f'*** Saved checkpoint at step {steps} ***')
        return {"generated_response": responses}

    updated_dataset = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing Batches"):
        batch = dataset[i:i + batch_size]
        processed_batch = process_batch(batch)
        updated_dataset.extend(processed_batch)
    return results

if __name__ == "__main__":
    
    data_path = "data/inter_HFdataset-v2"
    batch_size = 8
    save_steps = 20
    continue_from_checkpoint = True
    model_name = 'models/final_model/'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = datasets.load_from_disk(data_path)    
    prompts = formatting_prompts(dataset, tokenizer)
    dataset = dataset.add_column("formatted_prompt", prompts)

    results = run_inference(model_name, dataset, batch_size=batch_size, save_steps=save_steps, continue_from_checkpoint=continue_from_checkpoint, save_file=f'data/results_checkpoint-{gpu}.pkl')
    dataset = dataset.add_column("generated_response", results)

    dataset.save_to_disk(f'data/dataset_results_{gpu}')