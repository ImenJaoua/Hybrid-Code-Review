from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ExLlamaArguments:
    model_dir: Optional[str] = field(
        default="/data/rech/jaouaime/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4",
        metadata={"help": "Path to the local model directory."}
    )

    dataset_path: Optional[str] = field(
        default="final_manual_judge_new_prompt_36000.csv",
        metadata={"help": "Path to the huggingface dataset."}
    )

    output_path: Optional[str] = field(
        default="results23.jsonl",
        metadata={"help": "Path to the saved inference results after each save_steps steps."}
    )

    checkpoint_path: Optional[str] = field(
        default="",
        metadata={"help": "Path to the final saved inference results."}
    )

    save_steps: Optional[int] = field(
        default=2000,
        metadata={"help": "The inference results will be saved at these steps."}
    )

    max_seq_len: Optional[int] = field(
        default=32768,
        metadata={"help": "Maximum sequence length for input tokens."}
    )

    max_batch_size: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum batch size to be used during inference."}
    )

    max_q_size: Optional[int] = field(
        default=4,
        metadata={"help": "Maximum number of sequences to queue for processing at one time."}
    )

    gen_settings: Optional[Tuple[float, float]] = field(
        default=(1.0, 0),
        metadata={"help": "Pair of floats representing the token repetition penalty and sampling temperature settings for generation."}
    )

    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to generate."}
    )


if __name__ == "__main__":
    from transformers import HfArgumentParser
    parser = HfArgumentParser(ExLlamaArguments)
    model_args = parser.parse_args_into_dataclasses()[0]
    print(model_args.max_seq_len)


    