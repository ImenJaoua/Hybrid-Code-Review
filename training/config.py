import argparse


def parse_args(parser):
    parser.add_argument(
        "--train_data",
        "-f",
        type=str,
        default="",
        help="Path to the file that contains the training dataset.",
    )
    parser.add_argument(
        "--valid_data",
        type=str,
        default="",
        help="Path to the file that contains the validation dataset.",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="",
        help="Path to the file that contains the test dataset.",
    )
    parser.add_argument("--model_name",
                        "-m",
                        type=str,
                        default="codellama/CodeLlama-7b-Instruct-hf")
   
    parser.add_argument("--continue_from_checkpoint", action="store_true")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument(
        "--checkpoint_folder",
        default="",
        type=str,
        help=
        "Path to the checkpoint folder. Useful only if continue_from_checkpoint==True.",
    )
    

    parser.add_argument(
        "--reward_model",
        type=str,
        default="semantic",
        help="Type of the reward model.",
    )
    parser.add_argument("--num_epochs", "-e", type=int, default=5)
    parser.add_argument(
        "--output_dir",
        "-o",
        default="output/",
        type=str,
        help="Path to the output directory where the checkpoints are saved.",
    )
    parser.add_argument("--max_sequence_length", type=int, default=1500)
    parser.add_argument("--learning_rate", "-l", type=float, default=3e-4)
    parser.add_argument("--gradient_accumulation_steps",
                        "-g",
                        type=int,
                        default=4)
    parser.add_argument("--seed", "-s", type=int, default=12345)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_args(parser)
    print(args)