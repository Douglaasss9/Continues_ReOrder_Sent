"""
fine-tuning the encoder-decoder BART/T5 model.
"""
import os
import torch
import pickle
import logging
import argparse

from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss

from source.common import init_model, load_data, load_data2
from source.train import evaluate, train, set_seed, test

from source.common import RankModel


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


class EncoderDecoderTextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, block_size=512):
        print(file_path)
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        filename = f"{os.path.basename(args.model_type)}_cached_{block_size}_{filename}{'_' + args.task if args.task else ''}"
        cached_features_file = os.path.join(directory, filename)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)

                
        else:
            logger.info("Converting to token IDs")
            examples, target = load_data(file_path, args.task)
            self.recsent = load_data2(args.test_path, args.task)
            logger.info(examples[:5])


            # Add prefix to the output so we can predict the first real token in the decoder
            inputs = [
                [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ex[0])), tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ex[1]))]
                for ex in examples
            ]

            inputval = []
            # for i in range(len(examples)):
            #     print(examples[i][0], examples[i][1])
            #     print(len(inputs[i][0]), len(inputs[i][1]))
            for ex in inputs:
                tx1 = 0
                tx2 = 0
                for j in ex[0]:
                    if j in ex[1]:
                        tx1 += 1
                for j in ex[1]:
                    if j in ex[0]:
                        tx2 += 1
                inputval.append([tx1 * tx2])


            # for i in range(len(inputs)):
            #     print(inputs[i][-1])

            # Pad
            max_input_length = min(
                args.max_input_length, max(len(ex[0]) + len(ex[1]) + 3 for ex in inputs))


            

            input_lengths = [min(len(ex[0]) + len(ex[1]) + 3, max_input_length) for ex in inputs]


            self.max_input_length = max_input_length
            


            inputs = [tokenizer.encode(
                text = ex[0], text_pair = ex[1], add_special_tokens=False, max_length=max_input_length, pad_to_max_length=True)
                for ex in examples]


            self.examples = {
                "inputs": inputs,
                "tokenval": inputval,
                "outputs": target,
                "input_lengths": input_lengths
            }

        logger.info(f"Saving features into cached file {cached_features_file}")
        with open(cached_features_file, "wb") as handle:
            pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples["input_lengths"])

    def __getitem__(self, item):
        inputs = torch.tensor(self.examples["inputs"][item])
        outputs = torch.tensor(self.examples["outputs"][item])
        inputval = torch.tensor(self.examples["tokenval"][item])



        max_length = inputs.shape[0]

        input_lengths = self.examples["input_lengths"][item]
        input_mask = torch.tensor([1] * input_lengths + [0] * (max_length - input_lengths))


        
        return {
            "inputs": inputs,
            "input_mask": input_mask,
            "tokenval": inputval,
            "outputs": outputs
        }


def get_loss(args, batch, model):
    """
    Compute this batch loss
    """
    input_ids = batch["inputs"].to(args.device)
    input_mask = batch["input_mask"].to(args.device)
    targets = batch["outputs"].to(args.device)
    tokenval = batch["tokenval"].to(args.device)


    # We don't send labels to model.forward because we want to compute per token loss
    lm_logits = model(
        input_ids, input_mask, tokenval
    )  # use_cache=false is added for HF > 3.0

    # Compute loss for each instance and each token
    loss_fct = CrossEntropyLoss(reduction="none")
    loss = loss_fct(lm_logits, targets
    )

    # Only consider non padded tokens
    return loss


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--out_dir",
        default=None,
        type=str,
        required=True,
        help="Out directory for checkpoints.",
    )

    # Other parameters
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--device", default="0", type=str, help="GPU number or 'cpu'."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )

    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--eval_batch_size", default=64, type=int, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--eval_data_file",
        type=str,
        required=True,
        help="The input CSV validation file."
    )
    parser.add_argument(
        "--test_path",
        type=str,
        required=False,
        default = "data_100.jsonl",
        help="The input CSV validation file."
    )
    parser.add_argument(
        "--eval_during_train",
        action="store_true",
        help="Evaluate at each train logging step.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Steps before backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-6,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=-1,
        help="Log every X updates steps (default after each epoch).",
    )
    parser.add_argument(
        "--max_input_length",
        default=140,
        type=int,
        help="Maximum input event length in words.",
    )
    parser.add_argument(
        "--max_output_length",
        default=120,
        type=int,
        help="Maximum output event length in words.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: total number of training steps to perform.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bart-large",
        type=str,
        help="LM checkpoint for initialization.",
    )
    parser.add_argument(
        "--model_type",
        default="",
        type=str,
        help="which family of LM, e.g. gpt, gpt-xl, ....",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=2.0,
        type=float,
        help="Number of training epochs to perform.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached data."
    )
    parser.add_argument(
        "--overwrite_out_dir",
        action="store_true",
        help="Overwrite the output directory.",
    )
    parser.add_argument(
        "--continue_training",
        action="store_true",
        help="Continue training from the last checkpoint.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=-1,
        help="Save checkpoint every X updates steps (default after each epoch).",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization."
    )
    parser.add_argument(
        "--train_batch_size", default=64, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=False,
        help="The input CSV train file."
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--task",
        type=str,
        help="what is the task?"
    )
    args = parser.parse_args()

    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(args.out_dir)
        and len(os.listdir(args.out_dir)) > 1
        and args.do_train
        and not args.overwrite_out_dir
        and not args.continue_training
    ):
        raise ValueError(
            f"Output directory {args.out_dir} already exists and is not empty. "
            f"Use --overwrite_out_dir or --continue_training."
        )

    # Setup device
    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    args.device = device
    # Set seed
    set_seed(args)

    # Load the models
    if args.continue_training:
        args.model_name_or_path = args.out_dir
    # Delete the current results file
    else:
        eval_results_file = os.path.join(args.out_dir, "eval_results.txt")
        if os.path.exists(eval_results_file):
            os.remove(eval_results_file)

    tokenizer, model = init_model(
        args.model_name_or_path, device=args.device, args = args
    )

    args.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Pad token ID: {args.pad_token_id}")
    args.block_size = tokenizer.max_len_single_sentence
    logger.info(f"Training/evaluation parameters {args}")

    eval_dataset = None
    if args.do_eval or args.eval_during_train:
        eval_dataset = EncoderDecoderTextDataset(
            tokenizer, args, file_path=args.eval_data_file, block_size=args.block_size)

    # Add special tokens (if loading a model before fine-tuning)


    args.pad_token_id = tokenizer.pad_token_id

    # resize_token_embeddings for Bart doesn't work if the model is already on the device
    args.device = device
    model.to(args.device)

    # Training
    if args.do_train:
        train_dataset = EncoderDecoderTextDataset(
            tokenizer,
            args,
            file_path=args.train_file,
            block_size=args.block_size,
        )
        global_step, tr_loss = train(
            args,
            train_dataset,
            model,
            tokenizer,
            loss_fnc=get_loss,
            eval_dataset=eval_dataset,
        )
        logger.info(f" global_step = {global_step}, average loss = {tr_loss}")

        # Create output directory if needed
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

        logger.info(f"Saving model checkpoint to {args.out_dir}")

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # model_to_save = model.module if hasattr(model, "module") else model
        # model_to_save.save_pretrained(args.out_dir)\
        torch.save(model, args.out_dir + "/savemodel.pth")
        model.model.save_pretrained(args.out_dir)
        tokenizer.save_pretrained(args.out_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.out_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        tokenizer, model = init_model(
            args.out_dir, device=args.device, args=args
        )
        args.block_size = tokenizer.max_len_single_sentence
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint = args.out_dir
        logger.info(f"Evaluate the following checkpoint: {checkpoint}")
        prefix = (
            checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        )
        model = torch.load(args.out_dir + "/savemodel.pth")

        model.to(args.device)
        result = evaluate(eval_dataset, args, model, prefix=prefix, loss_fnc=get_loss)
        results.update(result)
    
    model = torch.load(args.out_dir + "/savemodel.pth")
    test_dataset = eval_dataset = EncoderDecoderTextDataset(tokenizer, args, file_path=args.test_path, block_size=args.block_size)
    # test_dataset.max_input_length = train_dataset.max_input_length
    model.to(args.device)
    test(test_dataset, args, model, tokenizer)

    return results


if __name__ == "__main__":
    main()
