import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

# local imports
from modeling import RobertaModelForCoverage
from transformers import SchedulerType
import argparse


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.24.0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--only_eval", action="store_true", help="Whether only to load best checkpoints and run eval on the dev set."
    )
    parser.add_argument(
        "--checkpoint_to_evaluate", type=str, default=None, help="the checkpoint to load for evaluation only."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--max_input_value_length",
        type=int,
        default=64,
        help=(
            "The maximum length (bpe tokens) of function input states (variable values)."
        ),
    )
    parser.add_argument(
        "--dynamic_padding",
        action="store_true",
        help="If passed, pad all samples dynamically. Otherwise, padding to `max_length`",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--loss_logging_steps",
        type=int,
        default=50,
        help="Number of steps to log the training loss.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Where to store the cached datasets")
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="sample_accuracy",
        choices=["loss", "sample_accuracy", "f1", "accuracy", "precision", "recall"],
        help="The metric to use to load best model at the end.",
    )
    parser.add_argument(
        "--eval_last_checkpoint",
        action="store_true",
        help="Whether to evalute the last checkpoint not.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    args = parser.parse_args()

    return args

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def compute_metrics_exec(predictions, logits, references):
    true_predictions = []
    true_labels = []
    true_logits = []
    for prediction, logit, label in zip(predictions, logits, references):
        true_predictions += [p for idx, p in enumerate(prediction) if label[idx] != -100]
        true_logits += [l for idx, l in enumerate(logit) if label[idx] != -100]
        true_labels += [l for l in label if l != -100]
    sample_true_predictions = [
        [int(p) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, references)
    ]
    sample_true_logits = [
        [softmax(p.cpu().numpy()).tolist() for (p, l) in zip(logit, label) if l != -100]
        for logit, label in zip(logits, references)
    ]
    sample_true_labels = [
        [int(l) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, references)
    ]
    assert len(true_predictions) == len(true_labels) == len(true_logits)
    return true_predictions, true_labels, sample_true_predictions, sample_true_logits, sample_true_labels



def main():
    args = parse_args()

    # Sanity checks
    if args.train_file is None or args.validation_file is None:
        raise ValueError("Need train_file and validation_file for exec pretraining.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Loading the dataset from local csv or json file.
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=args.cache_dir)

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
        

    model = RobertaModelForCoverage.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )

    padding = "max_length" if not args.dynamic_padding else False

    def preprocess_function(examples):
        # global num_input_tokens
        src_key = "src_method"
        src_label_key = "src_cov_label"
        input_key = "entry_variables"
        max_seq_length = args.max_seq_length - args.max_input_value_length
        total = len(examples[src_key])

        # count the number of tokens in the input
        # input_tokens = tokenizer(examples[input_key])
        # num_input_tokens += [len(tokens) for tokens in input_tokens["input_ids"]]

        # Tokenize and truncate entry variables
        tokenized_entries = tokenizer(
            examples[input_key], 
            padding=padding, 
            truncation=True, 
            max_length=args.max_input_value_length)

        
        # build raw samples w/ special tokens
        src_samples = []
        for idx in range(total):
            assert len(examples[src_key][idx]) == len(examples[src_label_key][idx])
            code = f"{tokenizer.mask_token}".join(examples[src_key][idx]) + f"{tokenizer.mask_token}" # "stmt_0 <mask> stmt_1 <mask> stmt_2 <mask>"
            src_samples.append(code)
        # tokenizer will add <s> and </s> to the beginning and end of the sample
        tokenized_inputs = tokenizer(
            src_samples,
            truncation=True,
            padding=padding,
            max_length=max_seq_length
        ) # <s> stmt_0 <mask> stmt_1 <mask> stmt_2 <mask> </s>
        assert len(tokenized_entries["input_ids"]) == len(tokenized_inputs["input_ids"]) == len(examples[src_label_key])

        features = {}
        for k in tokenized_inputs:
            features[k] = [tokenized_entries[k][i] + tokenized_inputs[k][i] for i in range(len(tokenized_inputs[k]))]
            # <s> entry_variables </s> <s> stmt_0 <mask> stmt_1 <mask> stmt_2 <mask> </s>
                

        labels = []
        for i, label in enumerate(examples[src_label_key]):
            # At the end of each line, there is a <mask>, and we use this to recognize the previous line
            word_ids = features["input_ids"][i]
            label_ids = []
            label_idx = 0
            # within a line of source code, only the <mask> token is labeled
            for word_idx in word_ids:
                if word_idx == tokenizer.mask_token_id:
                    label_ids.append(label[label_idx])
                    label_idx += 1
                else:
                    label_ids.append(-100) # the label is not a mask token, so we don't care about it
            assert len(label_ids) == len(features["input_ids"][i])
            labels.append(label_ids)
        
        features["labels"] = labels

        return features


    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    # We use the data collator for padding
    data_collator = default_data_collator

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("exec_pretrain", experiment_config)

    # Get the metric function
    acc_metric = evaluate.load("accuracy")
    prec_metric = evaluate.load("precision")
    rec_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    if not args.only_eval:
        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0
        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
                accelerator.load_state(args.resume_from_checkpoint)
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
            else:
                resume_step = int(training_difference.replace("step_", ""))
                starting_epoch = resume_step // len(train_dataloader)
                resume_step -= starting_epoch * len(train_dataloader)

        best_metric = float("-inf")
        best_checkpoint = None
        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()
            if args.with_tracking:
                total_loss = 0
            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        completed_steps += 1
                        continue
                outputs = model(**batch)
                loss = outputs.loss
                # log the training loss
                if args.loss_logging_steps > 0 and completed_steps > 0 and completed_steps % args.loss_logging_steps == 0:
                    # if args.with_tracking:
                    #     accelerator.log_metric("train_loss", loss.detach().float(), epoch=epoch, step=completed_steps)
                    logger.info(f"Epoch {epoch} Step {completed_steps}: loss {loss.detach().float()}")
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps }"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)

                if completed_steps >= args.max_train_steps:
                    break

            model.eval()
            samples_seen = 0
            stmt_predictions = []
            stmt_labels = []
            sample_predictions = []
            sample_logits = []
            sample_labels = []
            losses = []
            logger.info("***** Running evaluation *****")
            for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                logits = outputs.logits
                loss = outputs.loss
                losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
                inputs, predictions, references, logits = accelerator.gather((batch["input_ids"], predictions, batch["labels"], logits))
                # If we are in a multiprocess environment, the last batch has duplicates
                if accelerator.num_processes > 1:
                    if step == len(eval_dataloader) - 1:
                        inputs = inputs[: len(eval_dataloader.dataset) - samples_seen]
                        predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                        references = references[: len(eval_dataloader.dataset) - samples_seen]
                        logits = logits[: len(eval_dataloader.dataset) - samples_seen]
                    else:
                        samples_seen += references.shape[0]

                true_predictions, true_labels, sample_true_predictions, sample_true_logits, sample_true_labels = compute_metrics_exec(predictions, logits, references)
                stmt_predictions += true_predictions
                stmt_labels += true_labels
                sample_predictions += sample_true_predictions
                sample_logits += sample_true_logits
                sample_labels += sample_true_labels
                
            losses = torch.cat(losses)
            eval_loss = torch.mean(losses)

            
            # Compute sample accuracy manually
            sample_accuracy = sum([1 if p == l else 0 for p, l in zip(sample_predictions, sample_labels)]) / len(sample_predictions)
            eval_metric = {"sample_accuracy": sample_accuracy}
            eval_metric.update(acc_metric.compute(predictions=stmt_predictions, references=stmt_labels))
            eval_metric.update(prec_metric.compute(predictions=stmt_predictions, references=stmt_labels))
            eval_metric.update(rec_metric.compute(predictions=stmt_predictions, references=stmt_labels))
            eval_metric.update(f1_metric.compute(predictions=stmt_predictions, references=stmt_labels))
            logger.info(f"epoch {epoch} --- eval_loss: {eval_loss}, eval_sample_acc: {eval_metric['sample_accuracy']}, eval_stmt_acc: {eval_metric['accuracy']}, eval_stmt_prec: {eval_metric['precision']}, eval_stmt_rec: {eval_metric['recall']}, eval_stmt_f1: {eval_metric['f1']}")

            if args.with_tracking:
                accelerator.log(
                    {
                        "eval_loss": eval_loss,
                        "accuracy": eval_metric['accuracy'],
                        "precision": eval_metric['precision'],
                        "recall": eval_metric['recall'],
                        "f1": eval_metric['f1'],
                        "train_loss": total_loss.item() / len(train_dataloader),
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )

            if args.push_to_hub and epoch < args.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                    repo.push_to_hub(
                        commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                    )

            # TODO: extend the below to steps as well
            if args.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                # check whether this is the best checkpoint
                if args.metric_for_best_model == 'sample_accuracy':
                    tracking_metric = eval_metric['sample_accuracy']
                elif args.metric_for_best_model == 'f1':
                    tracking_metric = eval_metric['f1']
                elif args.metric_for_best_model == 'accuracy':
                    tracking_metric = eval_metric['accuracy']
                elif args.metric_for_best_model == 'precision':
                    tracking_metric = eval_metric['precision']
                elif args.metric_for_best_model == 'recall':
                    tracking_metric = eval_metric['recall']
                elif args.metric_for_best_model == 'loss':
                    tracking_metric = -eval_loss
                else:
                    raise ValueError(f"Metric {args.metric_for_best_model} not supported")
                if tracking_metric > best_metric:
                    best_metric = tracking_metric
                    output_dir += f"_best_{args.metric_for_best_model}"

                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)
                if "best" in output_dir:
                    best_checkpoint = output_dir
    else:
        # only evaluate
        assert args.checkpoint_to_evaluate is not None
        best_checkpoint = args.checkpoint_to_evaluate
    if args.eval_last_checkpoint:
        best_checkpoint = output_dir
    logger.info(f"Loading best/last checkpoint from {best_checkpoint}")
    accelerator.load_state(best_checkpoint)

    model.eval()
    samples_seen = 0
    
    stmt_predictions = []
    stmt_labels = []
    sample_predictions = []
    sample_logits = []
    sample_labels = []
    losses = []
    logger.info("***** Running evaluation *****")
    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        logits = outputs.logits
        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
        inputs, predictions, references, logits = accelerator.gather((batch["input_ids"], predictions, batch["labels"], logits))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader) - 1:
                inputs = inputs[: len(eval_dataloader.dataset) - samples_seen]
                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                references = references[: len(eval_dataloader.dataset) - samples_seen]
                logits = logits[: len(eval_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]

        true_predictions, true_labels, sample_true_predictions, sample_true_logits, sample_true_labels = compute_metrics_exec(predictions, logits, references)
        stmt_predictions += true_predictions
        stmt_labels += true_labels
        sample_predictions += sample_true_predictions
        sample_logits += sample_true_logits
        sample_labels += sample_true_labels
    
    losses = torch.cat(losses)
    eval_loss = torch.mean(losses)

    
    # Compute sample accuracy manually
    sample_accuracy = sum([1 if p == l else 0 for p, l in zip(sample_predictions, sample_labels)]) / len(sample_predictions)
    eval_metric = {"sample_accuracy": sample_accuracy}
    eval_metric.update(acc_metric.compute(predictions=stmt_predictions, references=stmt_labels))
    eval_metric.update(prec_metric.compute(predictions=stmt_predictions, references=stmt_labels))
    eval_metric.update(rec_metric.compute(predictions=stmt_predictions, references=stmt_labels))
    eval_metric.update(f1_metric.compute(predictions=stmt_predictions, references=stmt_labels))

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model) # this is the best model
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

    if args.output_dir is not None:
        all_results = {f"eval_{k}": v for k, v in eval_metric.items()}

        all_results["eval_loss"] = eval_loss.tolist()
        with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
            json.dump(all_results, f)
        with open(args.validation_file, 'r') as f:
            dev_data = f.readlines()
            dev_data = [json.loads(line) for line in dev_data]

        assert len(dev_data) == len(sample_predictions) == len(sample_labels), f"{len(dev_data)}, {len(sample_predictions)}, {len(sample_predictions)}"
        with open(os.path.join(args.output_dir, "eval_details.jsonl"), "w") as f:
            for d, pred, label, logit in zip(dev_data, sample_predictions, sample_labels, sample_logits):
                l = {}
                l['prediction'] = pred
                l['label'] = label
                l['logit'] = logit
                for k, v in d.items():
                    l[k] = v
                f.write(json.dumps(l) + '\n')


if __name__ == "__main__":
    # num_input_tokens = []
    main()