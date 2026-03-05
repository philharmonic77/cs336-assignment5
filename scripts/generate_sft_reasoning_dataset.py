import argparse
import json
import logging
import os
import random
import sys
from statistics import mean

from tqdm import tqdm
from vllm import LLM, SamplingParams
from xopen import xopen

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

logger = logging.getLogger(__name__)


def load_jsonl(path):
    examples = []
    with xopen(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def load_prompt_template(path: str) -> str:
    with xopen(path) as f:
        return f.read()


def get_question(example: dict) -> str:
    if "problem" in example and example["problem"] is not None:
        return str(example["problem"])
    raise KeyError("Could not find 'problem' field in example")


def get_ground_truth(example: dict):
    if "answer" in example and example["answer"] is not None:
        return example["answer"]
    raise KeyError("Could not find 'answer' field in example")


def format_prompt(question: str, prompt_template: str) -> str:
    return prompt_template.format(question=question)


def main(
    model_path: str,
    data_path: str,
    prompt_path: str,
    output_path: str,
    num_samples: int,
    seed: int,
):
    logger.info("Loading data from %s", data_path)
    examples = load_jsonl(data_path)
    logger.info("Loaded %d examples", len(examples))

    prompt_template = load_prompt_template(prompt_path)
    logger.info("Loaded prompt template from %s", prompt_path)

    rng = random.Random(seed)
    sampled_examples = [rng.choice(examples) for _ in range(num_samples)]
    logger.info("Sampled %d problems with replacement", len(sampled_examples))

    logger.info("Loading vLLM model from %s", model_path)
    model = LLM(
        model=model_path,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    all_metrics = []
    chunk_size = 256
    with xopen(output_path, "w") as fout:
        for start in tqdm(range(0, len(sampled_examples), chunk_size)):
            end = start + chunk_size
            ex_chunk = sampled_examples[start:end]
            prompts = [
                format_prompt(get_question(ex), prompt_template) for ex in ex_chunk
            ]
            ground_truths = [get_ground_truth(ex) for ex in ex_chunk]

            raw_outputs = model.generate(prompts, sampling_params)
            gen_chunk = [output.outputs[0].text for output in raw_outputs]

            for ex, prompt, gen, gt in zip(ex_chunk, prompts, gen_chunk, ground_truths):
                try:
                    metrics = r1_zero_reward_fn(gen, gt)
                except Exception as e:
                    logger.exception(
                        "Failed to score example; defaulting to zeros: %s", e
                    )
                    metrics = {
                        "format_reward": 0.0,
                        "answer_reward": 0.0,
                        "reward": 0.0,
                        "error": 1.0,
                    }
                all_metrics.append(metrics)
                fout.write(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "response": gen,
                            "metrics": metrics,
                        }
                    )
                    + "\n"
                )

    if all_metrics:
        metric_keys = sorted({k for m in all_metrics for k in m.keys()})
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            logger.info("%s: %.4f", key, mean(values))

        c_correct = sum(
            1
            for m in all_metrics
            if m.get("format_reward") == 1.0 and m.get("answer_reward") == 1.0
        )
        c_format_only = sum(
            1
            for m in all_metrics
            if m.get("format_reward") == 1.0 and m.get("answer_reward") == 0.0
        )
        c_zero = sum(
            1
            for m in all_metrics
            if m.get("format_reward") == 0.0 and m.get("answer_reward") == 0.0
        )
        logger.info(
            "count(correct_format_and_answer)= %d", c_correct
        )
        logger.info(
            "count(format_only)= %d", c_format_only
        )
        logger.info(
            "count(format0_answer0)= %d", c_zero
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="HF model name or local path",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/math/train.jsonl",
        help="Path to MATH train JSONL",
    )
    parser.add_argument(
        "--prompt-path",
        type=str,
        default="cs336_alignment/prompts/r1_zero.prompt",
        help="Prompt template path",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/math/sft.jsonl",
        help="Path to write output JSONL",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=15000,
        help="Number of samples to generate (with replacement)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling",
    )
    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))
    main(
        model_path=args.model_path,
        data_path=args.data_path,
        prompt_path=args.prompt_path,
        output_path=args.output_path,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    logger.info("finished running %s", sys.argv[0])
