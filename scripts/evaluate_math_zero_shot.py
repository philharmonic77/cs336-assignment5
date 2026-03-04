import argparse
import json
import logging
import os
import sys
from statistics import mean
from typing import Callable, List

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


def format_prompts(examples: list[dict], prompt_template: str) -> list[str]:
    prompts = []
    for ex in examples:
        question = get_question(ex)
        prompts.append(prompt_template.format(question=question))
    return prompts


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    examples: list[dict],
    ground_truths: list,
    output_path: str,
) -> tuple[List[str], list[dict[str, float]]]:

    assert len(prompts) == len(examples) == len(ground_truths)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with xopen(output_path, "w") as fout:
        outputs: list[str] = []
        all_metrics: list[dict[str, float]] = []
        chunk_size = 256
        for start in tqdm(range(0, len(prompts), chunk_size)):
            end = start + chunk_size
            prompt_chunk = prompts[start:end]
            ex_chunk = examples[start:end]
            gt_chunk = ground_truths[start:end]

            raw_outputs = vllm_model.generate(prompt_chunk, eval_sampling_params)
            gen_chunk = [output.outputs[0].text for output in raw_outputs]
            outputs.extend(gen_chunk)

            for ex, gen, gt in zip(ex_chunk, gen_chunk, gt_chunk):
                if "</answer>" not in gen:
                    gen = gen + "</answer>"
                try:
                    metrics = reward_fn(gen, gt)
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
                            **ex,
                            "model_response": gen,
                            "metrics": metrics,
                        }
                    )
                    + "\n"
                )
    return outputs, all_metrics


def main(
    model_path: str,
    data_path: str,
    prompt_path: str,
    output_path: str,
):
    logger.info("Loading data from %s", data_path)
    examples = load_jsonl(data_path)
    logger.info("Loaded %d examples", len(examples))

    prompt_template = load_prompt_template(prompt_path)
    prompts = format_prompts(examples, prompt_template)
    logger.info("Formatted %d prompts", len(prompts))

    logger.info("Loading vLLM model from %s", model_path)
    model = LLM(
        model=model_path,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"]
    )

    ground_truths = [get_ground_truth(ex) for ex in examples]
    logger.info("Generating, scoring, and writing to %s", output_path)
    generations, all_metrics = evaluate_vllm(
        model,
        r1_zero_reward_fn,
        prompts,
        sampling_params,
        examples,
        ground_truths,
        output_path,
    )
    assert len(generations) == len(prompts)

    if all_metrics:
        metric_keys = sorted({k for m in all_metrics for k in m.keys()})
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            logger.info("%s: %.4f", key, mean(values))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/Qwen2.5-Math-1.5B",
        help="Local path to model",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/math/validation.jsonl",
        help="Path to MATH validation JSONL",
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
        default="outputs/qwen25_math_1p5b_r1_zero_math_validation.jsonl",
        help="Path to write output JSONL",
    )
    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))
    main(
        model_path=args.model_path,
        data_path=args.data_path,
        prompt_path=args.prompt_path,
        output_path=args.output_path,
    )
    logger.info("finished running %s", sys.argv[0])
