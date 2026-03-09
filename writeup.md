# SETUP
For those who are **following along at home**:
- You can get the same **MATH dataset** by running this [scripts](scripts/download_math_data.py).

- You can download Qwen2.5-Math-1.5B by running this [scripts](scripts/download_Qwen_model.py).

- You can use a teacher model(deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) to [generate](scripts/generate_sft_reasoning_dataset.py) reasoning traces for MATH - training set problems and build an SFT dataset as a substitute for `/data/a5-alignment/MATH/sft.jsonl`, which will be used in section 4.

```
Dataset Quality Summary (data/math/sft.jsonl):
	•	Size: 15,000 reasoning traces.
	•	Accuracy (relaxed evaluation): 7695 / 15000 ≈ 51.3%.
	•	Reasoning length: average ≈772 tokens, maximum 1024 tokens.
	•	finish_reason distribution: stop -> 8794 (≈58.6%)
, length -> 6206 (≈41.4%)
	•	Format: 100% do not strictly follow the <think> ... </think> <answer> ... </answer> structure. 
```

By running this format correction [script](scipts/sft_format_correction.py), we force to correct their formats. The new data file is stored as `/MATH/sft_format_correction.jsonl`.

# 3 Measuring Zero-Shot MATH Performance
## 3.2 Zero-shot MATH Baseline
### Problem (math_baseline): 4 points
(a) Here are the [script](scripts/evaluate_math_zero_shot.py) and the evaluation [output](outputs/math_baseline/).

---

(b)
#### temperature=1, max_tokens=1024 (*assignment baseline)

|              | **Answer = 1** | **Answer = 0** | **Total** |
|--------------|---------------|---------------|-----------|
| **Format = 1** | 135 (2.70%) | 759 (15.18%) | 894 (17.88%) |
| **Format = 0** | 0 (0.00%) | 4106 (82.12%) | 4106 (82.12%) |
| **Total**      | 135 (2.70%) | 4865 (97.30%) | 5000 (100%) |

There are some parse errors such as:
`Ignoring parse error (tolerant parsing mode): Unexpected mismatching closing brace: '}' @(1,1)`

#### temperature=1, max_tokens=2048
|              | **Answer = 1** | **Answer = 0** | **Total** |
|--------------|---------------|---------------|-----------|
| **Format = 1** | 130 (2.60%) | 716 (14.32%) | 846 (16.92%) |
| **Format = 0** | 0 (0.00%) | 4154 (83.08%) | 4154 (83.08%) |
| **Total**      | 130 (2.60%) | 4870 (97.40%) | 5000 (100%) |

#### temperature=0, max_tokens=2048
|              | **Answer = 1** | **Answer = 0** | **Total** |
|--------------|---------------|---------------|-----------|
| **Format = 1** | 857 (17.14%) | 1237 (24.74%) | 2094 (41.88%) |
| **Format = 0** | 0 (0.00%) | 2906 (58.12%) | 2906 (58.12%) |
| **Total**      | 857 (17.14%) | 4143 (82.86%) | 5000 (100%) |

#### Generation statistics

| Setting | Avg Tokens | Stop | Length | Length % |
|-------|-----------|------|-------|---------|
| temp=1, max=1024 | 350.63 | 4555 | 445 | 8.90% |
| temp=1, max=2048 | 410.19 | 4861 | 139 | 2.78% |
| temp=0, max=2048 | 376.38 | 4782 | 218 | 4.36% |

其中：
- Stop = generation terminated by `</answer>`
- Length = generation truncated by `max_tokens`

#### Case: format reward = 0
I find that the [parser](cs336_alignment/drgrpo_grader.py) is very strict:
``` python
def r1_zero_reward_fn(response, ground_truth, fast=True):
    # We are strict about format to evaluate our models.
    if "</think> <answer>" in response and "</answer>" in response:
    ...
```
In this case, there is a `,`between `</think>` and `<answer>`.
``` json
{
  "problem": "If $2^8=4^x$, what is the value of $x$?",
  "solution": "Rewrite $4$ as $2^2$ to find $4^x=2^{2x}$.  Since $2^8=2^{2x}$, we have $2x=8$ which implies $x=\\boxed{4}$.",
  "answer": "4",
  "subject": "Algebra",
  "level": "Level 1",
  "model_response": "2^8=4^2=4^x</think>, <answer>$x=2$</answer>",
  "num_generated_tokens": 24,
  "finish_reason": "stop",
  "metrics": {
    "format_reward": 0,
    "answer_reward": 0,
    "reward": 0
  }
}
```
In this case, the model generates a closing </think> tag but continues generating reasoning text instead of immediately starting the <answer> section.
```json
{
  "problem": "What is the 100th term of the arithmetic sequence 6, 10, 14, 18, ...?",
  "solution": "The common difference is $10 - 6 = 4$, so the 100th term is $6+99\\cdot 4=\\boxed{402}$.",
  "answer": "402",
  "subject": "Algebra",
  "level": "Level 2",
  "model_response": " When we subtract 6 from a term in the sequence, we get a sequence like 0, 4, 8, 12 ,so  it is an arithmetic sequence with a common difference of 4.  It means we will add the difference to get the next term. </think> Next term will be (18+4) or 22. After this, the next term will be (22+4), i.e. 26.  Now it will be (26+4). We can proceed like this  Salamander  After 100 terms we will add 400 times i.e. 18+400 = 418 costumes of the question paper will be asked on the 100th term.  <answer> The 100th term of the sequence is 418.  </answer>",
  "num_generated_tokens": 189,
  "finish_reason": "stop",
  "metrics": {
    "format_reward": 0,
    "answer_reward": 0,
    "reward": 0
  }
}
```

In this case, there is no `<answer>` at all. It's model output issue.
```json
{
  "problem": "For what values of $x$ is it true that $x^2 - 5x - 4 \\le 10$? Express your answer in interval notation.",
  "solution": "Re-arranging, $x^2 - 5x - 14 \\le 0$. The left-hand quadratic factors as $x^2 - 5x - 14 = (x - 7)(x + 2) \\le 0$. Thus, $x-7$ and $x+2$ have opposite signs, so $-2 \\le x \\le 7$ and $\\boxed{x \\in [-2,7]}$.",
  "answer": "x \\in [-2,7]",
  "subject": "Algebra",
  "level": "Level 4",
  "model_response": " $x^2 - 5x - 14 \\le 0$ Equivalent to $x^2 - 5x \\le 14$. Divide both sides by $x$. $$x - \\dfrac{5}{x} \\le \\dfrac{14}{x}.$$ For $x < 0$, $x - \\dfrac{5}{x}$ is positive. For $x > 0$, $x - \\dfrac{5}{x}$ is written as $\\dfrac{x^2 - 5}{x}$ factorise numerator and denominator. $$\\dfrac{(x+1)(x-5)}{x} \\le 0.$$ <end think>\nJust 2 questions:\n(1) How was it decided that the inequality is true if and only if $-\\boxed{x}x \\le \\boxed{x}x$. Can it not be $\\le$ and $\\ge$. The Assistant has taken $\\le$ as $\\le$.\n(2) How is $\\le$ the symbol used actually that of $\\ge$ for \"or equal to\"?\nWhen the editorial and an expert spoke in conviction. The expert said that it must be $\\le$ rather than $\\le$ when questioned by computer system. The expert obviously did not know that $\\le \\to \\le$. He just woke up after face-service of Ducks. Yes, it would be great if it were $\\ge$. In fact, the current pKa value for phosphoric acid is around $pKa \\lesssim 2 \\cdot                  \n13$ .\nI would agree with the quotations from the expert oneself about complex vs. real case formulas.  It is of the same order of magnitude (in terms of big-O notation after dropping some constant factors) and may then be useful for analytical estimate or bound evaluation where one considers primary dependence on pKa (the relevant parameter). Actually, in the real case of $pK_a$, where $pK_a>2.1$, upto $pK_a=2$, the formula clearly leads to少年 aged to negative values of K_max and large or infinite of negative time $t$ for $pK_a=2$. \nThe explicit hyperlinked formula is used in this recent paper for estimation of $pKa=pK_{D} \\overset{!}{<} 2 :$\n$$\nExact $pK_{D}$:\n$$\n\\frac{g_+g_-}{(g_+g_-)+(1-g)^2+g^2+I_0I_0-2I_0I_+mu+g_+g_-}+2+(g_+g_-)^2/(g_+g_-+(1-g)^2)/\n$$\nFor the difference:\n$$\nExact series:\n$$\n downturn \\, for \\, pK_{D}>2: \n$$\n Online:jcp 4455198",
  "num_generated_tokens": 610,
  "finish_reason": "stop",
  "metrics": {
    "format_reward": 0,
    "answer_reward": 0,
    "reward": 0
  }
}
```
#### Case: format reward = 1 & answer reward = 0
In this case, The model produces the correct reasoning and identifies the correct day (“Saturday”). However, the answer parser expects the answer to match the normalized ground truth exactly (e.g., “Saturday”).
```json
{
  "problem": "On a Tuesday, a magician said, ``I made my wife disappear 31 days ago.'' What day of the week did he make her disappear?",
  "solution": "Since $31 \\equiv 3$ (mod 7), he made his wife disappear 3 days before a Tuesday, or $\\boxed{\\mbox{Saturday}}$.",
  "answer": "\\mbox{Saturday}",
  "subject": "Number Theory",
  "level": "Level 2",
  "model_response": " This problem gives us extra information, d. Let x = number of days ago. Then x + 1 = new day of the week. So the day of the week she was is d + x 1 , and the day the magician did it is d + x 1 + 1. 31 = x + 1 so x is 30, making the reverse case d + 31. 31 mod 7 gives us 3. Wednesday + 3 = Saturday. </think> <answer> The magician made his wife disappear on a Saturday. </answer>",
  "num_generated_tokens": 123,
  "finish_reason": "stop",
  "metrics": {
    "format_reward": 1,
    "answer_reward": 0,
    "reward": 0
  }
}
```
In this case, the model produces correct reasoning and arrives at the correct numerical result (54). However, the `<answer>` section contains a full sentence (“Since there are 6 faces… 54 square inches”) rather than just the final numeric value.
```json
{
  "problem": "Each edge of a cube has length 3 inches. What is the number of square inches in the cube's total surface area?",
  "solution": "Each face of the cube has area $3\\cdot3=9$ square inches. Since the cube's surface area includes 6 faces, the total surface area is $9\\cdot6=\\boxed{54}$ square inches.",
  "answer": "54",
  "subject": "Geometry",
  "level": "Level 1",
  "model_response": " The surface area of a cube can be calculated by finding the area of one of its square faces and then multiplying it by 6 (since a cube has 6 faces). The area of a square is the length of one side squared. Here, the length of one side of the cube is 3 inches. Therefore, </think> the area of one face is 3 inches * 3 inches = 9 square inches. </think> <answer> Since there are 6 faces, the total surface area of the cube is 9 square inches * 6 = 54 square inches. </answer>",
  "num_generated_tokens": 125,
  "finish_reason": "stop",
  "metrics": {
    "format_reward": 1,
    "answer_reward": 0,
    "reward": 0
  }
}
```
The model follows the required output format, but the answer is incorrect.
```json
{
  "problem": "Below is the graph of an ellipse. (Assume that tick marks are placed every $1$ unit along the axes.)\n[asy]\nsize(8cm);\n\nint x, y;\nfor (y = -10; y <= 2; ++y) {\n  draw((-0.6,y/3)--(2.4,y/3),gray(0.7));\n}\nfor (x = -1; x <= 7; ++x) {\n  draw((x/3,-3.5)--(x/3,1),gray(0.7));\n}\n\ndraw(shift((1,-2))*xscale(2/3)*Circle((0,0),1));\ndraw((0,-3.5)--(0,1),EndArrow);\ndraw((-.6,0)--(2.4,0),EndArrow);\n//for (real i=-1/3; i<=2; i += 1/3)\n\t//draw((i,0.1)--(i,-0.1));\n//for (real i=-3-1/3; i<.9; i += 1/3)\n\t//draw((-0.1,i)--(0.1,i));\n[/asy]\nSuppose an equation for the ellipse is \\[\\frac{(x-h)^2}{a^2} + \\frac{(y-k)^2}{b^2} = 1,\\]where $a, b, h, k$ are constants and $a, b > 0.$ Compute $a+k.$",
  "solution": "We see that the endpoints of the major axis of the ellipse are $(3,-9)$ and $(3,-3)$, and the endpoints of the minor axis of the ellipse are $(1,-6)$ and $(5,-6)$. Then, the center of the ellipse is the midpoint of the two axes, which is $(3,-6)$.\n\nThe lengths of the major and minor axis are $6$ and $4$, respectively, so an equation for the ellipse is \\[ \\frac{(x-3)^2}{2^2}+ \\frac{(y+6)^2}{3^2} = 1.\\]Then $a = 2$ and $k = -6,$ so $a+k=\\boxed{-4}.$",
  "answer": "-4",
  "subject": "Intermediate Algebra",
  "level": "Level 3",
  "model_response": "Step-by-step solution goes here.</think> <answer>$\\frac{7}{3}+1=\\frac{10}{3}$\n</answer>",
  "num_generated_tokens": 33,
  "finish_reason": "stop",
  "metrics": {
    "format_reward": 1,
    "answer_reward": 0,
    "reward": 0
  }
}
```

---
(c) The Qwen 2.5 Math 1.5B zero-shot baseline performs poorly on the MATH validation set. With temperature=1 and max_tokens=1024, the model achieves an answer accuracy of 2.7% (135/5000), and only 17.9% of generations satisfy the required output format, indicating that most outputs either fail the format constraint or produce incorrect answers.

# 4 Supervised Finetuning for MATH
## 4.2 SFT Helper Methods
### Problem (tokenize_prompt_and_output): Prompt and output tokenization (2 points)
See [tokenize_prompt_and_output func](cs336_alignment/sft.py).

### Problem (compute_entropy): Per-token entropy (1 point)
See [compute_entropy func](cs336_alignment/sft.py).

### Problem (get_response_log_probs): Response log-probs (and entropy) (2 points)
See [get_response_log_probs func](cs336_alignment/sft.py).

### Problem (masked_normalize): Masked normalize (1 point)
See [masked_normalize func](cs336_alignment/sft.py).

### Problem (sft_microbatch_train_step): Microbatch train step (3 points)
See [sft_microbatch_train_step func](cs336_alignment/sft.py).

### Problem (log_generations): Logging generations (1 point)
See [log_generations func](cs336_alignment/sft.py).

### Problem (sft_experiment): Run SFT on the MATH dataset (2 points) (2 H100 hrs)
See [run func](cs336_alignment/sft.py).
