# SETUP
For those who are following along at home, you can get the same **MATH dataset** by running this [scripts](scripts/download_math_data.py).

To download Qwen2.5-Math-1.5B, you can run this [scripts](scripts/download_Qwen_model.py).

# 3 Measuring Zero-Shot MATH Performance
## 3.2 Zero-shot MATH Baseline
### Problem (math_baseline): 4 points
(a) Here are the [script](scripts/evaluate_math_zero_shot.py) and the evaluation [output](outputs/qwen25_math_1p5b_r1_zero_math_validation%20.jsonl).

(b)
|              | **Answer = 1** | **Answer = 0** | **Total** |
|--------------|---------------|---------------|-----------|
| **Format = 1** | 859 (17.18%) | 1245 (24.90%) | 2104 (42.08%) |
| **Format = 0** | 0 (0.00%) | 2896 (57.92%) | 2896 (57.92%) |
| **Total**      | 859 (17.18%) | 4141 (82.82%) | 5000 (100%) |