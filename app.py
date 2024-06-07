from operator import truediv
from datasets import load_dataset, Dataset
import pandas as pd

"""
HellaSwag

hellaswag_ds = load_dataset("Rowan/hellaswag", trust_remote_code=True)
hellaswag_ds.save_to_disk("data/hellaswag")
"""

"""
Truthful_QA

gen_dataset = load_dataset("truthful_qa", "generation")[
            "validation"
        ]
mc_dataset = load_dataset("truthful_qa", "multiple_choice")[
    "validation"
]
df_mc, df_gen = mc_dataset.to_pandas(), gen_dataset.to_pandas()
merged_df = pd.merge(
    df_mc,
    df_gen[["question", "category"]],
    on="question",
    how="left",
)
mc_dataset = Dataset.from_pandas(merged_df)
mc_dataset.save_to_disk("data/truthfulqa")
"""

"""
HumanEval
humaneval_ds = load_dataset("openai_humaneval", trust_remote_code=True)
humaneval_ds.save_to_disk("data/humaneval")
"""

"""
MMLU

mmlu_ds = load_dataset("lukaemon/mmlu", 'formal_logic', trust_remote_code=True)
mmlu_ds.save_to_disk("data/mmlu")
"""

"""
GSM8K
"""
gsm8k_ds = load_dataset("gsm8k", "main", trust_remote_code=True)
gsm8k_ds.save_to_disk("data/gsm8k")