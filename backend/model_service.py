
from pathlib import Path
import threading
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "models" / "tinyllama-sql-lora-adapter"
PROMPT_VERSION = "v1"
MAX_NEW_TOKENS = 160
DO_SAMPLE = False


class SQLModelService:
    def __init__(self):
        if not ADAPTER_PATH.exists():
            raise FileNotFoundError(
                f"Adapter not found: {ADAPTER_PATH}"
            )

        config = PeftConfig.from_pretrained(str(ADAPTER_PATH))
        base_model_id = config.base_model_name_or_path
        self.base_model_id = base_model_id

        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(ADAPTER_PATH)
        )

        chat_template_path = ADAPTER_PATH / "chat_template.jinja"
        if not chat_template_path.exists():
            raise FileNotFoundError(
                f"Chat template not found: {chat_template_path}"
            )
        self.tokenizer.chat_template = chat_template_path.read_text(
            encoding="utf-8"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = (
            torch.float16 if self.device == "cuda"
            else torch.float32
        )

        print(f"Loading base model: {base_model_id}")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            str(ADAPTER_PATH)
        )

        self.model.to(self.device)
        self.model.eval()

        # Prevent concurrent adapter switching.
        self.lock = threading.Lock()

        print(f"Models ready on {self.device}")

    def generate_sql(
        self,
        question: str,
        schema: str,
        model_type: str = "finetuned"
    ):
        if model_type not in ("base", "finetuned"):
            raise ValueError("Unsupported model type")

        system_prompt = (
            "You are a precise SQLite SQL assistant. Given the exact table "
            "schema and a user's question, write one read-only SELECT query. "
            "Preserve the user's requested operation: list means return rows, "
            "count means COUNT, average means AVG, and 'by' or 'each' means "
            "GROUP BY. Use only tables and columns present in the schema. "
            "Do not invent filters, values, joins, or aggregations. "
            "Reply with ONLY the SQL query, without markdown or explanation."
        )
        examples = (
            "\nExamples:\n"
            "Schema: CREATE TABLE employees (name TEXT, department TEXT, salary INTEGER);\n"
            "Question: List Engineering employees earning more than 100000.\n"
            "SQL: SELECT name, salary FROM employees WHERE department = 'Engineering' AND salary > 100000;\n"
            "Question: How many employees are in each department?\n"
            "SQL: SELECT department, COUNT(*) AS employee_count FROM employees GROUP BY department;\n"
            "Question: What is the average salary by department?\n"
            "SQL: SELECT department, AVG(salary) AS average_salary FROM employees GROUP BY department;\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Schema:\n{schema}\n\n"
                    f"{examples}\n"
                    f"Question: {question}"
                )
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        queue_start = time.perf_counter()
        with self.lock:
            queue_wait_seconds = time.perf_counter() - queue_start
            if self.device == "cuda":
                torch.cuda.synchronize()
            inference_start = time.perf_counter()
            with torch.inference_mode():
                if model_type == "base":
                    # Temporarily disable LoRA weights.
                    with self.model.disable_adapter():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=DO_SAMPLE,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=DO_SAMPLE,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
            if self.device == "cuda":
                torch.cuda.synchronize()
            inference_time_seconds = time.perf_counter() - inference_start

        new_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        raw_output = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        )
        extracted_sql = self.extract_sql(raw_output)
        if extracted_sql is None:
            raise ValueError("The model did not return a SELECT query.")
        return {
            "sql": extracted_sql,
            "raw_output": raw_output,
            "extracted_sql": extracted_sql,
            "model": model_type,
            "status": "success",
            "inference_time_seconds": inference_time_seconds,
            "queue_wait_seconds": queue_wait_seconds,
            "input_tokens": inputs["input_ids"].shape[1],
            "generated_tokens": new_tokens.shape[0],
            "device": self.device,
            "generation_config": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": DO_SAMPLE,
            },
            "prompt_version": PROMPT_VERSION,
            "postprocessed": False,
        }

    @staticmethod
    def extract_sql(raw_output: str):
        """Extract the first SELECT statement without changing model output."""
        match = re.search(r"\bSELECT\b[\s\S]*", raw_output, flags=re.IGNORECASE)
        if not match:
            return None
        extracted = match.group(0).split("</s>", 1)[0].split("\n\n", 1)[0].strip()
        if ";" in extracted:
            extracted = extracted.split(";", 1)[0].strip() + ";"
        return extracted or None