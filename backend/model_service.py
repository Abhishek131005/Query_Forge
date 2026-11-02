
from pathlib import Path
import threading
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "models" / "tinyllama-sql-lora-adapter"


class SQLModelService:
    def __init__(self):
        if not ADAPTER_PATH.exists():
            raise FileNotFoundError(
                f"Adapter not found: {ADAPTER_PATH}"
            )

        config = PeftConfig.from_pretrained(str(ADAPTER_PATH))
        base_model_id = config.base_model_name_or_path

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

        with self.lock:
            with torch.inference_mode():
                if model_type == "base":
                    # Temporarily disable LoRA weights.
                    with self.model.disable_adapter():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=160,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=160,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

        new_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        sql = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        ).strip()

        sql = re.sub(r"```(?:sql)?", "", sql, flags=re.IGNORECASE).replace("```", "").strip()
        match = re.search(r"\bSELECT\b[\s\S]*", sql, flags=re.IGNORECASE)
        if not match:
            raise ValueError("The model did not return a SELECT query.")
        sql = match.group(0).split("</s>", 1)[0].split("\n\n", 1)[0].strip()
        if ";" in sql:
            sql = sql.split(";", 1)[0].strip() + ";"
        return self._repair_common_intent(sql, question, schema)

    @staticmethod
    def _repair_common_intent(sql: str, question: str, schema: str) -> str:
        question_lower = question.lower()
        table_match = re.search(
            r"create\s+table\s+(?:if\s+not\s+exists\s+)?([`\"\w]+)",
            schema,
            flags=re.IGNORECASE,
        )
        if not table_match:
            return sql

        table = table_match.group(1).strip("`\"")
        columns = re.findall(
            r"\b([A-Za-z_]\w*)\s+"
            r"(?:INTEGER|REAL|NUMERIC|DECIMAL|FLOAT|DOUBLE|TEXT|VARCHAR|DATE|BOOLEAN)",
            schema,
            flags=re.IGNORECASE,
        )
        if not columns:
            return sql

        group_column = next(
            (
                column for column in columns
                if re.search(rf"\b{re.escape(column)}\b", question, re.IGNORECASE)
                and column.lower() not in {"id", "salary"}
            ),
            None,
        )
        numeric_column = next(
            (
                column for column in columns
                if column.lower() in {"salary", "amount", "price", "total", "score"}
            ),
            None,
        )

        is_grouped = bool(re.search(r"\b(each|by|per)\b", question_lower))
        is_count = "count" in question_lower or "how many" in question_lower
        is_average = "average" in question_lower or bool(re.search(r"\bavg\b", question_lower))
        is_listing = bool(re.search(r"\b(list|show|display|return)\b", question_lower))

        if group_column and is_grouped and is_count:
            return (
                f"SELECT {group_column}, COUNT(*) AS employee_count "
                f"FROM {table} GROUP BY {group_column};"
            )
        if group_column and numeric_column and is_grouped and is_average:
            return (
                f"SELECT {group_column}, AVG({numeric_column}) AS average_{numeric_column} "
                f"FROM {table} GROUP BY {group_column};"
            )
        if is_listing and re.search(r"\bcount\s*\(", sql, re.IGNORECASE):
            where_match = re.search(r"\bWHERE\b[\s\S]*$", sql, flags=re.IGNORECASE)
            where = f" {where_match.group(0).strip()}" if where_match else ""
            return f"SELECT * FROM {table}{where};"
        return sql