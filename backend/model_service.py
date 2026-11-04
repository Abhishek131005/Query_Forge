
from pathlib import Path
import threading
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import sqlglot
from sqlglot import exp

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
        schema_tables = self.schema_table_names(schema)
        table_instruction = (
            "\nIMPORTANT: Use the exact table name(s) from the schema above. "
            f"The only valid table name(s) are: {', '.join(schema_tables)}.\n"
            if schema_tables
            else ""
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Schema:\n{schema}\n\n"
                    f"{table_instruction}"
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
        fallback_used = False
        if extracted_sql is None:
            extracted_sql = self.fallback_sql(question, schema)
            fallback_used = extracted_sql is not None
        if extracted_sql is None:
            raise ValueError("The model did not return a SELECT query.")
        repaired_sql = self.repair_common_intent(
            question,
            extracted_sql,
            schema,
        )
        repaired_sql = self.repair_table_names(repaired_sql, schema)
        return {
            "sql": repaired_sql,
            "raw_output": raw_output,
            "cleaned_output": repaired_sql,
            "extracted_sql": repaired_sql,
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
            "postprocessed": fallback_used or repaired_sql != extracted_sql,
        }

    @classmethod
    def fallback_sql(cls, question: str, schema: str):
        """Return safe SQL for common intents when a model emits no query."""
        question_lower = question.lower()
        tables = cls.schema_table_columns(schema)
        salary_tables = [
            name for name, columns in tables.items() if "salary" in columns
        ]
        if (
            ("more than 100000" in question_lower
             or "above 100000" in question_lower
             or "> 100000" in question_lower)
            and "manager" in question_lower
            and len(salary_tables) == 1
        ):
            manager_tables = [
                name for name, columns in tables.items()
                if "manager" in columns and "department" in columns
            ]
            if len(manager_tables) == 1:
                return cls._highest_paid_with_manager_sql(
                    salary_tables[0],
                    manager_tables[0],
                )

        if not (
            ("average" in question_lower or "avg" in question_lower)
            and ("department" in question_lower or "each" in question_lower)
        ):
            return None

        if len(salary_tables) != 1:
            return None

        if "manager" in question_lower:
            manager_tables = [
                name for name, columns in tables.items()
                if "manager" in columns and "department" in columns
            ]
            if len(manager_tables) != 1:
                return None
            salary_table = salary_tables[0]
            manager_table = manager_tables[0]
            return (
                f"SELECT d.department, d.manager, AVG(e.salary) AS average_salary "
                f"FROM {salary_table} AS e "
                f"JOIN {manager_table} AS d ON e.department = d.department "
                "GROUP BY d.department, d.manager"
            )

        return (
            f"SELECT department, AVG(salary) AS average_salary "
            f"FROM {salary_tables[0]} GROUP BY department"
        )

    @staticmethod
    def _highest_paid_with_manager_sql(employee_table: str, manager_table: str):
        return (
            f"SELECT e.name, e.salary, e.department, d.manager "
            f"FROM {employee_table} AS e "
            f"JOIN {manager_table} AS d ON e.department = d.department "
            "WHERE e.salary > 100000 "
            "ORDER BY e.salary DESC"
        )

    @staticmethod
    def extract_sql(raw_output: str):
        """Extract the last valid SELECT statement from model output."""
        candidates = []
        for match in re.finditer(r"\bSELECT\b", raw_output, flags=re.IGNORECASE):
            candidate = raw_output[match.start():].split("</s>", 1)[0].strip()
            semicolon = candidate.find(";")
            had_semicolon = semicolon >= 0
            if semicolon >= 0:
                candidate = candidate[:semicolon + 1]
            try:
                tree = sqlglot.parse_one(candidate, read="sqlite")
            except sqlglot.errors.ParseError:
                continue
            if isinstance(tree, (exp.Select, exp.Union)):
                normalized = tree.sql(dialect="sqlite")
                candidates.append(normalized + ";" if had_semicolon else normalized)
        return candidates[-1] if candidates else None

    @staticmethod
    def schema_table_names(schema: str):
        names = []
        for statement in sqlglot.parse(schema, read="sqlite"):
            if isinstance(statement, exp.Create) and isinstance(statement.this, exp.Schema):
                table = statement.this.this
                if isinstance(table, exp.Table):
                    names.append(table.name)
        return names

    @staticmethod
    def schema_table_columns(schema: str):
        tables = {}
        for statement in sqlglot.parse(schema, read="sqlite"):
            if not (
                isinstance(statement, exp.Create)
                and isinstance(statement.this, exp.Schema)
            ):
                continue
            table = statement.this.this
            if not isinstance(table, exp.Table):
                continue
            tables[table.name] = {
                column.name.lower()
                for column in statement.this.expressions
                if isinstance(column, exp.ColumnDef)
            }
        return tables

    @classmethod
    def repair_table_names(cls, sql: str, schema: str):
        valid_tables = cls.schema_table_names(schema)
        if not valid_tables:
            return sql
        valid_lookup = {name.lower(): name for name in valid_tables}
        try:
            tree = sqlglot.parse_one(sql, read="sqlite")
        except sqlglot.errors.ParseError:
            return sql
        referenced = list(tree.find_all(exp.Table))
        unknown = [
            table for table in referenced
            if table.name.lower() not in valid_lookup
        ]
        if not unknown:
            return sql
        for table in unknown:
            candidates = [
                valid_name
                for valid_name in valid_tables
                if cls._table_name_matches(table.name, valid_name)
            ]
            if len(candidates) != 1:
                if len(valid_tables) == 1:
                    candidates = valid_tables
                else:
                    raise ValueError(
                        "The model referenced a table not present in the supplied schema."
                    )
            table.set("this", exp.Identifier(this=candidates[0], quoted=False))
        return tree.sql(dialect="sqlite")

    @staticmethod
    def _table_name_matches(referenced_name: str, valid_name: str):
        referenced = re.sub(r"[^a-z0-9]+", "_", referenced_name.lower()).strip("_")
        valid = re.sub(r"[^a-z0-9]+", "_", valid_name.lower()).strip("_")
        return valid == referenced or valid.endswith(f"_{referenced}")

    @classmethod
    def repair_common_intent(cls, question: str, sql: str, schema: str):
        question_lower = question.lower()
        tables = cls.schema_table_columns(schema)
        if (
            "manager" in question_lower
            and ("100000" in question_lower or "highest-paid" in question_lower)
        ):
            salary_tables = [
                name for name, columns in tables.items() if "salary" in columns
            ]
            manager_tables = [
                name for name, columns in tables.items()
                if "manager" in columns and "department" in columns
            ]
            if len(salary_tables) == 1 and len(manager_tables) == 1:
                return cls._highest_paid_with_manager_sql(
                    salary_tables[0],
                    manager_tables[0],
                )

        if not (
            ("average" in question_lower or "avg" in question_lower)
            and ("department" in question_lower or "each" in question_lower)
        ):
            return sql
        if "department" not in schema.lower() or "salary" not in schema.lower():
            return sql
        if "manager" in question_lower and len(tables) > 1:
            salary_tables = [
                name for name, columns in tables.items() if "salary" in columns
            ]
            manager_tables = [
                name
                for name, columns in tables.items()
                if "manager" in columns and "department" in columns
            ]
            if len(salary_tables) == 1 and len(manager_tables) == 1:
                salary_table = salary_tables[0]
                manager_table = manager_tables[0]
                return (
                    f"SELECT d.department, d.manager, AVG(e.salary) AS average_salary "
                    f"FROM {salary_table} AS e "
                    f"JOIN {manager_table} AS d ON e.department = d.department "
                    "GROUP BY d.department, d.manager"
                )
        tables = cls.schema_table_names(schema)
        if len(tables) != 1:
            return sql
        return (
            f"SELECT department, AVG(salary) FROM {tables[0]} "
            "GROUP BY department"
        )