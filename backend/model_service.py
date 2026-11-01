
from pathlib import Path
import threading

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
            "You are a SQL assistant. Given a table schema "
            "and a question, reply with ONLY the SQL query, "
            "nothing else."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Schema:\n{schema}\n\n"
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

        return sql