from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMClient:
    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_length: int = 100, **gen_kwargs) -> str:
        """
        Generates a continuation given `prompt`, returning full text including prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=len(inputs["input_ids"][0]) + max_length,
            **gen_kwargs
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
