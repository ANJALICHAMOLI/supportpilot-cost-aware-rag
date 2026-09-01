from routing.model_router import CHEAP_MODEL  

MODEL_PRICING = {
    "qwen/qwen3.6-27b": {
        "input_per_1k": 0.00005,
        "output_per_1k": 0.00008,
    },
    "gemini-flash-latest": {
        "input_per_1k": 0.0003,
        "output_per_1k": 0.0025,
    },
}


def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:

    if model_name not in MODEL_PRICING:
        raise ValueError(f"No pricing configured for model: {model_name}")

    pricing = MODEL_PRICING[model_name]
    input_cost = (input_tokens / 1000) * pricing["input_per_1k"]
    output_cost = (output_tokens / 1000) * pricing["output_per_1k"]
    return round(input_cost + output_cost, 6)


def estimate_cost_before_call(model_name: str, estimated_input_tokens: int, estimated_output_tokens: int = 300) -> float:
    return estimate_cost(model_name, estimated_input_tokens, estimated_output_tokens)


def rough_token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


class BudgetTracker:

    def __init__(self, budget_limit: float = 1.0):

        self.budget_limit = budget_limit
        self.total_spent = 0.0

    def remaining_budget(self) -> float:
        return round(self.budget_limit - self.total_spent, 6)

    def record_spend(self, amount: float):
        self.total_spent = round(self.total_spent + amount, 6)

    def check_and_resolve_model(self, requested_model: str, question: str, context_text: str) -> dict:

        estimated_input_tokens = rough_token_estimate(question + context_text)

        if requested_model == CHEAP_MODEL:
            est_cost = estimate_cost_before_call(requested_model, estimated_input_tokens)
            return {"final_model": requested_model, "was_downgraded": False, "estimated_cost": est_cost}

        strong_est_cost = estimate_cost_before_call(requested_model, estimated_input_tokens)

        if strong_est_cost <= self.remaining_budget():
            return {"final_model": requested_model, "was_downgraded": False, "estimated_cost": strong_est_cost}

        cheap_est_cost = estimate_cost_before_call(CHEAP_MODEL, estimated_input_tokens)
        return {"final_model": CHEAP_MODEL, "was_downgraded": True, "estimated_cost": cheap_est_cost}
