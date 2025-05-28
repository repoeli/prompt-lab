import os
import time
import logging
from typing import Any, Dict, List, Optional

import openai

# Configure logging for cost tracking
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Token pricing per model ($ per token)
# Customize via environment variable OPENAI_PRICING if needed
PRICE: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"in": 0.60e-6, "out": 2.40e-6},
    # Add other models and pricing here
}

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def run_chat(
    model: str,
    messages: List[Dict[str, Any]],
    user: Optional[str] = None,
    **kwargs: Any
) -> str:
    """
    Sends a chat completion request and logs detailed cost and latency metrics.

    Args:
        model: Model name, e.g. 'gpt-4o-mini'.
        messages: List of message dicts with 'role' and 'content'.
        user: Optional user identifier for OpenAI billing.
        **kwargs: Additional parameters to pass to OpenAI API.

    Returns:
        Assistant's reply as a string.
    """
    # Start timer
    start_time = time.perf_counter()

    # Prepare API call parameters
    params = {
        "model": model,
        "messages": messages,
    }
    if user:
        params["user"] = user
    params.update(kwargs)

    try:
        # Invoke OpenAI API
        response = client.chat.completions.create(**params)

        # Extract usage metrics
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)

        # Compute cost
        pricing = PRICE.get(model)
        if pricing:
            cost = (prompt_tokens * pricing["in"]) + (completion_tokens * pricing["out"])
        else:
            cost = 0.0
            logger.warning(f"Pricing not found for model {model}; cost set to 0.")

        # Compute latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Log structured cost and usage
        logger.info(
            "Model=%s User=%s TokensIn=%d TokensOut=%d TotalTokens=%d Cost=%.6f LatencyMs=%.2f",
            model,
            user or "unknown",
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cost,
            latency_ms,
        )

        # Return the assistant message
        return response.choices[0].message.content

    except Exception as e:
        logger.error("OpenAI API call failed: %s", e, exc_info=True)
        raise

# Convenience function for notebook use
def run(model: str, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
    """
    Simplified interface for notebook use.
    Alias for run_chat with cleaner signature.
    """
    return run_chat(model, messages, **kwargs)