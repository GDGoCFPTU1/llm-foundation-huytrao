"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

# Import OpenAI client
from openai import OpenAI

# Import Google Gemini client
from google import genai
from google.genai import types

# Import Anthropic client
import anthropic

# Load environment variables from .env file if present
from dotenv import load_dotenv
load_dotenv()

# Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call OpenAI Chat Completions API.

    Returns:
        (
            response_text,
            latency_seconds,
            {
                "input_tokens": int,
                "output_tokens": int
            }
        )
    """

    start_time = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    latency = time.time() - start_time

    response_text = response.choices[0].message.content or ""

    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }

    return response_text, latency, usage
    


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    # Create config with parameters
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    
    # Measure latency
    start_time = time.time()
    response = gemini_client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.time() - start_time
    
    # Extract response text
    response_text = response.text or ""
    
    # Extract usage metadata
    usage = {
        "input_tokens": response.usage_metadata.input_token_count,
        "output_tokens": response.usage_metadata.output_token_count,
    }
    
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    # Measure latency
    start_time = time.time()
    message = anthropic_client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    latency = time.time() - start_time
    
    # Extract response text
    response_text = message.content[0].text or ""
    
    # Extract usage metadata
    usage = {
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
    }
    
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    # Call OpenAI GPT-4o
    gpt4o_response, gpt4o_latency, gpt4o_usage = call_openai(prompt, model=OPENAI_MODEL)
    gpt4o_cost = (gpt4o_usage["input_tokens"] * PRICING_1M_TOKENS["gpt-4o"]["input"] + 
                  gpt4o_usage["output_tokens"] * PRICING_1M_TOKENS["gpt-4o"]["output"]) / 1_000_000
    
    # Call OpenAI GPT-4o Mini
    gpt4o_mini_response, gpt4o_mini_latency, gpt4o_mini_usage = call_openai(prompt, model=OPENAI_MINI_MODEL)
    gpt4o_mini_cost = (gpt4o_mini_usage["input_tokens"] * PRICING_1M_TOKENS["gpt-4o-mini"]["input"] + 
                       gpt4o_mini_usage["output_tokens"] * PRICING_1M_TOKENS["gpt-4o-mini"]["output"]) / 1_000_000
    
    # Call Gemini 2.5 Flash
    gemini_response, gemini_latency, gemini_usage = call_gemini(prompt, model=GEMINI_MODEL)
    gemini_cost = (gemini_usage["input_tokens"] * PRICING_1M_TOKENS["gemini-2.5-flash"]["input"] + 
                   gemini_usage["output_tokens"] * PRICING_1M_TOKENS["gemini-2.5-flash"]["output"]) / 1_000_000
    
    return {
        "gpt4o": {
            "response": gpt4o_response,
            "latency": gpt4o_latency,
            "cost": gpt4o_cost,
            "input_tokens": gpt4o_usage["input_tokens"],
            "output_tokens": gpt4o_usage["output_tokens"],
        },
        "gpt4o_mini": {
            "response": gpt4o_mini_response,
            "latency": gpt4o_mini_latency,
            "cost": gpt4o_mini_cost,
            "input_tokens": gpt4o_mini_usage["input_tokens"],
            "output_tokens": gpt4o_mini_usage["output_tokens"],
        },
        "gemini_flash": {
            "response": gemini_response,
            "latency": gemini_latency,
            "cost": gemini_cost,
            "input_tokens": gemini_usage["input_tokens"],
            "output_tokens": gemini_usage["output_tokens"],
        },
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.

    Behaviour:
        - Streams response tokens from Gemini 2.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    history = []
    
    print("\n=== Gemini 2.5 Streaming Chatbot ===")
    print("(Type 'quit' or 'exit' to end)\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        
        # Skip empty inputs
        if not user_input:
            continue
        
        # Add user message to history
        history.append({"role": "user", "content": user_input})
        
        # Keep only last 3 turns (6 messages)
        if len(history) > 6:
            history = history[-6:]
        
        # Create streaming config
        config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=256,
        )
        
        # Stream response from Gemini
        print("\nBot: ", end="", flush=True)
        full_response = ""
        
        response_stream = gemini_client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=history,
            config=config,
        )
        
        for chunk in response_stream:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text
        
        print("\n")
        
        # Add bot response to history
        history.append({"role": "model", "content": full_response})


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    last_exception = None
    
    # Try initial call
    try:
        return fn()
    except Exception as e:
        last_exception = e
    
    # Retry with exponential backoff
    for attempt in range(max_retries):
        delay = base_delay * (2 ** attempt)
        time.sleep(delay)
        
        try:
            return fn()
        except Exception as e:
            last_exception = e
            continue
    
    # Raise last exception if all retries failed
    raise last_exception


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    results = []
    
    for prompt in prompts:
        comparison = compare_models(prompt)
        comparison["prompt"] = prompt
        results.append(comparison)
    
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    table = "| Prompt | Model | Response (truncated) | Latency (s) | Tokens (In/Out) | Cost (USD) |\n"
    table += "|--------|-------|----------------------|-------------|-----------------|------------||\n"
    
    for result in results:
        prompt = result["prompt"][:30] + "..." if len(result["prompt"]) > 30 else result["prompt"]
        
        # Add row for each model
        for model_key in ["gpt4o", "gpt4o_mini", "gemini_flash"]:
            if model_key in result:
                model_data = result[model_key]
                response_text = model_data["response"][:50] + "..." if len(model_data["response"]) > 50 else model_data["response"]
                latency = f"{model_data['latency']:.2f}"
                tokens = f"{model_data['input_tokens']}/{model_data['output_tokens']}"
                cost = f"${model_data['cost']:.6f}"
                
                table += f"| {prompt} | {model_key} | {response_text} | {latency} | {tokens} | {cost} |\n"
    
    return table


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
