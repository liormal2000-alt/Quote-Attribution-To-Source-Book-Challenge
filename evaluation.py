import os

try:
    import litellm

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

# --- Configuration for the Judge Model ---

# We choose a powerful and reliable model to act as the judge.
JUDGE_MODEL = "openrouter/x-ai/grok-4-fast"

# This is the master prompt that instructs the judge model.
# It is designed to be clear, specific, and to force a simple output.
JUDGE_PROMPT_TEMPLATE = """
You are an expert evaluator for an AI research project. Your task is to determine if a language model's answer correctly identifies a book title.

You will be given the ground truth (the correct book title) and the model's full response.

**Instructions:**
1.  Read the "Correct Answer" and the "Model's Response".
2.  The model's response is correct if it clearly and correctly identifies the book title.
3.  Be forgiving of extra text. If the response is "The book is 'The Hobbit'", and the correct answer is "The Hobbit", the answer is correct.
4.  Be strict about the title itself. "The Lord of the Rings" is NOT the same as "The Hobbit".
5.  Pay close attention to negations. If the response says "This is NOT from 'The Hobbit'", the answer is incorrect.
6.  Your final output must be a single word: **Yes** or **No**. Do not add any other explanation.

**Correct Answer:** "{true_answer}"

**Model's Response:** "{predicted_answer}"

Is the model's identification of the book title correct?
Your Answer (Yes/No):
"""


def eval(predicted_answer: str, ground_truth_example: dict, api_key: str) -> bool:
    """
    Evaluates if a model's prediction is correct by using another powerful LLM as a judge.

    Args:
        predicted_answer (str): The full text output from the model being tested.
        ground_truth_example (dict): The example object from the dataset.
        api_key (str): The OpenRouter API key to use for the judge model call.

    Returns:
        bool: True if the judge model determines the answer is correct, False otherwise.
    """
    if not LITELLM_AVAILABLE:
        return False

    true_title = ground_truth_example.get("answer")

    # Basic sanity checks to avoid unnecessary API calls
    if not true_title or not predicted_answer or "Error:" in predicted_answer:
        return False

    # Format the prompt for the judge model
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        true_answer=true_title,
        predicted_answer=predicted_answer
    )

    messages = [{"role": "user", "content": judge_prompt}]

    try:
        # Call the judge model using litellm
        response = litellm.completion(
            model=JUDGE_MODEL,
            messages=messages,
            api_key=api_key,
        )

        # Process the judge's response
        judge_response = response.choices[0].message.content.strip().lower()

        # Check if the judge's simple "Yes/No" answer indicates correctness
        if "yes" in judge_response:
            return True

    except Exception as e:
        print(f"   An error occurred while calling the Judge LLM: {e}")
        # If the judge fails for any reason, we must assume the answer is incorrect
        # to avoid accidentally giving points for a failed check.
        return False

    # If the judge does not say "yes", the answer is considered incorrect.
    return False