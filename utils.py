import replicate
import time

# Initialize debounce variables
last_call_time = 0

def debounce_replicate_run(llm, prompt, max_len, temperature, top_p, API_TOKEN, debounce_interval=2):
    """
    Prevents rapid consecutive calls to replicate.run by enforcing a debounce interval.

    Parameters:
    - llm: Language model instance
    - prompt: Text prompt for the language model
    - max_len: Maximum length of the generated text
    - temperature: Temperature for text generation
    - top_p: Top-p value for text generation
    - API_TOKEN: API token for the replicate.run function
    - debounce_interval: Minimum time (in seconds) to wait between calls

    Returns:
    - Generated text or an error message if calls are too frequent.
    """
    global last_call_time

    # Get the current time
    current_time = time.time()

    # Calculate the time elapsed since the last call
    elapsed_time = current_time - last_call_time

    # Check if the elapsed time is less than the debounce interval
    if elapsed_time < debounce_interval:
        return "Please wait a moment before making another request."

    try:
        # Update the last call time to the current time
        last_call_time = current_time

        output = replicate.run(
            llm,
            input={"prompt": prompt + "Assistant: ", "max_length": max_len, "temperature": temperature, "top_p": top_p, "repetition_penalty": 1},
            api_token=API_TOKEN
        )

        return output
    except Exception as e:
        return f"An error occurred: {str(e)}"
