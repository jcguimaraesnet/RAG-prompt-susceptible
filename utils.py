import pandas as pd
import time

from dotenv import load_dotenv
import os

def remove_nul_chars_from_string(s):
    """Remove NUL characters from a single string."""
    return s.replace('\x00', '')

def remove_nul_chars_from_run_data(run_data):
    """Iterate over all fields of RunData to remove NUL characters."""
    for run in run_data:
        run.reference_question = remove_nul_chars_from_string(run.reference_question)
        run.reference_answer = remove_nul_chars_from_string(run.reference_answer)
        run.llm_answer = remove_nul_chars_from_string(run.llm_answer)
        run.llm_context = [remove_nul_chars_from_string(context) for context in run.llm_context]

def make_get_llama_response(query_engine):
    def get_llama_response(prompt):
        response = query_engine.custom_query(prompt)
        answer = response['response']
        context = response['llm_context_list']

        return {
            "llm_answer": answer,
            "llm_context_list": context
        }
    return get_llama_response

def chunked_iterable(iterable, size):
    """Yield successive size chunks from iterable."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

# Function to load and validate configuration settings
def load_config():
    # Load environment variables from a .env file
    load_dotenv()


def run_experiment(experiment_name, query_engine, rate_replaced_words, scorer, benchmark, validate_api, project_key, upload_results=True, runs=5):
    # List to store results dictionaries
    results_list = []

    for i in range(runs):
        get_llama_response_func = make_get_llama_response(query_engine)

        run = scorer.score(benchmark,
                           get_llama_response_func,
                           callback_parallelism=1,
                           scoring_parallelism=1)
        print(f"{experiment_name} Run {i+1} Overall Scores: {run.overall_scores} \n\n")
        #print(f"Detailed Scores: {run.run_data} \n\n")
        remove_nul_chars_from_run_data(run.run_data)

        # Add this run's results to the list
        results_list.append({'Run': i+1, 'Experiment': experiment_name, 'OverallScores': run.overall_scores})

        if upload_results:
            validate_api.upload_run(project_key, run=run, run_metadata={"approach task": experiment_name, "rate replaced words": rate_replaced_words})
        else:
            print(f"Skipping upload for {experiment_name} Run {i+1}.")

    # Create a DataFrame from the list of results dictionaries
    results_df = pd.DataFrame(results_list)

    # Return the DataFrame containing all the results
    return results_df

def filter_large_nodes(nodes, max_length=8000):
    """
    Filters out nodes with 'window' or 'text' length greater than max_length.
    Needed bcs sometimes the sentences are too long due to tables or refereneces in data.
    It creates one giga long non-sensical sentence. Before filtering please do analysis
    so that you dont throw out anything important.

    Args:
    - nodes (list): List of node objects.
    - max_length (int): Maximum allowed length for 'window' and 'text'.

    Returns:
    - list: Filtered list of nodes.
    """
    filtered_nodes = []
    for node in nodes:
        text_length = len(node.text)
        window_length = len(node.metadata.get('window', ''))

        if text_length <= max_length and window_length <= max_length:
            filtered_nodes.append(node)
    return filtered_nodes
