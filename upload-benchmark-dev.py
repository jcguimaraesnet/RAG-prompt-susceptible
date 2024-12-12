import os
import json
from tonic_validate import ValidateApi, Benchmark
from dotenv import load_dotenv

load_dotenv()

BENCHMARK_NAME="benchmark-sensible-prompt-final"

tonic_validate_api_key = os.getenv("TONIC_VALIDATE_API_KEY")
validate_api = ValidateApi(tonic_validate_api_key)

MODEL = os.getenv("MODEL")

# Carregar o JSON a partir do arquivo
with open('eval_questions/benchmark.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# dev
# questions = data['questions'][:10]
# answers = data['ground_truths'][:10]

# final
questions = data['questions']
answers = data['ground_truths']

#print(questions)

benchmark = Benchmark(questions=questions, answers=answers)

tonic_validate_benchmark_key = validate_api.new_benchmark(benchmark=benchmark, 
                                                          benchmark_name=BENCHMARK_NAME)
print(f"Created benchmark '{BENCHMARK_NAME}' with key: {tonic_validate_benchmark_key} in https://validate.tonic.ai")
