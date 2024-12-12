import os
from dotenv import load_dotenv
import openai
import chromadb
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from tonic_validate import ValidateScorer, ValidateApi
from tonic_validate.metrics import RetrievalPrecisionMetric, AnswerConsistencyBinaryMetric, AnswerSimilarityMetric, AugmentationAccuracyMetric, AugmentationPrecisionMetric
import pandas as pd
from utils import run_experiment
from custom_posrewriting_query_engine import CustomPosRewritingQueryEngine

load_dotenv()

MODEL = os.getenv("MODEL")

# Set the OpenAI API key for authentication.
openai.api_key = os.getenv("OPENAI_API_KEY")
tonic_validate_api_key = os.getenv("TONIC_VALIDATE_API_KEY")
tonic_validate_project_key = os.getenv("TONIC_VALIDATE_PROJECT_KEY")
tonic_validate_benchmark_key = os.getenv("TONIC_VALIDATE_BENCHMARK_KEY")
validate_api = ValidateApi(tonic_validate_api_key)
# Tonic Validate setup
print(f"Loading benchmark with key: {tonic_validate_benchmark_key}")
benchmark = validate_api.get_benchmark(tonic_validate_benchmark_key)

# Service context
llm = OpenAI(model=MODEL, temperature=0.0)
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Traditional VDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_collection("ai_arxiv_full")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection, embed_model=embed_model)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

# prompt template
text_qa_template_str = ""
with open("resources/text_qa_template.txt", 'r', encoding='utf-8') as file:
    text_qa_template_str = file.read()
text_qa_template = PromptTemplate(text_qa_template_str)

text_qa_template_ptbr_str = ""
with open("resources/text_qa_template_ptbr.txt", 'r', encoding='utf-8') as file:
    text_qa_template_ptbr_str = file.read()
text_qa_template_ptbr = PromptTemplate(text_qa_template_ptbr_str)

text_qa_template_invert_str = ""
with open("resources/text_qa_template_invert.txt", 'r', encoding='utf-8') as file:
    text_qa_template_invert_str = file.read()
text_qa_template_invert = PromptTemplate(text_qa_template_invert_str)


retriever = index.as_retriever(similarity_top_k=3)
query_engine_custom_naive = CustomPosRewritingQueryEngine.from_args(retriever=retriever,
                                                              llm = llm,
                                                              text_qa_template=text_qa_template,
                                                              verbose=True) 

query_engine_custom_1 = CustomPosRewritingQueryEngine.from_args(retriever=retriever,
                                                              llm = llm,
                                                              text_qa_template=text_qa_template,
                                                              pos_rewriting_threshold=0.1,
                                                              verbose=True) 

query_engine_custom_5 = CustomPosRewritingQueryEngine.from_args(retriever=retriever,
                                                              llm = llm,
                                                              text_qa_template=text_qa_template,
                                                              pos_rewriting_threshold=0.5,
                                                              verbose=True) 

query_engine_custom_8 = CustomPosRewritingQueryEngine.from_args(retriever=retriever,
                                                              llm = llm,
                                                              text_qa_template=text_qa_template,
                                                              pos_rewriting_threshold=0.8,
                                                              verbose=True) 

query_engine_custom_translate_query = CustomPosRewritingQueryEngine.from_args(retriever=retriever,
                                                              llm = llm,
                                                              text_qa_template=text_qa_template,
                                                              translate_query=True,
                                                              verbose=True) 

query_engine_custom_translate_context = CustomPosRewritingQueryEngine.from_args(retriever=retriever,
                                                              llm = llm,
                                                              text_qa_template=text_qa_template,
                                                              translate_context=True,
                                                              verbose=True) 

query_engine_custom_translate_all = CustomPosRewritingQueryEngine.from_args(retriever=retriever,
                                                              llm = llm,
                                                              text_qa_template=text_qa_template_ptbr,
                                                              translate_query=True,
                                                              translate_context=True,
                                                              verbose=True) 

query_engine_custom_invert = CustomPosRewritingQueryEngine.from_args(retriever=retriever,
                                                              llm = llm,
                                                              text_qa_template=text_qa_template_invert,
                                                              verbose=True) 

# run experiments -------------------------------------------------------------------------------------------------------
# dictionary of experiments, now referencing the predefined query engine objects
experiments = {
    "RAG (baseline)": (query_engine_custom_naive, "0"),
    "RAG (variations prompt 10%)": (query_engine_custom_1, "1.0"),
    "RAG (variations prompt 50%)": (query_engine_custom_5, "4.09"),
    "RAG (variations prompt 80%)": (query_engine_custom_8, "6.51"),
    "RAG (query translated pt-br)": (query_engine_custom_translate_query, "0"),
    "RAG (context translated pt-br)": (query_engine_custom_translate_context, "0"),
    "RAG (all translated pt-br)": (query_engine_custom_translate_all, "0"),
    "RAG (prompt invert pt-br)": (query_engine_custom_invert, "0"),
}

# Initialize an empty DataFrame to collect results from all experiments
all_experiments_results_df = pd.DataFrame(columns=['Run', 'Experiment', 'OverallScores'])

# https://docs.tonic.ai/validate/about-rag-metrics/tonic-validate-rag-metrics-reference
scorer = ValidateScorer(metrics=[RetrievalPrecisionMetric(), 
#                                 AugmentationAccuracyMetric(),
#                                 AugmentationPrecisionMetric(),
                                 AnswerSimilarityMetric()],
                                 model_evaluator=MODEL)

# Loop through each experiment configuration, run it, and collect results
for experiment_name, (query_engine, rate_replaced_words) in experiments.items():
    experiment_results_df = run_experiment(experiment_name,
                                            query_engine,
                                            rate_replaced_words,
                                            scorer,
                                            benchmark,
                                            validate_api,
                                            tonic_validate_project_key,
                                            upload_results=True,
                                            runs=1)  # Adjust the number of runs as needed

    # Append the results of this experiment to the master DataFrame
    all_experiments_results_df = pd.concat([all_experiments_results_df, experiment_results_df], ignore_index=True)

# Assuming all_experiments_results_df is your DataFrame
all_experiments_results_df['RetrievalPrecision'] = all_experiments_results_df['OverallScores'].apply(lambda x: x['retrieval_precision'])
