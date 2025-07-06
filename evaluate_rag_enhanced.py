"""
Enhanced RAG Evaluation using DeepEval + RAGAS with FastAPI Integration
Based on evaluate_rag.py with minimal changes to add RAGAS metrics and winner selection
"""
# import time
# time.sleep(60*90)  # Delay to ensure all imports are loaded correctly
# ---------------------------------------------------------------------
import requests
import json
import warnings
import os
import pandas as pd
import csv
import ast
from typing import Dict, List
from dotenv import load_dotenv
import numpy as np
from collections.abc import Mapping
from tenacity import retry, wait_exponential, stop_after_attempt

# Updated imports to avoid deprecation warnings
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    try:
        from langchain_community.chat_models import ChatOpenAI
        from langchain_community.embeddings import OpenAIEmbeddings
    except ImportError:
        # Fallback to deprecated imports
        from langchain.chat_models import ChatOpenAI
        from langchain.embeddings import OpenAIEmbeddings

from deepeval import evaluate as deepeval_evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric,
    AnswerRelevancyMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_core.language_models import BaseLanguageModel

# RAGAS imports
from ragas import evaluate, SingleTurnSample, EvaluationDataset
from ragas.metrics import (
    ResponseRelevancy,
    Faithfulness,
    LLMContextPrecisionWithoutReference,
)

# Import winner selection logic
import winner

import asyncio

warnings.filterwarnings("ignore")
load_dotenv()

# Configuration
API_URL = "http://localhost:8000/query"
API_KEY = "sk-iVwkyCuP5miUkRQdmkEfTw"
API_BASE = "https://litellm.sandbox.dge.gov.ae/v1"

EMBED_MODEL     = os.getenv("EMBED_MODEL")
MODEL_SUFFIX = EMBED_MODEL.replace("/", "_").replace("-", "_")

# Models to test (4 models as specified)
TEST_MODELS = [
    'deepseek',    # Will map to azure_ai/deepseek-v3-0324
    # 'gpt-4.1',      # Will map to gpt-4o
    # 'cohere',      # Will map to azure_ai/cohere-command-a  
    # 'qwen'         # Will map to ollama/qwen3:14b
]

# Model mapping for API calls
MODEL_MAPPING = {
    'deepseek': 'azure_ai/deepseek-v3-0324',
    'gpt-4.1': 'gpt-4.1',
    'cohere': 'azure_ai/cohere-command-a',
    'qwen': 'ollama/qwen3:14b'
}

class LangchainLLMWrapper(DeepEvalBaseLLM):
    """Wrapper for LangChain LLM to work with DeepEval"""
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm

    def generate(self, prompt: str) -> str:
        return self.llm.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return await asyncio.to_thread(self.generate, prompt)

    def get_model_name(self) -> str:
        return getattr(self.llm, 'model_name', 'gpt-4o')

    def load_model(self):
        pass

def get_rag_response(query: str, model: str = None) -> Dict:
    """
    Get response from FastAPI RAG server (enhanced for model mapping)
    
    Args:
        query: Question to ask
        model: Model to use for answer generation
        
    Returns:
        Dict with 'answer' and 'context' keys
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "query": query,
        "mode": "hybrid",
        "top_k": 10,
        "only_need_context": False,
    }
    
    # Set model-specific parameters (similar to run_deep_eval 1.py)
    if model == 'cohere':
        payload["llm_model_name"] = "azure_ai/cohere-command-a"
    elif model == 'qwen':
        payload["llm_model_name"] = "ollama/qwen3:14b"
    elif model == 'deepseek':
        payload["llm_model_name"] = "azure_ai/deepseek-v3-0324"
    elif model != 'deepseek':  # gpt-4o and others
        payload["llm_model_name"] = MODEL_MAPPING.get(model, model)
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=3000)  # 5 min timeout
        response.raise_for_status()
        data = response.json()
        
        return {
            "answer": data.get("answer", ""),
            "context": data.get("context", "")
        }
    except Exception as e:
        print(f"❌ Error calling RAG API for query '{query}' with model '{model}': {e}")
        return {"answer": "", "context": ""}

@retry(wait=wait_exponential(multiplier=1, min=2, max=3000), stop=stop_after_attempt(10))
def evaluate_with_deepeval(query: str, answer: str, context: str, truth: str) -> Dict:
    """
    Evaluate a single query using DeepEval metrics
    
    Args:
        query: The question
        answer: The RAG response
        context: The retrieved context
        
    Returns:
        Dict with metric scores
    """
    # Initialize LLM for evaluation (gpt-4o only as specified)
    llm = ChatOpenAI(
        model_name="gpt-4.1",
        openai_api_key=API_KEY,
        openai_api_base=API_BASE,
        temperature=0.3,
        request_timeout=600,  # 10 minutes
    )
    model = LangchainLLMWrapper(llm=llm)
    
    # Create test case
    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        expected_output=truth,
        context=[context],
        retrieval_context=[context],
    )
    
    # Define metrics
    metrics = [
        AnswerRelevancyMetric(model=model),
        FaithfulnessMetric(model=model),
        HallucinationMetric(model=model),
        BiasMetric(model=model),
        ToxicityMetric(model=model),
    ]
    
    try:
        # Run evaluation
        result = deepeval_evaluate([test_case], metrics=metrics)
        
        # Extract scores
        scores = {}
        for key, value in result:
            if key == "test_results":
                test_results = value
                break
                
        for result in test_results:
            for metric in result.metrics_data:
                scores[metric.name] = metric.score
                
        return scores
        
    except Exception as e:
        print(f"❌ DeepEval error: {e}")
        return {
            "Answer Relevancy": 0.0,
            "Faithfulness": 0.0,
            "Hallucination": 0.0,
            "Bias": 0.0,
            "Toxicity": 0.0
        }

def evaluate_with_ragas(query: str, answer: str, context: str, truth: str) -> Dict:
    """
    Evaluate using RAGAS metrics 
    Args:
        query: Input query
        answer: Model response  
        context: Retrieved context
        
    Returns:
        Dictionary of metric scores
    """
    try:
        # Initialize LLM and embeddings (gpt-4o only as specified)
        llm = ChatOpenAI(
            model_name="gpt-4.1",
            openai_api_key=API_KEY,
            openai_api_base=API_BASE,
            temperature=0.7,
            request_timeout=600,  # 10 minutes
        )
        
        embeddings = OpenAIEmbeddings(
            # model="text-embedding-3-large",
            model="azure_ai/embed-v-4-0",
            openai_api_key=API_KEY,
            openai_api_base=API_BASE,
            timeout=300,  # 5 minutes
        )
        
        # Create RAGAS sample and dataset
        sample = SingleTurnSample(
            user_input=query,
            response=answer,
            reference=truth,
            retrieved_contexts=[context],
        )
        dataset = EvaluationDataset(samples=[sample])
        
        # Define metrics
        metrics = [
            ResponseRelevancy(),
            Faithfulness(),
            LLMContextPrecisionWithoutReference(),
        ]
        
        # Run evaluation
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            show_progress=False,
        )
        
        return result.scores[0] if result.scores else {}
        
    except Exception as e:
        print(f"❌ RAGAS error: {e}")
        return {
            "answer_relevancy": 0.0,
            "faithfulness": 0.0,
            "llm_context_precision_without_reference": 0.0
        }

def round_numeric_values(d: Mapping, decimals: int = 2) -> dict:
    return {
        k: round(float(v), decimals) if isinstance(v, (float, np.floating)) else v
        for k, v in d.items()
    }

def load_existing_results(base_filename: str = "enhanced_rag_evaluation_results") -> pd.DataFrame:
    """
    Load existing evaluation results from cache files to avoid re-processing
    
    Args:
        base_filename: Base filename for cache files
        
    Returns:
        DataFrame with existing results or empty DataFrame if no cache found
    """
    excel_file = f"{base_filename}_{MODEL_SUFFIX}.xlsx"
    csv_file = f"{base_filename}_{MODEL_SUFFIX}.csv"
    
    # Try to load from Excel first, then CSV
    for file_path in [excel_file, csv_file]:
        if os.path.exists(file_path):
            try:
                if file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                else:
                    df = pd.read_csv(file_path)
                print(f"📥 Loaded {len(df)} existing results from {file_path}")
                return df
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")
                continue
    
    print("📝 No existing results found, starting fresh")
    return pd.DataFrame()

def is_query_already_evaluated(query: str, models: List[str], existing_results: pd.DataFrame) -> bool:
    """
    Check if a query has already been evaluated for all specified models with strict validation
    
    Args:
        query: The query to check
        models: List of models to test
        existing_results: DataFrame with existing results
        
    Returns:
        True if query is already fully evaluated, False otherwise
    """
    if existing_results.empty:
        print(f"  🔍 No existing results found")
        return False
    
    # Find matching query
    matching_rows = existing_results[existing_results['Query'] == query]
    if matching_rows.empty:
        print(f"  🔍 Query not found in cache")
        return False
    
    row = matching_rows.iloc[0]
    print(f"  🔍 Found cached query, validating completeness...")
    
    # Check if all models have been evaluated with strict validation
    for model in models:
        response_col = f"{model}_Response"
        scores_col = f"{model}_scores"
        
        # Check if response exists and is meaningful
        if response_col not in row or pd.isna(row[response_col]) or str(row[response_col]).strip() == "":
            print(f"    ❌ Missing or empty response for {model}")
            return False
        
        # Check if response looks like an error
        response_str = str(row[response_col]).lower()
        if any(error_word in response_str for error_word in ['error', 'failed', 'exception', 'timeout']):
            print(f"    ❌ Response contains error indicators for {model}")
            return False
        
        # Check if scores exist and are meaningful
        if scores_col not in row or pd.isna(row[scores_col]) or str(row[scores_col]).strip() == "":
            print(f"    ❌ Missing or empty scores for {model}")
            return False
        
        # Try to validate score content
        try:
            scores_str = str(row[scores_col])
            # Check if it looks like a dictionary or contains numeric values
            if not any(char.isdigit() for char in scores_str):
                print(f"    ❌ Scores don't contain numeric values for {model}")
                return False
            
            # If it's a string representation of a dict, try to parse it
            if scores_str.startswith('{') and scores_str.endswith('}'):
                import ast
                try:
                    parsed_scores = ast.literal_eval(scores_str)
                    if not isinstance(parsed_scores, dict) or len(parsed_scores) == 0:
                        print(f"    ❌ Invalid score format for {model}")
                        return False
                except:
                    print(f"    ❌ Cannot parse scores for {model}")
                    return False
        except Exception as e:
            print(f"    ❌ Error validating scores for {model}: {e}")
            return False
    
    print(f"  ✅ All models validated successfully - using cached result")
    return True

def save_single_result(query_result: dict, base_filename: str = "enhanced_rag_evaluation_results"):
    """
    Save a single query result incrementally to cache files
    
    Args:
        query_result: Dictionary containing the evaluation result for one query
        base_filename: Base filename for cache files
    """
    # Load existing results
    existing_df = load_existing_results(base_filename)
    
    # Create new result DataFrame
    new_result_df = pd.DataFrame([query_result])
    
    # Combine with existing results
    if not existing_df.empty:
        # Remove any existing row with the same query
        existing_df = existing_df[existing_df['Query'] != query_result['Query']]
        combined_df = pd.concat([existing_df, new_result_df], ignore_index=True)
    else:
        combined_df = new_result_df
    
    # Save to files
    save_results_enhanced(combined_df, base_filename)

def process_excel_questions_enhanced(excel_file_path: str, models: List[str] = None, force_reeval: bool = False) -> pd.DataFrame:
    """
    Process questions from Excel file with enhanced evaluation (DeepEval + RAGAS + Winner)
    
    Args:
        excel_file_path: Path to Excel file with questions
        models: List of models to test (defaults to TEST_MODELS)
        force_reeval: If True, bypass cache and re-evaluate all queries
        
    Returns:
        DataFrame with evaluation results in queries_evaluation_results.csv format
    """
    if models is None:
        models = TEST_MODELS
    
    # Load existing results for caching (unless forced re-evaluation)
    existing_results = pd.DataFrame() if force_reeval else load_existing_results()
    if force_reeval:
        print("🔄 Force re-evaluation enabled - bypassing cache")
    
    # Read Excel file
    try:
        if excel_file_path.endswith('.csv'):
            df = pd.read_csv(excel_file_path)
        else:
            df = pd.read_excel(excel_file_path)
            
        # Assume first column contains questions
        questions = df.iloc[:, 0].tolist()
        ground_truth = df.iloc[:, 1].tolist() 
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return pd.DataFrame()
    
    print(f"📋 Processing {len(questions)} questions with {len(models)} models...")
    print(f"🤖 Models: {', '.join(models)}")
    
    # Process each query
    results = []
    processed_count = 0
    skipped_count = 0
    
    for i, (query,gt) in enumerate(zip(questions,ground_truth), 1):
        if pd.isna(query) or not str(query).strip():
            continue
            
        query = str(query).strip()
        print(f"\n[{i}/{len(questions)}] 🔍 Processing: {query[:60]}...")
        
        # Check if query is already evaluated (unless force_reeval is True)
        if not force_reeval and is_query_already_evaluated(query, models, existing_results):
            print(f"  ✅ Query already evaluated, skipping...")
            # Add existing result to results list
            existing_row = existing_results[existing_results['Query'] == query].iloc[0]
            results.append(existing_row.to_dict())
            skipped_count += 1
            continue
        
        # Initialize query result with format matching queries_evaluation_results.csv
        query_result = {
            "Query": query,
            "GroundTruth": gt if pd.notna(gt) else "",  # Use ground truth if available
            "RAG": "Contextual RAG",  # Hardcoded as specified
            "Context": ""  # Will be filled from first successful response
        }
        
        model_scores = {}
        
        # Test each model for this question
        for model in models:
            print(f"  🤖 Testing with {model}...")
            
            # Get RAG response
            rag_response = get_rag_response(query, model)
            answer = rag_response["answer"]
            context = rag_response["context"]
            
            if not answer:
                print(f"    ❌ No answer received for {model}")
                continue
                
            print(f"    ✅ Got answer: {answer[:50]}...")
            
            # Store context (use first successful one)
            if not query_result["Context"] and context:
                query_result["Context"] = context
            
            # Store response
            query_result[f"{model}_Response"] = answer
            
            # Evaluate with DeepEval
            deepeval_scores = evaluate_with_deepeval(query, answer, context, gt)
            print(f"    📊 DeepEval completed for {model}")
            
            # Evaluate with RAGAS  
            # ragas_scores = evaluate_with_ragas(query, answer, context, gt)
            # print(f"    📊 RAGAS completed for {model}")
            
            # Combine scores with prefixes (similar to run_deep_eval 1.py)
            combined_scores = {}
            for metric, score in deepeval_scores.items():
                combined_scores[f"deep_eval_{metric}"] = score
            
            # for metric, score in ragas_scores.items():
                # combined_scores[f"ragas_{metric}"] = score
            
            # Store scores
            query_result[f"{model}_scores"] = combined_scores
            model_scores[model] = combined_scores
        
        # Determine winner using existing winner.py logic
        if len(model_scores) >= 2:
            try:
                winner_model, aggregate_scores = winner.decide_winner(model_scores)
                query_result['Winner'] = winner_model
                # aggregate_scores = {k: round(float(v), 2) for k, v in aggregate_scores.items()}
                aggregate_scores = round_numeric_values(aggregate_scores)
                query_result['Scores'] = aggregate_scores
                print(f"    🏆 Winner: {winner_model}")
            except Exception as e:
                print(f"    ❌ Winner selection error: {e}")
                query_result['Winner'] = "Undetermined"
                query_result['Scores'] = {}
        else:
            query_result['Winner'] = "Insufficient_Data"
            query_result['Scores'] = {}
        
        results.append(query_result)
        processed_count += 1
        
        # Save result incrementally
        save_single_result(query_result)
        print(f"  💾 Result saved incrementally")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    print(f"\n✅ Processed {processed_count} new evaluations, skipped {skipped_count} cached evaluations")
    print(f"📊 Total results: {len(results)}")
    
    return results_df

def save_results_enhanced(results_df: pd.DataFrame, base_filename: str = "enhanced_rag_evaluation_results"):
    """Save results to both Excel (primary) and CSV (backup) formats"""
    
    if results_df.empty:
        print("❌ No results to save")
        return
    
    excel_file = f"{base_filename}_{MODEL_SUFFIX}.xlsx"
    csv_file = f"{base_filename}_{MODEL_SUFFIX}.csv"
    
    try:
        # Primary: Save to Excel
        results_df.to_excel(excel_file, index=False)
        print(f"💾 Results saved to Excel: {excel_file}")
        
        # Backup: Save to CSV
        results_df.to_csv(
            csv_file,
            index=False,
            sep=",",
            quoting=csv.QUOTE_ALL,
            quotechar='"',
            doublequote=True,
            escapechar=None,
            lineterminator="\n",
            encoding="utf-8",
        )
        print(f"💾 Results saved to CSV: {csv_file}")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        # Fallback to CSV only
        try:
            results_df.to_csv(csv_file, index=False)
            print(f"💾 Fallback: Results saved to CSV: {csv_file}")
        except Exception as e2:
            print(f"❌ Fallback save failed: {e2}")

if __name__ == "__main__":
    # Example usage with sample questions
    # test_questions = [
    #     # "How national, sector and entity context are interlinked?",
    #     # "What is the role of NESA in national coordination?",
    #     # "Outline NESA's compliance monitoring and audit processes. How can this be improved?",
    #     # "What is Risk Analysis under M.*?",
    #     # "The UAE IA Standards promotes a lifecycle approach for establishing, implementing, maintaining and continuously improving information assurance."
    #     "What are the main components of the Procurement Capability Model?"
    # ]
    
    # # Create a simple DataFrame for testing
    # test_df = pd.DataFrame({"Query": test_questions,"Ground_Truth":"Strategic \n Tactical \n Operational \n Support"})
    # test_df.to_excel("sample_questions_enhanced.xlsx", index=False)
    
    # Process the questions with enhanced evaluation
    print("🚀 Starting enhanced evaluation (DeepEval + RAGAS + Winner Selection)...")
    results = process_excel_questions_enhanced("sample_questions.xlsx")
    
    if not results.empty:
        save_results_enhanced(results)
        print("\n📊 Sample Results:")
        print(results[["Query", "Winner"]].head())
        
        # Show winner distribution
        if 'Winner' in results.columns:
            print("\n🏆 Winner Distribution:")
            winner_counts = results['Winner'].value_counts()
            for model, count in winner_counts.items():
                print(f"  {model}: {count} wins")
    else:
        print("❌ No results generated")