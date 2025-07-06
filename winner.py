from typing import Dict, List, Tuple

# ---------------------------------------------------------------------
# CONFIGURATION – edit weights / directions if your scoring scheme differs
# ---------------------------------------------------------------------
#   key                       +1 → higher is better,  -1 → lower is better
METRIC_DIRECTION = {
    "deep_eval_Answer Relevancy":  1,
    "deep_eval_Faithfulness":     1,
    "deep_eval_Hallucination":   -1,
    "deep_eval_Bias":            -1,
    "deep_eval_Toxicity":        -1,
    # "ragas_answer_relevancy":  1,  # add if you enable it
    # "ragas_faithfulness":       1,  # add if you enable it
    # "ragas_llm_context_precision_without_reference": 1,  # add if you enable it
    # "ContextualPrecisionMetric": 1,  # add if you enable it
}

# Optional importance weighting (default = 1.0 everywhere)
# METRIC_WEIGHT = {
#     "deep_eval_Answer Relevancy":  1.0,
#     "deep_eval_Faithfulness":     1.0,
#     "deep_eval_Hallucination":    1.0,
#     "deep_eval_Bias":             0.5,   # example: make bias & toxicity half weight
#     "deep_eval_Toxicity":         0.5,
#     "ragas_answer_relevancy":  1.0,  # add if you enable it
#     "ragas_faithfulness":       1.0,  # add if
#     "ragas_llm_context_precision_without_reference": 1.0,  # add if you enable it
# }
METRIC_WEIGHT = {
    "deep_eval_Answer Relevancy": 0.5,
    "deep_eval_Faithfulness": 0.5,
    "deep_eval_Hallucination": 1.0, 
    "deep_eval_Bias": 0.5,   # example: make bias & toxicity half weight
    "deep_eval_Toxicity": 0.5,
    # "ragas_answer_relevancy": 0.5,  # add if you enable it
    # "ragas_faithfulness": 0.5,  # add if
    # "ragas_llm_context_precision_without_reference": 1.0,  # add if you enable it
}
# ---------------------------------------------------------------------
def decide_winner(
    results: Dict[str, Dict[str, float]],
) -> Tuple[str, Dict[str, float]]:
    """
    Decide which system wins based on a dictionary of metric scores.
    
    Parameters
    ----------
    results : dict
        {
          "lightrag":   {"Answer Relevancy": 0.86, "Faithfulness": 0.83, ...},
          "llamaindex": {"Answer Relevancy": 0.81, "Faithfulness": 0.88, ...}
        }
        *Higher* is assumed better unless the metric is listed in
        METRIC_DIRECTION with direction = -1.
    
    Returns
    -------
    winner : str          # model name with the highest weighted score
    scores : dict         # aggregate scores per model so you can inspect
    """
    # --- sanity-check -------------------------------------------------
    missing = set(METRIC_DIRECTION) - {
        m for sys in results.values() for m in sys.keys()
    }
    if missing:
        raise ValueError(f"Missing metric(s) in input: {', '.join(missing)}")

    # --- aggregate ----------------------------------------------------
    agg: Dict[str, float] = {sys: 0.0 for sys in results}

    for metric, direction in METRIC_DIRECTION.items():
        weight = METRIC_WEIGHT.get(metric, 1.0)
        for system, metrics in results.items():
            score = metrics[metric]
            agg[system] += direction * weight * score

    # --- decide -------------------------------------------------------
    # winner = max(agg, key=agg.get)
    top_2_winners = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:2]
    top_2_winners = ", ".join(name for name, _ in top_2_winners)
    
    return top_2_winners, agg

# ---------------------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # fake numbers just for illustration
    evaluation_results = {
        "deepseek": {'deep_eval_Answer Relevancy': 1.0, 'deep_eval_Faithfulness': 0.8333333333333334, 'deep_eval_Hallucination': 0.0, 'deep_eval_Bias': 0.0, 'deep_eval_Toxicity': 0.0},
        "gpt4.1":{'deep_eval_Answer Relevancy': 1.0, 'deep_eval_Faithfulness': 0.5, 'deep_eval_Hallucination': 0.0, 'deep_eval_Bias': 0.0, 'deep_eval_Toxicity': 0.0},
        "cohere":{'deep_eval_Answer Relevancy': 1.0, 'deep_eval_Faithfulness': 0.6666666666666666, 'deep_eval_Hallucination': 0.0, 'deep_eval_Bias': 0.0, 'deep_eval_Toxicity': 0.0},
    }

    champion, totals = decide_winner(evaluation_results)
    print("Winner:", champion)
    print("Aggregate scores:", totals)
