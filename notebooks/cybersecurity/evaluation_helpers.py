import logging

from langfuse import Langfuse
import dspy
import matplotlib.pyplot as plt
import numpy as np

from langfuse_extensions import EvaluateWithLangfuse

langfuse = Langfuse()


def validate_answer(example, pred, trace=None):
    return example.category == pred.category.lower()


def fetch_traces(run_id, limit=100):
    traces = langfuse.fetch_traces(session_id=run_id, limit=limit)
    results = []
    score_client = langfuse.client.score
    for t in traces.data:
        scores = score_client.get(score_ids=",".join(t.scores))
        res = dict()
        for s in scores.data:
            res[s.name] = s.value if s.data_type=="NUMERIC" else s.string_value
        results.append(res)
    return results


def calculate_metrics(results, classes):
    metrics = dict()
    for c in classes:
        metrics[c] = dict()
        metrics[c]["tp"] = 0
        metrics[c]["fp"] = 0
        metrics[c]["fn"] = 0
        metrics[c]["tn"] = 0
        metrics[c]["precision"] = 0
        metrics[c]["recall"] = 0
        metrics[c]["f1"] = 0
    for r in results:
        for c in classes:
            if r["ground_truth"] == c:
                if r["prediction"] == c:
                    metrics[c]["tp"] += 1
                else:
                    metrics[c]["fn"] += 1
            else:
                if r["prediction"] == c:
                    metrics[c]["fp"] += 1
                else:
                    metrics[c]["tn"] += 1
    for c in classes:
        tp = metrics[c]["tp"]
        fp = metrics[c]["fp"]
        fn = metrics[c]["fn"]
        tn = metrics[c]["tn"]
        metrics[c]["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics[c]["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics[c]["f1"] = 2 * (metrics[c]["precision"] * metrics[c]["recall"]) / (
                metrics[c]["precision"] + metrics[c]["recall"]) if (metrics[c]["precision"] + metrics[c][
            "recall"]) > 0 else 0
    metrics["macro"] = dict()
    metrics["macro"]["precision"] = sum([metrics[c]["precision"] for c in classes]) / len(classes)
    metrics["macro"]["recall"] = sum([metrics[c]["recall"] for c in classes]) / len(classes)
    metrics["macro"]["f1"] = sum([metrics[c]["f1"] for c in classes]) / len(classes)
    return metrics




def plot_metrics(metrics, labels, title):
    x = np.arange(len(labels))  # the label locations
    width = 0.05  # the width of the bars
    fig, ax = plt.subplots()
    k = 0
    for m in metrics:
        ax.bar(
            x + width * k,
            [metrics[m]["precision"], metrics[m]["recall"], metrics[m]["f1"]],
            width,
            label=m,
            )
        k += 1
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Scores")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=4)
    ax.grid(True)
    fig.set_size_inches(12, 5)
    plt.show()


def run_evaluation(models, program , test_ds, classes,  timestamp, prefix="baseline", cache=False):
    for model in models:
        logging.info(f"Evaluating {model}")
        lm=dspy.LM(f"ollama/{model}", cache=cache)
        dspy.settings.configure(lm=lm, track_usage=False)
        session_id = f"{prefix}-{model}-{timestamp}"
        evaluator = EvaluateWithLangfuse(
            devset=test_ds,
            num_threads=1,
            display_progress=True,
            session_id=session_id,
            lm=lm,
            max_errors=100,
            provide_traceback=True,
        )

        dspy.configure(callbacks=[evaluator])
        evaluator(program=program, metric=validate_answer)

def get_all_metric(models, timestamp, classes, prefix="baseline"):
    metrics = dict()
    for model in models:
        traces = fetch_traces(run_id=f"{prefix}-{model}-{timestamp}")
        metrics[model] = calculate_metrics(traces, classes)["macro"]
    return metrics