import types
from datetime import datetime, timedelta
from typing import Optional, Callable, List, Union, Any
import logging
from dspy import Evaluate
from langfuse import Langfuse

from langfuse.model import ModelUsage
import tqdm

import dspy
from dspy.utils.callback import with_callbacks
from dspy.utils.parallelizer import ParallelExecutor

try:
    from IPython.display import HTML
    from IPython.display import display as display

except ImportError:

    def display(obj: Any):
        """
        Display the specified Python object in the console.

        :param obj: The Python object to display.
        """
        print(obj)

    def HTML(x: str) -> str:
        """
        Obtain the HTML representation of the specified string.
        """
        # NB: This method exists purely for code compatibility with the IPython HTML() function in
        # environments where IPython is not available. In such environments where IPython is not
        # available, this method will simply return the input string.
        return x


# TODO: Counting failures and having a max_failure count. When that is exceeded (also just at the end),
# we print the number of failures, the first N examples that failed, and the first N exceptions raised.

logger = logging.getLogger(__name__)

import types
from datetime import datetime, timedelta
from typing import Optional, Callable, List, Union, Any, Dict
import logging
from dspy import Evaluate
from langfuse import Langfuse

from langfuse.model import ModelUsage
import tqdm

import dspy
from dspy.utils.callback import with_callbacks, BaseCallback
from dspy.utils.parallelizer import ParallelExecutor

try:
    from IPython.display import HTML
    from IPython.display import display as display

except ImportError:

    def display(obj: Any):
        """
        Display the specified Python object in the console.

        :param obj: The Python object to display.
        """
        print(obj)

    def HTML(x: str) -> str:
        """
        Obtain the HTML representation of the specified string.
        """
        # NB: This method exists purely for code compatibility with the IPython HTML() function in
        # environments where IPython is not available. In such environments where IPython is not
        # available, this method will simply return the input string.
        return x


# TODO: Counting failures and having a max_failure count. When that is exceeded (also just at the end),
# we print the number of failures, the first N examples that failed, and the first N exceptions raised.

logger = logging.getLogger(__name__)

class EvaluateWithLangfuse(Evaluate, BaseCallback):

    def __init__(self, session_id, lm, **kwargs):
        self.session_id = session_id
        self.langfuse = Langfuse()
        self.gen_starts = {}
        self.gen_ends = {}
        self.lm = lm
        super().__init__(**kwargs)

    def langfuse_trace(self, i, o, session, score, ground_truth, pred, prediction_start, prediction_end, **kwargs):
        model = o['model'] if 'model' in o else None
        name = f"Evaluation-{model}"
        trace_client = self.langfuse.trace(session_id=session, input=i, output=o, name=name, metadata=kwargs)
        # Unpack args if they are being used to pass i, o, etc.
        output = o['choices'][0]['message']['content'] if 'choices' in o else None
        prompt_tokens = o['usage']['prompt_tokens'] if 'usage' in o else None
        completion_tokens = o['usage']['completion_tokens'] if 'usage' in o else None

        model_parameters = self.lm.kwargs
        usage = ModelUsage(
            input=prompt_tokens,
            output=completion_tokens
        )
        generation_client = self.langfuse.generation(
            trace_id=trace_client.id,
            start_time=prediction_start,
            end_time=prediction_end,
            input=i, output=output, name=name, model=model, usage=usage, model_parameters=model_parameters
        )
        self.langfuse.score(
            trace_id=trace_client.id,
            observation_id=generation_client.id,  # optional
            name="model",
            value=model
        )
        self.langfuse.score(
            trace_id=trace_client.id,
            observation_id=generation_client.id,  # optional
            name="accuracy",
            value=score,
            data_type="NUMERIC",  # optional, inferred if not provided
        )
        if ground_truth:
            self.langfuse.score(
                trace_id=trace_client.id,
                observation_id=generation_client.id,  # optional
                name="ground_truth",
                value=ground_truth
            )
        if pred:
            self.langfuse.score(
                trace_id=trace_client.id,
                observation_id=generation_client.id,  # optional
                name="prediction",
                value=pred
            )

    def __call__(
            self,
            program: "dspy.Module",
            metric: Optional[Callable] = None,
            devset: Optional[List["dspy.Example"]] = None,
            num_threads: Optional[int] = None,
            display_progress: Optional[bool] = None,
            display_table: Optional[Union[bool, int]] = None,
            return_all_scores: Optional[bool] = None,
            return_outputs: Optional[bool] = None,
            callback_metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Args:
            program (dspy.Module): The DSPy program to evaluate.
            metric (Callable): The metric function to use for evaluation. if not provided, use `self.metric`.
            devset (List[dspy.Example]): the evaluation dataset. if not provided, use `self.devset`.
            num_threads (int): The number of threads to use for parallel evaluation. if not provided, use
                `self.num_threads`.
            display_progress (bool): Whether to display progress during evaluation. if not provided, use
                `self.display_progress`.
            display_table (Union[bool, int]): Whether to display the evaluation results in a table. if not provided, use
                `self.display_table`. If a number is passed, the evaluation results will be truncated to that number before displayed.
            return_all_scores (bool): Whether to return scores for every data record in `devset`. if not provided,
                use `self.return_all_scores`.
            return_outputs (bool): Whether to return the dspy program's outputs for every data in `devset`. if not
                provided, use `self.return_outputs`.
            callback_metadata (dict): Metadata to be used for evaluate callback handlers.

        Returns:
            The evaluation results are returned in different formats based on the flags:

            - Base return: A float percentage score (e.g., 67.30) representing overall performance

            - With `return_all_scores=True`:
                Returns (overall_score, individual_scores) where individual_scores is a list of
                float scores for each example in devset

            - With `return_outputs=True`:
                Returns (overall_score, result_triples) where result_triples is a list of
                (example, prediction, score) tuples for each example in devset

            - With both flags=True:
                Returns (overall_score, result_triples, individual_scores)

        """
        metric = metric if metric is not None else self.metric
        devset = devset if devset is not None else self.devset
        num_threads = num_threads if num_threads is not None else self.num_threads
        display_progress = display_progress if display_progress is not None else self.display_progress
        display_table = display_table if display_table is not None else self.display_table
        return_all_scores = return_all_scores if return_all_scores is not None else self.return_all_scores
        return_outputs = return_outputs if return_outputs is not None else self.return_outputs

        if callback_metadata:
            logger.debug(f"Evaluate is called with callback metadata: {callback_metadata}")

        tqdm.tqdm._instances.clear()

        executor = ParallelExecutor(
            num_threads=num_threads,
            disable_progress_bar=not display_progress,
            max_errors=self.max_errors,
            provide_traceback=self.provide_traceback,
            compare_results=True,
        )


        def process_item(example):
            prediction_start = datetime.now()
            prediction = program(**example.inputs())
            prediction_end = datetime.now()
            score = metric(example, prediction, self.session_id)

            self.langfuse_trace(
                i = self.lm.history[-1]['messages'][0]['content'],
                o = self.lm.history[-1],
                session=self.session_id,
                score=1.0 if score else 0.0,
                ground_truth=example.category,
                pred=prediction.category, prediction_start=prediction_start, prediction_end=prediction_end)
            # Increment assert and suggest failures to program's attributes
            if hasattr(program, "_assert_failures"):
                program._assert_failures += dspy.settings.get("assert_failures")
            if hasattr(program, "_suggest_failures"):
                program._suggest_failures += dspy.settings.get("suggest_failures")

            return prediction, score

        results = executor.execute(process_item, devset)
        assert len(devset) == len(results)

        results = [((dspy.Prediction(), self.failure_score) if r is None else r) for r in results]
        results = [(example, prediction, score) for example, (prediction, score) in zip(devset, results)]
        ncorrect, ntotal = sum(score for *_, score in results), len(devset)

        logger.info(f"Average Metric: {ncorrect} / {ntotal} ({round(100 * ncorrect / ntotal, 1)}%)")

        # Rename the 'correct' column to the name of the metric object
        metric_name = metric.__name__ if isinstance(metric, types.FunctionType) else metric.__class__.__name__
        # Construct a pandas DataFrame from the results
        result_df = self._construct_result_table(results, metric_name)

        if display_table:
            self._display_result_table(result_df, display_table, metric_name)

        if return_all_scores and return_outputs:
            return round(100 * ncorrect / ntotal, 2), results, [score for *_, score in results]
        if return_all_scores:
            return round(100 * ncorrect / ntotal, 2), [score for *_, score in results]
        if return_outputs:
            return round(100 * ncorrect / ntotal, 2), results

        return round(100 * ncorrect / ntotal, 2)