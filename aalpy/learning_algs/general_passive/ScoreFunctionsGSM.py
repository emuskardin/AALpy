import warnings
from collections import defaultdict
from math import sqrt, log
from typing import Callable, Dict, Union, List, Iterable

from aalpy.learning_algs.general_passive.GsmNode import GsmNode

Score = Union[bool, float]
LocalCompatibilityFunction = Callable[[GsmNode, GsmNode], bool]
ScoreFunction = Callable[[Dict[GsmNode, GsmNode]], Score]
AggregationFunction = Callable[[Iterable[Score]], Score]


class ScoreCalculation:
    def __init__(self, local_compatibility: LocalCompatibilityFunction = None, score_function: ScoreFunction = None):
        # This is a hack that gives a simple implementation where we can easily - determine whether the default is
        # overridden (for optimization) - override behavior in a functional way by providing the functions as
        # arguments (no extra class) - override behavior in a stateful way by implementing a new class that provides
        # `local_compatibility` and / or `score_function` methods
        if not hasattr(self, "local_compatibility"):
            self.local_compatibility: LocalCompatibilityFunction = local_compatibility or self.default_local_compatibility
        if not hasattr(self, "score_function"):
            self.score_function: ScoreFunction = score_function or self.default_score_function

    def reset(self):
        pass

    @staticmethod
    def default_local_compatibility(a, b):
        return True

    @staticmethod
    def default_score_function(part):
        return True

    def has_score_function(self):
        return self.score_function is not self.default_score_function

    def has_local_compatibility(self):
        return self.local_compatibility is not self.default_local_compatibility


def hoeffding_compatibility(eps, compare_original=True) -> LocalCompatibilityFunction:
    eps_fact = sqrt(0.5 * log(2 / eps))
    count_name = "original_count" if compare_original else "count"

    def similar(a: GsmNode, b: GsmNode):
        for in_sym in filter(lambda x: x in a.transitions.keys(), b.transitions.keys()):
            # could create appropriate dict here
            a_trans, b_trans = (x.transitions[in_sym] for x in [a, b])
            a_total, b_total = (sum(getattr(x, count_name) for x in trans.values()) for trans in (a_trans, b_trans))
            if a_total == 0 or b_total == 0:
                continue
            threshold = eps_fact * (sqrt(1 / a_total) + sqrt(1 / b_total))
            for out_sym in set(a_trans.keys()).union(b_trans.keys()):
                ac, bc = (getattr(x[out_sym], count_name) if out_sym in x else 0 for x in (a_trans, b_trans))
                if abs(ac / a_total - bc / b_total) > threshold:
                    return False
        return True

    return similar


def non_det_compatibility(allow_subset=False) -> LocalCompatibilityFunction:
    def compatible(a: GsmNode, b: GsmNode):
        for in_sym, a_trans in a.transitions.items():
            b_trans = b.transitions.get(in_sym)
            if b_trans is None:
                continue
            c1 = allow_subset and any(out_sym not in a_trans for out_sym in b_trans.keys())
            c2 = not allow_subset and set(a_trans.keys()) != set(b_trans.keys())
            if c1 or c2:
                return False
        return True

    return compatible


class NoRareEventNonDetScore(ScoreCalculation):
    def __init__(self, thresh, p_min: Union[dict, float], reject_local_score_only=False, no_global_score=False):
        super().__init__()
        warnings.warn("Using experimental compatibility criterion for nondeterministic automata.")

        # Transform parameters to log space and create dict
        self.thresh = log(thresh)
        if isinstance(p_min, float):
            cost = log(1 - p_min)
            self.miss_dict = defaultdict(lambda: cost)
        else:
            self.miss_dict = {k: log(1 - v) for k, v in p_min.items()}

        self.score = 0
        self.reject_local_score_only = reject_local_score_only
        if no_global_score:
            self.global_score = self.default_score_function

    def reset(self):
        self.score = 0

    def local_compatibility(self, a: GsmNode, b: GsmNode):
        score_local = 0
        for in_sym in filter(lambda x: x in a.transitions.keys(), b.transitions.keys()):
            a_trans, b_trans = (x.transitions[in_sym] for x in [a, b])
            a_total, b_total = (sum(x.count for x in x.values()) for x in (a_trans, b_trans))
            for out_sym in set(a_trans.keys()).union(b_trans.keys()):
                a_miss, b_miss = (out_sym not in trans for trans in [a_trans, b_trans])
                if a_miss:
                    score_local += a_total * self.miss_dict[in_sym, out_sym]
                if b_miss:
                    score_local += b_total * self.miss_dict[in_sym, out_sym]

        self.score += score_local
        if self.reject_local_score_only:
            return self.thresh < score_local
        return self.thresh < self.score

    def score_function(self, part: Dict[GsmNode, GsmNode]) -> Score:
        # I don't think that we have to reevaluate on the full partition.
        return self.score


class ScoreWithKTail(ScoreCalculation):
    """Applies k-Tails to a compatibility function: Compatibility is only evaluated up to a certain depth k."""

    def __init__(self, other_score: ScoreCalculation, k: int):
        super().__init__(None, other_score.score_function)
        self.other_score = other_score
        self.k = k

        self.depth_offset = None

    def reset(self):
        self.other_score.reset()
        self.depth_offset = None

    def local_compatibility(self, a: GsmNode, b: GsmNode):
        # assuming b is tree shaped.
        if self.depth_offset is None:
            self.depth_offset = b.prefix_length
        depth = b.prefix_length - self.depth_offset
        if self.k <= depth:
            return True

        return self.other_score.local_compatibility(a, b)


class ScoreWithSinks(ScoreCalculation):
    def __init__(self, other_score: ScoreCalculation, sink_cond: Callable[[GsmNode], bool], allow_sink_merge=True):
        super().__init__(None, other_score.score_function)
        self.other_score = other_score
        self.sink_cond = sink_cond
        self.allow_sink_merge = allow_sink_merge

        self.is_first = True

    def reset(self):
        self.other_score.reset()
        self.is_first = True

    def local_compatibility(self, a: GsmNode, b: GsmNode):
        if self.is_first:
            self.is_first = False
            a_sink, b_sink = self.sink_cond(a), self.sink_cond(b)
            if a_sink != b_sink:
                return False
            if a_sink and b_sink and not self.allow_sink_merge:
                return False
        return self.other_score.local_compatibility(a, b)


class ScoreCombinator(ScoreCalculation):
    """
    This class is used to combine several scoring / compatibility mechanisms by aggregating the results of the
    individual methods in a user defined manner. It uses generator expressions to allow for short circuit evaluation.
    """

    def __init__(self, scores: List[ScoreCalculation], aggregate_compatibility: AggregationFunction = None,
                 aggregate_score: AggregationFunction = None):
        super().__init__()
        self.scores = scores
        self.aggregate_compatibility = aggregate_compatibility or self.default_aggregate_compatibility
        self.aggregate_score = aggregate_score or self.default_aggregate_score

    def reset(self):
        for score in self.scores:
            score.reset()

    def local_compatibility(self, a: GsmNode, b: GsmNode):
        return self.aggregate_compatibility(score.local_compatibility(a, b) for score in self.scores)

    def score_function(self, part: Dict[GsmNode, GsmNode]):
        return self.aggregate_score(score.score_function(part) for score in self.scores)

    @staticmethod
    def default_aggregate_compatibility(compatibility_iterable):
        """Commits to the first value that is not inconclusive (== None). Accepts if in doubt."""
        for compat in compatibility_iterable:
            if compat is None:
                continue
            return compat
        return True

    @staticmethod
    def default_aggregate_score(score_iterable):
        return list(score_iterable)


def local_to_global_score(local_fun: LocalCompatibilityFunction) -> ScoreFunction:
    """
    Converts a local score function to a global score function by evaluating the local compatibility for each of the new
    partitions with all nodes that make up that partition. One use case for this is to evaluate a local score function
    after the partitions are complete.
    """

    def fun(part: Dict[GsmNode, GsmNode]):
        for old_node, new_node in part.items():
            if local_fun(new_node, old_node) is False:
                return False
        return True

    return fun


def differential_info(part: Dict[GsmNode, GsmNode]):
    relevant_nodes_old = list(part.keys())
    relevant_nodes_new = set(part.values())

    partial_llh_old = sum(node.local_log_likelihood_contribution() for node in relevant_nodes_old)
    partial_llh_new = sum(node.local_log_likelihood_contribution() for node in relevant_nodes_new)

    num_params_old = sum(1 for node in relevant_nodes_old for _ in node.transition_iterator())
    num_params_new = sum(1 for node in relevant_nodes_new for _ in node.transition_iterator())

    return partial_llh_old - partial_llh_new, num_params_old - num_params_new


def transform_score(score, transform: Callable):
    if isinstance(score, Callable):
        return lambda *args: transform(score(*args))
    if isinstance(score, ScoreCalculation):
        score.score_function = lambda *args: transform(score.score_function(*args))
        return score
    return transform(score)


def make_greedy(score):
    return transform_score(score, lambda x: x is not False)


def lower_threshold(score, thresh):
    return transform_score(score, lambda x: x if thresh < x else False)


# def likelihood_ratio_score(alpha=0.05) -> ScoreFunction:
#     # TODO remove and add as an example?
#     from scipy.stats import chi2
#
#     if not 0 < alpha <= 1:
#         raise ValueError(f"Confidence {alpha} not between 0 and 1")
#
#     def score_fun(part: Dict[GsmNode, GsmNode]):
#         llh_diff, param_diff = differential_info(part)
#         if param_diff == 0:
#             # This should cover the corner case when the partition merges only states with no outgoing transitions.
#             return -1  # Let them be very bad merges.
#         score = 1 - chi2.cdf(2 * llh_diff, param_diff)
#         return lower_threshold(score, alpha)  # Not entirely sure if implemented correctly
#
#     return score_fun


def AIC_score(alpha=0) -> ScoreFunction:
    def score(part: Dict[GsmNode, GsmNode]):
        llh_diff, param_diff = differential_info(part)
        return lower_threshold(param_diff - llh_diff, alpha)

    return score


def EDSM_score(min_evidence=-1) -> ScoreFunction:
    def score(part: Dict[GsmNode, GsmNode]):
        total_evidence = 0
        for old_node, new_node in part.items():
            for in_sym, trans_old in old_node.transitions.items():
                trans_new = new_node.transitions.get(in_sym)
                if not trans_new:
                    continue
                for out_sym, trans_info in trans_old.items():
                    if out_sym in trans_new:
                        total_evidence += trans_info.count
        return lower_threshold(total_evidence, min_evidence)

    return score
