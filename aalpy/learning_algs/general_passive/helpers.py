import itertools
from collections import deque
from enum import Enum
from typing import Dict, Any, Tuple, Callable, Set, Union

from aalpy.learning_algs.general_passive.GsmNode import GsmNode

Score = Union[bool, float]


class Partition:
    def __init__(self, a_nodes=None, b_nodes=None):
        self.a_nodes: Set[GsmNode] = a_nodes or set()
        self.b_nodes: Set[GsmNode] = b_nodes or set()

    def __len__(self):
        return len(self.a_nodes) + len(self.b_nodes)


class Partitioning:
    def __init__(self, red: GsmNode, blue: GsmNode):
        self.red: GsmNode = red
        self.blue: GsmNode = blue
        self.score: Score = False
        self.red_mapping: Dict[GsmNode, GsmNode] = dict()
        self.full_mapping: Dict[GsmNode, GsmNode] = dict()


class FoldResult:
    def __init__(self):
        self.partitions: Set[Partition] = set()
        self.counter_examples = []


class StopMode(Enum):
    Stop = 0
    StopExploration = 1
    Continue = 2


def try_fold(a: 'GsmNode', b: 'GsmNode',
             compatible: Callable[[GsmNode, GsmNode, Dict[GsmNode, Partition]], Any] = None,
             stop_on_error: Callable[[Any], StopMode] = None
             ) -> FoldResult:
    """
    compute the partitions of two automata that result from grouping two nodes.
    supports custom compatibility criteria for early stopping in case of incompatibility and/or reporting mismatches.
    """
    compatible = compatible or (lambda x, y, z: True)
    stop_on_error = stop_on_error or (lambda err: StopMode.Stop)
    result = FoldResult()

    partition_map: Dict[GsmNode, Partition] = dict()
    q: deque[Tuple[GsmNode, GsmNode, list]] = deque([(a, b, [])])
    pair_set: Set[Tuple[GsmNode, GsmNode]] = {(a, b)}

    while len(q) != 0:
        a, b, prefix = q.popleft()

        # get partitions
        a_part = partition_map.get(a)
        b_part = partition_map.get(b)

        if a_part is None:
            a_part = Partition({a}, set())
            partition_map[a] = a_part
            result.partitions.add(a_part)
        if b_part is None:
            b_part = Partition(set(), {b})
            partition_map[b] = b_part
            result.partitions.add(b_part)
        if a_part is b_part:
            continue

        # determine compatibility
        compatibility_result = compatible(a, b, partition_map)
        if compatibility_result is not True:
            if compatibility_result is not False:
                error = (compatibility_result, prefix)
            else:
                error = prefix
            result.counter_examples.append(error)
            stop_mode = stop_on_error(compatibility_result)
            if stop_mode == StopMode.Stop:
                break
            elif stop_mode == StopMode.StopExploration:
                continue
            elif stop_mode == StopMode.Continue:
                pass  # Just continue as if nothing happened.

        # merge partitions
        if len(a_part) < len(b_part):
            other_part, part = a_part, b_part
        else:
            part, other_part = a_part, b_part

        part.a_nodes.update(other_part.a_nodes)
        part.b_nodes.update(other_part.b_nodes)

        for node in itertools.chain(other_part.a_nodes, other_part.b_nodes):
            partition_map[node] = part

        result.partitions.remove(other_part)

        # add children to work queue
        for in_sym, a_trans in a.transitions.items():
            b_trans = b.transitions.get(in_sym)
            if b_trans is None:
                continue
            for out_sym, a_next in a_trans.items():
                b_next = b_trans.get(out_sym)
                if b_next is None or (a_next.target, b_next.target) in pair_set:
                    continue
                pair_set.add((a_next.target, b_next.target))
                q.append((a_next.target, b_next.target, prefix + [(in_sym, out_sym)]))

    return result
