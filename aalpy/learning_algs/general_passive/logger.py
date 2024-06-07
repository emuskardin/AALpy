import time
from functools import wraps

from aalpy.learning_algs.general_passive.GsmNode import GsmNode
from aalpy.learning_algs.general_passive.helpers import Partitioning


class DebugInfo:
    def __init__(self, lvl):
        self.lvl = lvl

    @staticmethod
    def min_lvl(lvl):
        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kw):
                if args[0].lvl < lvl:
                    return
                fn(*args, **kw)

            return wrapper

        return decorator


class DebugInfoGSM(DebugInfo):
    min_lvl = DebugInfo.min_lvl

    def __init__(self, lvl, instance: 'GeneralizedStateMerging'):
        super().__init__(lvl)
        if lvl < 1:
            return
        self.instance = instance
        self.log = []
        self.pta_size = None
        self.nr_merged_states_total = 0
        self.nr_merged_states = 0
        self.nr_red_states = 1

    @min_lvl(1)
    def pta_construction_done(self, start_time):
        print(f'PTA Construction Time: {round(time.time() - start_time, 2)}')
        if self.lvl != 1:
            states = self.instance.root.get_all_nodes()
            leafs = [state for state in states if len(state.transitions.keys()) == 0]
            depth = [state.prefix_length for state in leafs]
            self.pta_size = len(states)
            print(f'PTA has {len(states)} states leading to {len(leafs)} leafs')
            print(f'min / avg / max depth : {min(depth)} / {sum(depth) / len(depth)} / {max(depth)}')

    def print_status(self):
        print_str = f'\rCurrent automaton size: {self.nr_red_states}'
        if self.lvl != 1:
            print_str += f' Merged: {self.nr_merged_states_total} Remaining: ' \
                         f'{self.pta_size - self.nr_red_states - self.nr_merged_states_total} '
        print(print_str, end="")

    @min_lvl(1)
    def log_promote(self, node: GsmNode, red_states):
        self.log.append(["promote", (node.get_prefix(),)])
        self.nr_red_states = len(red_states)  # could be done incrementally, here for historic reasons
        self.print_status()

    @min_lvl(1)
    def log_merge(self, part: Partitioning):
        self.log.append(["merge", (part.red.get_prefix(), part.blue.get_prefix_output())])
        self.nr_merged_states_total += len(part.full_mapping) - len(part.red_mapping)
        self.nr_merged_states += 1
        self.print_status()

    @min_lvl(1)
    def learning_done(self, red_states, start_time):
        print(f'\nLearning Time: {round(time.time() - start_time, 2)}')
        print(f'Learned {len(red_states)} state automaton via {self.nr_merged_states} merges.')
        if 2 < self.lvl:
            self.instance.root.visualize("model", self.instance.output_behavior)
