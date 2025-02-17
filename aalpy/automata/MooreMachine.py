from typing import Generic, Dict

from aalpy.base import AutomatonState, DeterministicAutomaton
from aalpy.base.Automaton import InputType, OutputType


class MooreState(AutomatonState, Generic[InputType,OutputType]):
    """
    Single state of a Moore machine. Each state has an output value.
    """

    def __init__(self, state_id, output=None):
        super().__init__(state_id)
        self.output : OutputType = output
        self.transitions : Dict[InputType, MooreState] = dict()


class MooreMachine(DeterministicAutomaton[MooreState[InputType, OutputType]]):

    def __init__(self, initial_state: AutomatonState, states: list):
        super().__init__(initial_state, states)

    def step(self, letter):
        """
        In Moore machines outputs depend on the current state.

        Args:

            letter: single input that is looked up in the transition function leading to a new state

        Returns:

            the output of the reached state

        """
        if letter is not None:
            self.current_state = self.current_state.transitions[letter]
        return self.current_state.output

    def compute_characterization_set(self, char_set_init=None, online_suffix_closure=True, split_all_blocks=True,
                                     return_same_states=False, raise_warning=True):
        return super(MooreMachine, self).compute_characterization_set(char_set_init if char_set_init else [()],
                                                                      online_suffix_closure, split_all_blocks,
                                                                      return_same_states, raise_warning)

    def compute_output_seq(self, state, sequence):
        if not sequence:
            return [state.output]
        return super(MooreMachine, self).compute_output_seq(state, sequence)

    def to_state_setup(self):
        state_setup_dict = {}

        # ensure prefixes are computed
        self.compute_prefixes()

        sorted_states = sorted(self.states, key=lambda x: len(x.prefix) if x.prefix is not None else len(self.states))
        for s in sorted_states:
            state_setup_dict[s.state_id] = (s.output, {k: v.state_id for k, v in s.transitions.items()})

        return state_setup_dict

    @staticmethod
    def from_state_setup(state_setup : dict, **kwargs):
        """
        First state in the state setup is the initial state.
        Example state setup:
        state_setup = {
                "a": ("a", {"x": "b1", "y": "a"}),
                "b1": ("b", {"x": "b2", "y": "a"}),
                "b2": ("b", {"x": "b3", "y": "a"}),
                "b3": ("b", {"x": "b4", "y": "a"}),
                "b4": ("b", {"x": "c", "y": "a"}),
                "c": ("c", {"x": "a", "y": "a"}),
            }

        Args:

            state_setup: map from state_id to tuple(output and transitions_dict)

        Returns:

            Moore machine
        """

        # build states with state_id and output
        states = {key: MooreState(key, val[0]) for key, val in state_setup.items()}

        # add transitions to states
        for state_id, state in states.items():
            for _input, target_state_id in state_setup[state_id][1].items():
                state.transitions[_input] = states[target_state_id]

        # states to list
        states = [state for state in states.values()]

        # build moore machine with first state as starting state
        mm = MooreMachine(states[0], states)

        for state in states:
            state.prefix = mm.get_shortest_path(mm.initial_state, state)

        return mm