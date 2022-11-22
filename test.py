from aalpy.SULs import DfaSUL, MealySUL, MooreSUL
from aalpy.learning_algs import run_Lstar
from aalpy.learning_algs.deterministic.KV import run_KV
from aalpy.oracles import RandomWordEqOracle, RandomWMethodEqOracle, StatePrefixEqOracle
from aalpy.utils import generate_random_deterministic_automata, get_Angluin_dfa


from random import seed
# dfa = get_Angluin_dfa()
dfa = generate_random_deterministic_automata('mealy', num_states=2000, input_alphabet_size=4, output_alphabet_size=3,)
input_al = dfa.get_input_alphabet()
sul = MealySUL(dfa)

#eq_oracle = RandomWordEqOracle(input_al, sul, min_walk_len=10, max_walk_len=15)
#eq_oracle = RandomWMethodEqOracle(input_al, sul, walks_per_state=20, walk_len=20)
eq_oracle = RandomWordEqOracle(input_al, sul, num_walks=5000, reset_after_cex=True)

import cProfile

#pr = cProfile.Profile()
#pr.enable()
learned_model = run_KV(input_al, sul, eq_oracle, automaton_type='mealy', cex_processing='rs', print_level=2)
#pr.disable()
#pr.print_stats(sort='tottime')


exit()
learned_model.visualize()

sul1 = DfaSUL(dfa)
eq_oracle = RandomWordEqOracle(input_al, sul)
learned_model = run_Lstar(input_al, sul1, eq_oracle, automaton_type='moore', cex_processing='rs',suffix_closedness=True,closedness_type='prefix',closing_strategy='single_longest',print_level=1)

# 44395 , False, single_lonest