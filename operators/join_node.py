import math
import re
from abc import abstractmethod

import numpy as np

from operators.node import Node
from visitors.visitor import Visitor
from scipy.stats import norm


class JoinNode(Node):

    def __init__(self, plan, postgres, children=None):
        super().__init__(plan, postgres, children)
        self.f_key = self.generate_f_key()
        self.f_mean = self.plan_rows / self.children[0].plan_rows / self.children[1].plan_rows
        left_card = self.children[0].plan_rows if len(self.children[0].tables) > 1 else postgres.alias_to_rows[
            list(self.children[0].tables)[0]]
        right_card = self.children[1].plan_rows if len(self.children[1].tables) > 1 else postgres.alias_to_rows[
            list(self.children[1].tables)[0]]
        self.join_product = self.join_product * (self.plan_rows / left_card / right_card)
        self.f_nodes.add(self)
        self.f_std = self.UNCERTAINTY * self.f_mean
        # self.p_std = self.derive_join_variance(plan, postgres, left_card, right_card)
        # self.f_std = self.p_std
        prod_std_mean_square = 1
        prod_mean_square = 1
        # Derive the standard deviation of the join cardinality
        for f_node in self.f_nodes.union(self.u_nodes):
            prod_std_mean_square = prod_std_mean_square * (1 + self.UNCERTAINTY ** 2)
            prod_mean_square = prod_mean_square * f_node.f_mean
        f_error = (prod_std_mean_square - 1) ** 0.5
        delta_variance = f_error * prod_mean_square
        self.d_std = self.card_product * self.filter_product * delta_variance
        self.d_mean = self.plan_rows
        base_product = math.prod([postgres.alias_to_rows[t] for t in self.tables])
        self.b_std = self.UNCERTAINTY * math.sqrt(self.join_product * (1 - self.join_product) * base_product)
        # print(self.f_key, self.p_std)

    def generate_f_key(self):
        left_keys = "-".join(sorted(self.children[0].tables))
        right_keys = "-".join(sorted(self.children[1].tables))
        return left_keys + ":" + right_keys
        # if left_keys < right_keys:
        #     return left_keys + ":" + right_keys
        # else:
        #     return right_keys + ":" + left_keys

    def derive_join_variance(self, plan, postgres, left_card, right_card):
        condition_keys = [k for k in list(plan.keys()) if k in
                          ["Join Filter", "Recheck Cond", "Hash Cond", "Merge Cond"]]
        if len(condition_keys) == 0:
            if hasattr(self.children[1], "index_cond"):
                join_predicate = self.children[1].index_cond
            elif hasattr(self.children[1], "recheck_cond"):
                join_predicate = self.children[1].recheck_cond
            else:
                return self.UNCERTAINTY * self.f_mean
        else:
            condition_key = condition_keys[0]
            join_predicate = plan[condition_key]
        join_columns = re.findall(r"\w+.\w+", join_predicate)
        left_columns = join_columns[0].split(".")
        right_columns = join_columns[1].split(".")
        left_table = postgres.alias_to_tables[left_columns[0]]
        right_table = postgres.alias_to_tables[right_columns[0]]
        left_stats = postgres.stats_table[left_table][left_columns[1]] \
            if left_table in postgres.stats_table else None
        right_stats = postgres.stats_table[right_table][right_columns[1]] \
            if right_table in postgres.stats_table else None
        # Unique left column
        if left_stats is not None and right_stats is not None:
            if left_stats["n_distinct"] < 0:
                p_std = self.get_f_variance(right_stats)
            elif right_stats["n_distinct"] < 0:
                p_std = self.get_f_variance(left_stats)
            else:
                stats_table = left_stats if left_card > right_card else right_stats
                p_std = self.get_f_variance(stats_table)
        else:
            if left_stats is not None:
                p_std = self.get_f_variance(left_stats)
            elif right_stats is not None:
                p_std = self.get_f_variance(right_stats)
            else:
                p_std = self.UNCERTAINTY * self.f_mean
        return p_std

    def get_f_variance(self, stats_table):
        if stats_table["most_common_vals"] is not None:
            common_freqs = stats_table["most_common_freqs"]
            nr_right_distinct = stats_table["n_distinct"]
            nr_uncommon_vals = nr_right_distinct - len(common_freqs)
            uncommon_freq = 0 if nr_uncommon_vals == 0 else (1 - sum(common_freqs)) / nr_uncommon_vals
            freq_list = common_freqs
            if nr_uncommon_vals > 0:
                freq_list += [uncommon_freq] * int(nr_uncommon_vals)
            p_std = np.std(freq_list)
        else:
            p_std = self.UNCERTAINTY * self.f_mean
        return p_std

    def get_prob(self):
        z = 0.1 * self.f_mean / self.f_std
        return 2 * (norm.cdf(z) - 0.5)

    @abstractmethod
    def accept(self, visitor: Visitor) -> None:
        pass

    def is_join_node(self):
        return True
