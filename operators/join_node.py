import math
import re
from abc import abstractmethod
from statistics import NormalDist

import numpy as np
from scipy.stats import poisson
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
        # Propagate buckets for the join node
        self.buckets = {}
        self.buketize_by_poisson(postgres, plan)

    def buketize_by_normal(self, postgres):
        # Generate buckets for the selectivity distribution
        distribution = NormalDist(mu=self.f_mean, sigma=self.f_std)
        upper = self.f_mean + 3 * self.f_std
        nr_blocks = postgres.nr_blocks
        min_val = -1 * upper / (2 * nr_blocks - 1)
        lower = max(self.f_mean - 3 * self.f_std, min_val)
        stride = max((upper - lower) / nr_blocks, 0)
        ranges = np.arange(lower, upper, stride, dtype=float)
        cdf_list = np.array([distribution.cdf(x) for x in ranges])
        probs_list = cdf_list[1:] - cdf_list[:-1]
        probs_list = probs_list / np.sum(probs_list)
        value_list = (ranges[1:] + ranges[:-1]) / 2
        np.clip(value_list, 0, None, out=value_list)
        for left_bucket, left_prob in self.children[0].buckets.items():
            for right_bucket, right_prob in self.children[1].buckets.items():
                for idx, selectivity in enumerate(value_list):
                    prob = probs_list[idx] * left_prob * right_prob
                    join_bucket = int(round(left_bucket * right_bucket * selectivity))
                    if join_bucket not in self.buckets:
                        self.buckets[join_bucket] = prob
                    else:
                        self.buckets[join_bucket] += prob
        # Re-bucketize the join node
        if len(self.buckets) > nr_blocks:
            bucket_keys = sorted(list(self.buckets.keys()))
            min_val = min(bucket_keys)
            max_val = max(bucket_keys)
            stride = (max_val - min_val) / nr_blocks
            left_ranges = np.arange(min_val, max_val, stride, dtype=float)
            right_ranges = left_ranges + stride
            val_ranges = (left_ranges + right_ranges) / 2
            range_pos = 0
            new_buckets = {}
            for bucket_key in bucket_keys:
                left = left_ranges[range_pos]
                right = right_ranges[range_pos]
                if left <= bucket_key < right:
                    if val_ranges[range_pos] not in new_buckets:
                        new_buckets[val_ranges[range_pos]] = self.buckets[bucket_key]
                    else:
                        new_buckets[val_ranges[range_pos]] += self.buckets[bucket_key]
                else:
                    while bucket_key >= right and range_pos < len(left_ranges) - 1:
                        range_pos += 1
                        left = left_ranges[range_pos]
                        right = right_ranges[range_pos]
                    if val_ranges[range_pos] not in new_buckets:
                        new_buckets[val_ranges[range_pos]] = self.buckets[bucket_key]
                    else:
                        new_buckets[val_ranges[range_pos]] += self.buckets[bucket_key]
                    # new_buckets[val_ranges[range_pos]] = self.buckets[bucket_key]
            self.buckets = new_buckets

    def buketize_by_poisson(self, postgres, plan):
        nr_blocks = postgres.nr_blocks
        for left_bucket, left_prob in self.children[0].buckets.items():
            for right_bucket, right_prob in self.children[1].buckets.items():
                prob = left_prob * right_prob
                mu = int(round(left_bucket * right_bucket * self.f_mean))
                interval_pair = poisson.interval(0.95, mu)
                if interval_pair[0] == interval_pair[1]:
                    if 0 not in self.buckets:
                        self.buckets[0] = prob
                    else:
                        self.buckets[0] += prob
                    continue
                min_val = int(interval_pair[0])
                max_val = int(interval_pair[1])
                stride = max(round((max_val - min_val) / nr_blocks), 1)
                bucket_ranges = [min_val + _ * int(stride) for _ in range(nr_blocks - 1)]
                bucket_size = bucket_ranges[1] - bucket_ranges[0]
                for idx, bucket in enumerate(bucket_ranges):
                    middle = bucket + bucket_size // 2 if idx < len(bucket_ranges) - 1 else bucket
                    p = poisson.pmf(middle, mu) * bucket_size
                    if p > 0:
                        if bucket not in self.buckets:
                            self.buckets[bucket] = p * prob
                        else:
                            self.buckets[bucket] += p * prob
                if 0 not in bucket_ranges:
                    self.buckets[0] = 0.05 * prob
                else:
                    self.buckets[bucket_ranges[-1] + stride] = 0.05 * prob
        # Re-bucketize the join node
        if len(self.buckets) > nr_blocks:
            bucket_keys = sorted(list(self.buckets.keys()))
            min_val = min(bucket_keys)
            max_val = max(bucket_keys)
            stride = max(round((max_val - min_val) / nr_blocks), 1)
            left_ranges = np.arange(min_val, max_val+1, stride, dtype=int)[:nr_blocks]
            right_ranges = left_ranges + int(stride)
            range_pos = 0
            new_buckets = {}
            for bucket_key in bucket_keys:
                left = left_ranges[range_pos]
                right = right_ranges[range_pos]
                if left <= bucket_key < right:
                    if left not in new_buckets:
                        new_buckets[left] = self.buckets[bucket_key]
                    else:
                        new_buckets[left] += self.buckets[bucket_key]
                else:
                    while bucket_key >= right and range_pos < len(left_ranges) - 1:
                        range_pos += 1
                        left = left_ranges[range_pos]
                        right = right_ranges[range_pos]
                    if left not in new_buckets:
                        new_buckets[left] = self.buckets[bucket_key]
                    else:
                        new_buckets[left] += self.buckets[bucket_key]
                    # new_buckets[val_ranges[range_pos]] = self.buckets[bucket_key]
            self.buckets = new_buckets
        # Normalize to 1
        prob_sum = sum(list(self.buckets.values()))
        for bucket_key in self.buckets:
            self.buckets[bucket_key] = self.buckets[bucket_key] / prob_sum

    def buketize_by_bernoulli(self, postgres, plan):
        coefficiency = self.UNCERTAINTY
        selectivity_ranges = []
        selectivity_probs = []
        nr_blocks = postgres.nr_blocks
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
        if ((left_stats is not None and left_stats["n_distinct"] < 0) or
                (right_stats is not None and right_stats["n_distinct"] < 0)):
            selectivity_ranges.append(self.f_mean)
            selectivity_probs.append(1)
        else:
            selectivity_ranges.append(self.f_mean)
            selectivity_probs.append(1 - coefficiency * 2)
            selectivity_ranges.append(1)
            selectivity_probs.append(coefficiency)
            selectivity_ranges.append(0)
            selectivity_probs.append(coefficiency)

        # if len(self.tables) == 2 and "mi" in self.tables and "it1" in self.tables:
        #     print("here")
        for left_bucket, left_prob in self.children[0].buckets.items():
            for right_bucket, right_prob in self.children[1].buckets.items():
                for idx, selectivity in enumerate(selectivity_ranges):
                    prob = selectivity_probs[idx] * left_prob * right_prob
                    join_bucket = int(round(left_bucket * right_bucket * selectivity))
                    if join_bucket not in self.buckets:
                        self.buckets[join_bucket] = prob
                    else:
                        self.buckets[join_bucket] += prob
        # Re-bucketize the join node
        if len(self.buckets) > nr_blocks:
            bucket_keys = sorted(list(self.buckets.keys()))
            min_val = min(bucket_keys)
            max_val = max(bucket_keys)
            stride = (max_val - min_val) / nr_blocks
            left_ranges = np.arange(min_val, max_val, stride, dtype=float)
            right_ranges = left_ranges + stride
            val_ranges = (left_ranges + right_ranges) / 2
            range_pos = 0
            new_buckets = {}
            for bucket_key in bucket_keys:
                left = left_ranges[range_pos]
                right = right_ranges[range_pos]
                if left <= bucket_key < right:
                    if val_ranges[range_pos] not in new_buckets:
                        new_buckets[val_ranges[range_pos]] = self.buckets[bucket_key]
                    else:
                        new_buckets[val_ranges[range_pos]] += self.buckets[bucket_key]
                else:
                    while bucket_key >= right and range_pos < len(left_ranges) - 1:
                        range_pos += 1
                        left = left_ranges[range_pos]
                        right = right_ranges[range_pos]
                    if val_ranges[range_pos] not in new_buckets:
                        new_buckets[val_ranges[range_pos]] = self.buckets[bucket_key]
                    else:
                        new_buckets[val_ranges[range_pos]] += self.buckets[bucket_key]
                    # new_buckets[val_ranges[range_pos]] = self.buckets[bucket_key]
            self.buckets = new_buckets

    def generate_f_key(self):
        left_keys = "-".join(sorted(self.children[0].tables))
        right_keys = "-".join(sorted(self.children[1].tables))
        return left_keys + ":" + right_keys

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
