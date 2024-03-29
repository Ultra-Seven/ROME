import itertools
import re
import subprocess
import time
import sys
from itertools import chain, combinations
from statistics import NormalDist

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from db.postgres import Postgres
from plan.plan_tree import PlanTree
from utils.pg_utils import get_im_result_count, create_connection
from utils.sql_utils import get_join_predicates


class Planner:
    def __init__(self, plan_trees, postgres, nr_threads=24,
                 topk=3, nr_blocks=10,
                 method="max_results",
                 max_para=3, solver="greedy", penalty=0.1, conn=None):
        self.f_keys_constants = None
        self.f_keys_to_ranges = None
        self.f_keys_to_plans = None
        self.f_keys_to_dists = None

        self.nr_selected_ims = -1
        self.plan_trees = plan_trees
        self.postgres = postgres
        self.nr_threads = nr_threads
        self.policy = method
        self.optimized_policy = solver
        self.verbose = False
        self.topk = int(topk)
        self.max_para = max_para
        self.nr_blocks = int(nr_blocks)
        self.penalty = penalty
        self.conn = conn
        self.bouquet_plans = {}

    def optimal_arms(self, plan_trees):
        # return self.greedy_max_accurate_plan(plan_trees)
        if self.policy == "max_results":
            return self.maximize_unique_results(self.max_para)
        if self.policy == "max_im_ilp":
            return self.maximize_unique_im_ilp()
        if self.policy == "max_im_ilp_parallel":
            return self.maximize_unique_im_ilp(self.max_para)
        elif self.policy == "fix_plan":
            return [self.max_para]
        elif self.policy == "probability_model":
            return self.probability_model(plan_trees, mat_im=False)
        elif self.policy == "predicate":
            return self.predicate_model(plan_trees)
        elif self.policy == "topk_least":
            return self.topk_least(plan_trees, self.max_para)
        elif self.policy == "bouquet" or self.policy == "rome_bouquet" or self.policy == "bouquet_topk":
            return self.plan_bouquet(topk=self.max_para, json=(self.policy == "rome_bouquet"))
        else:
            return self.probability_model(plan_trees, mat_im=False)

    def maximize_unique_results(self, max_parallelism=-1):
        optimal_arms = []
        # Create a set to hold the covered elements
        covered = set()
        # Create a set to hold the uncovered elements
        uncovered = set()
        plan_intermediate_results = {arm_idx: set([f_node.f_key for f_node in self.plan_trees[arm_idx].root.f_nodes])
                                     for arm_idx in self.plan_trees}
        for arm_idx in plan_intermediate_results:
            uncovered |= plan_intermediate_results[arm_idx]

        # Iterate until all elements are covered
        while max_parallelism == -1 or len(optimal_arms) < max_parallelism:
            # Choose the set that covers the most uncovered elements
            remaining_arms = [arm_idx for arm_idx in self.plan_trees if arm_idx not in optimal_arms]
            best_arm = max(remaining_arms, key=lambda a: len(plan_intermediate_results[a].intersection(uncovered)))
            best_set = plan_intermediate_results[best_arm]
            # Add the set to the minimum set coverage
            covered |= best_set
            # Remove the covered elements from the set of uncovered elements
            uncovered -= best_set
            optimal_arms.append(best_arm)
        self.nr_selected_ims = len(list(covered))
        return optimal_arms

    def maximize_unique_im_ilp(self, max_parallelism=-1):
        optimal_arms = []
        # Create a set to hold the uncovered elements
        uncovered = set()
        plan_intermediate_results = {arm_idx: set([f_node.f_key for f_node in self.plan_trees[arm_idx].root.f_nodes])
                                     for arm_idx in self.plan_trees}
        for arm_idx in plan_intermediate_results:
            uncovered |= plan_intermediate_results[arm_idx]
        try:
            # Create a new model
            m = gp.Model("maximize_unique_im")
            m.Params.LogToConsole = 0
            # Create plan variables
            plan_vars = {arm_idx: m.addVar(vtype=GRB.BINARY, name="p_" + str(arm_idx))
                         for arm_idx in self.plan_trees}
            # Create intermediate result variables
            intermediate_vars = {f_key: m.addVar(vtype=GRB.BINARY, name="i_" + str(f_key))
                                 for f_key in uncovered}

            # Add constraints: intermediate result is selected <=> at least one plan is selected
            if max_parallelism > 0:
                plan_im_vars = {arm_idx: {f_key: m.addVar(vtype=GRB.BINARY, name="pi_" + str(f_key))
                                          for f_key in plan_intermediate_results[arm_idx]}
                                for arm_idx in self.plan_trees}
                for arm_idx, arm_var in plan_vars.items():
                    plan_im_results = [plan_im_vars[arm_idx][f_key] for f_key in plan_im_vars[arm_idx]]
                    nr_im = len(plan_im_results)

                    m.addConstr(gp.quicksum([im_var for im_var in plan_im_results]) == nr_im * arm_var,
                                "cp_" + str(arm_idx))
                for f_key, im_var in intermediate_vars.items():
                    im_plans = [arm_idx for arm_idx in plan_intermediate_results
                                if f_key in plan_intermediate_results[arm_idx]]
                    m.addConstr(gp.quicksum([plan_im_vars[arm_idx][f_key] for arm_idx in im_plans])
                                >= im_var, "ci_" + str(f_key))
            else:
                for f_key, im_var in intermediate_vars.items():
                    im_plans = [arm_idx for arm_idx in plan_intermediate_results
                                if f_key in plan_intermediate_results[arm_idx]]
                    m.addConstr(gp.quicksum([plan_vars[arm_idx] for arm_idx in im_plans]) >= 1,
                                "c_" + str(f_key))
            # Add constraint: parallelism constraint
            if max_parallelism > 0:
                m.addConstr(gp.quicksum([plan_vars[arm_idx] for arm_idx in self.plan_trees])
                            <= max_parallelism, "c_max_parallelism")
                # Set objective
                m.setObjective(gp.quicksum([inter_var for inter_var in list(intermediate_vars.values())]) -
                               0.001 * gp.quicksum([arm_var for arm_var in list(plan_vars.values())]), GRB.MAXIMIZE)
            else:
                # Set objective
                m.setObjective(gp.quicksum([arm_var for arm_var in list(plan_vars.values())]), GRB.MINIMIZE)

            m.update()
            # Optimize model
            m.optimize()
            optimal_arms = [arm_idx for arm_idx in self.plan_trees if m.getVarByName(f"p_{arm_idx}").X == 1]
            selected_ims = [f_key for f_key in uncovered if m.getVarByName(f"i_{f_key}").X == 1]
            self.nr_selected_ims = len(selected_ims)
            print(optimal_arms)

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))

        except AttributeError as e:
            print('Encountered an attribute error: ' + str(e))
        return optimal_arms

    def greedy_max_accurate_plan(self, plan_trees, nr_threads=24):
        optimal_arms = []
        accurate_prob = 0.9
        intermediate_groups = {}
        for arm_idx in plan_trees:
            plan = plan_trees[arm_idx].root
            intermediate_selectivity = plan.f_nodes
            for selectivity_node in intermediate_selectivity:
                f_key = selectivity_node.f_key
                if f_key not in intermediate_groups:
                    intermediate_groups[f_key] = []
                intermediate_groups[f_key].append(arm_idx)
        # Prune equivalent plans
        plan_maps = {}
        for k in plan_trees:
            p_key = frozenset([i for i in intermediate_groups if k in intermediate_groups[i]])
            if p_key not in plan_maps:
                plan_maps[p_key] = []
            plan_maps[p_key].append(k)
        removed_plans = []
        for p_key in plan_maps:
            plan_val_list = sorted(list(plan_maps[p_key]), key=lambda m: plan_trees[m].root.total_cost)
            if len(plan_maps[p_key]) > 1:
                removed_plans = removed_plans + plan_val_list[1:]
        for i_key in intermediate_groups:
            for r_plan in removed_plans:
                if r_plan in intermediate_groups[i_key]:
                    intermediate_groups[i_key].remove(r_plan)
        indicator_dict = {}
        for intermediate_result in intermediate_groups:
            involved_plans = intermediate_groups[intermediate_result]
            i_key = frozenset(involved_plans)
            if i_key not in indicator_dict:
                indicator_dict[i_key] = []
            indicator_dict[i_key].append(intermediate_result)

        plan_keys = sorted(list(set(plan_trees.keys()).difference(removed_plans)))
        combinations_results = list(chain(*map(lambda x: combinations(plan_keys, x),
                                          range(0, len(plan_keys) + 1))))
        # Add an arm to the candidate set
        next_plan_index = {}
        for combination in combinations_results:
            if len(combination) == 0:
                continue

            accurate_results = set()
            inaccurate_plans = set(plan_keys).difference(set(combination))
            inaccurate_indicator_dict = {}
            for i_key, i_val in indicator_dict.items():
                if i_key.intersection(set(combination)):
                    accurate_results = accurate_results.union(i_val)
                else:
                    c_prob = accurate_prob ** len(i_val)
                    inaccurate_indicator_dict[i_key] = {"correct": c_prob, "incorrect": 1 - c_prob}
            prob = accurate_prob ** len(accurate_results)
            # The rest of plans are inaccurate
            rest_combinations = list(chain(*map(lambda x: combinations(list(inaccurate_indicator_dict.keys()), x),
                                                range(0, len(inaccurate_indicator_dict.keys()) + 1))))
            incorrect_prob = 1 if len(rest_combinations) == 1 else 0
            for rest_combination in rest_combinations[1:]:
                covering_set = frozenset.union(*list(rest_combination))
                if covering_set.__eq__(inaccurate_plans):
                    incorrect_list = [ia_v["incorrect"] if ia_key in rest_combination else ia_v["correct"]
                                      for ia_key, ia_v in inaccurate_indicator_dict.items()]
                    incorrect_prob += np.prod(incorrect_list)
            prob = prob * incorrect_prob
            next_plan_index[frozenset(combination)] = prob

        # Find the candidates of optimal arms in a greedy way
        best_arm = None
        best_utility = sys.maxsize
        max_cost = max([plan_trees[ap].root.total_cost for ap in plan_trees]) * 10
        while True:
            nr_threads_per_plan = nr_threads // (len(optimal_arms) + 1)
            for arm_idx in plan_trees:
                if arm_idx in optimal_arms or arm_idx in removed_plans:
                    continue
                utility = 0
                current_arms = frozenset(optimal_arms + [arm_idx])

                for accurate_plans_key in next_plan_index:
                    accurate_plans = accurate_plans_key.intersection(current_arms)
                    if len(accurate_plans) > 0:
                        cost = min([plan_trees[ap].root.total_cost for ap in accurate_plans])
                        delta_cost = (next_plan_index[accurate_plans_key] * cost / nr_threads_per_plan)
                    else:
                        delta_cost = (next_plan_index[accurate_plans_key] * max_cost)
                        if arm_idx == 4 or len(current_arms) == 2:
                            print("Above is ")
                    utility += delta_cost
                    if arm_idx == 4 or len(current_arms) == 2:
                        print(accurate_plans_key, delta_cost)
                if arm_idx == 4 or len(current_arms) == 2:
                    print("Current key:", current_arms, utility, plan_trees[arm_idx].root.total_cost)
                if utility < best_utility:
                    best_utility = utility
                    best_arm = arm_idx
            if best_arm is None:
                break
            optimal_arms.append(best_arm)
            best_arm = None
        return optimal_arms

    def topk_least(self, plan_trees, topk=3):
        plan_intermediate_results = {arm_idx: set([f_node.f_key for f_node in self.plan_trees[arm_idx].root.f_nodes])
                                     for arm_idx in self.plan_trees}
        covered = set()
        total_cost_dict = {}
        for arm_idx in plan_trees:
            total_cost = plan_trees[arm_idx].plan['Plan']['Total Cost']
            total_cost_dict[total_cost] = arm_idx
        sorted_costs = sorted(list(total_cost_dict.keys()))[:topk]
        sorted_arms = [total_cost_dict[c] for c in sorted_costs]
        for best_arm in sorted_arms:
            best_set = plan_intermediate_results[best_arm]
            # Add the set to the minimum set coverage
            covered |= best_set
        self.nr_selected_ims = len(list(covered))
        return sorted_arms

    def plan_bouquet(self, topk=3, nr_blocks=5, nr_predicates=3, json=False):
        self.bouquet_plans = {}
        sql = self.postgres.sql.replace(";", "")
        join_predicates = get_join_predicates(sql)
        table_pairs = set()
        def get_op(predicate):
            if ">=" in predicate:
                return ">="
            elif "<=" in predicate:
                return "<="
            elif "<" in predicate:
                return "<"
            elif ">" in predicate:
                return ">"
            else:
                return "="
        for join_predicate in join_predicates:
            op = get_op(join_predicate)
            join_list = join_predicate.split(op)
            left_alias = join_list[0].strip().split(".")[0]
            right_alias = join_list[1].strip().split(".")[0]
            if left_alias > right_alias:
                table_pairs.add((right_alias, left_alias))
            else:
                table_pairs.add((left_alias, right_alias))
        unary_predicates = [u.strip() for u in sql.split("WHERE")[-1].split("AND")
                            if u.strip() not in join_predicates]
        table_pairs = sorted(list(table_pairs))
        priority_pairs = set()
        not_priority_pairs = set()
        for unary_predicate in unary_predicates:
            if ("LIKE " in unary_predicate.upper() or ">" in unary_predicate or
                    "<" in unary_predicate or " IN " in unary_predicate.upper()
                    or " BETWEEN " in unary_predicate.upper()):
                for p in table_pairs:
                    if p not in priority_pairs and (f"{p[0]}." in unary_predicate or f"{p[1]}." in unary_predicate):
                        priority_pairs.add(p)
                        print(p, unary_predicate)
                        break
                if len(priority_pairs) == nr_predicates:
                    break
        for unary_predicate in unary_predicates:
            if not ("LIKE " in unary_predicate.upper() or ">" in unary_predicate or
                    "<" in unary_predicate or " IN " in unary_predicate.upper()
                    or " BETWEEN " in unary_predicate.upper()):
                for p in table_pairs:
                    if p not in priority_pairs and (f"{p[0]}." in unary_predicate or f"{p[1]}." in unary_predicate):
                        not_priority_pairs.add(p)
        not_priority_pairs = sorted(list(not_priority_pairs))
        while len(priority_pairs) < nr_predicates and len(not_priority_pairs) > 0:
            priority_pairs.add(not_priority_pairs.pop())
        # Generate hyperspace
        block_dict = {}
        for p in priority_pairs:
            max_size = self.postgres.alias_to_rows[p[0]] * self.postgres.alias_to_rows[p[1]]
            block_size = max_size // nr_blocks + 1
            block_lefts = np.arange(0, max_size, block_size)
            block_lefts = block_lefts + 1
            block_dict[p] = block_lefts

        keys = list(block_dict.keys())
        values = [block_dict[k] for k in keys]
        all_combinations = list(itertools.product(*list(values)))
        hint_dict = {}
        value_set = set()
        for c in all_combinations:
            hints = []
            for idx, card in enumerate(c):
                hints.append(f"Rows ({keys[idx][0]} {keys[idx][1]} #{card})")
            hint_comment = f"/*+\n" + "\n".join(hints) + "\n*/"
            if json:
                target_sql = f"EXPLAIN (COSTS, VERBOSE, FORMAT JSON) {hint_comment} {sql};"
            else:
                target_sql = f"EXPLAIN {hint_comment} {sql};"

            # cur.execute(target_sql)
            # results = cur.fetchall()
            # costs = [int(x.split("=")[-1]) for x in re.findall(r"cost=\d+", results[0][0])]
            result = subprocess.run(['psql', '-h', 'localhost',
                                     '-U', 'postgres', '-d', self.postgres.database, '-XqAt',
                                     '-c', "LOAD 'pg_hint_plan';", '-c', target_sql],
                                    stdout=subprocess.PIPE)
            self.bouquet_plans[hint_comment] = result
            result_str = result.stdout.decode('utf-8')
            if json:
                costs = [int(x.split(": ")[-1]) for x in re.findall(r'"Total Cost": \d+', result_str)]
            else:
                costs = [int(x.split("=")[-1]) for x in re.findall(r"cost=\d+", result_str)]
            cost = max(costs)
            if cost not in value_set:
                value_set.add(cost)
                hint_dict[hint_comment] = cost
        sorted_cost_hints = sorted(list(hint_dict.keys()), key=lambda x: hint_dict[x])
        return sorted_cost_hints[:topk]

    def probability_model(self, plan_trees, nr_threads=24, mat_im=False):
        # Prune equivalent plans
        intermediate_groups = {}
        for arm_idx in plan_trees:
            plan = plan_trees[arm_idx].root
            intermediate_selectivity = plan.f_nodes
            for selectivity_node in intermediate_selectivity:
                f_key = selectivity_node.f_key
                if f_key not in intermediate_groups:
                    intermediate_groups[f_key] = []
                intermediate_groups[f_key].append(arm_idx)
        if mat_im:
            postgres = Postgres("localhost", 5432, "postgres",
                                "postgres", self.postgres.database)
            postgres.set_sql_query(self.postgres.sql, self.postgres.query_name)
            for arm_idx in plan_trees:
                plan = plan_trees[arm_idx].root
                intermediate_selectivity = plan.f_nodes
                with open(f"./figs/{self.postgres.query_name}/im_{arm_idx}.txt", "w") as f:
                    for selectivity_node in intermediate_selectivity:
                        f_key = selectivity_node.f_key
                        left_key = f_key.split(":")[0].split("-")
                        right_key = f_key.split(":")[1].split("-")
                        joined_tables = sorted(list(set(left_key).union(set(right_key))))
                        estimated_card = selectivity_node.d_mean
                        true_card = get_im_result_count(postgres, self.postgres.sql, joined_tables)
                        f.write(f"{','.join(joined_tables)}|{estimated_card}|{true_card}\n")
                        print(f"{','.join(joined_tables)}|{estimated_card}|{true_card}")
            postgres.close()
        plan_maps = {}
        for k in plan_trees:
            p_key = frozenset([i for i in intermediate_groups if k in intermediate_groups[i]])
            if p_key not in plan_maps:
                plan_maps[p_key] = []
            plan_maps[p_key].append(k)
        removed_plans = []
        for p_key in plan_maps:
            plan_val_list = sorted(list(plan_maps[p_key]), key=lambda m: plan_trees[m].root.total_cost)
            if len(plan_maps[p_key]) > 1:
                removed_plans = removed_plans + plan_val_list[1:]
        print("Remaining plans: ", set(range(0, 6)).difference(removed_plans))
        if self.optimized_policy == "monte carlo":
            optimal_arms = self.monte_carlo_method(removed_plans)
        elif self.optimized_policy == "ilp" or self.policy == "probability_model_ilp":
            optimal_arms = self.minimize_cost_ilp(removed_plans, self.max_para, self.topk, self.nr_blocks)
        else:
            # optimal_arms = self.block_based_distorted_objective(removed_plans, self.topk, self.nr_blocks)
            optimal_arms = self.dist_free_distorted_objective(removed_plans, self.topk, self.nr_blocks)
        return optimal_arms

    def predicate_model(self, plan_trees):
        # Prune equivalent plans
        intermediate_groups = {}
        for arm_idx in plan_trees:
            plan = plan_trees[arm_idx].root
            intermediate_selectivity = plan.f_nodes
            for selectivity_node in intermediate_selectivity:
                f_key = selectivity_node.f_key
                if f_key not in intermediate_groups:
                    intermediate_groups[f_key] = []
                intermediate_groups[f_key].append(arm_idx)
        plan_maps = {}
        for k in plan_trees:
            p_key = frozenset([i for i in intermediate_groups if k in intermediate_groups[i]])
            if p_key not in plan_maps:
                plan_maps[p_key] = []
            plan_maps[p_key].append(k)
        removed_plans = []
        for p_key in plan_maps:
            plan_val_list = sorted(list(plan_maps[p_key]), key=lambda m: plan_trees[m].root.total_cost)
            if len(plan_maps[p_key]) > 1:
                removed_plans = removed_plans + plan_val_list[1:]
        if self.optimized_policy == "monte carlo":
            optimal_arms = self.monte_carlo_method_predicate(removed_plans)
        elif self.optimized_policy == "ilp":
            optimal_arms = self.minimize_cost_ilp(removed_plans)
        else:
            optimal_arms = self.expected_minimum_normal_variables(removed_plans)
        return optimal_arms

    def generate_ranges(self, removed_plans, top_k, nr_blocks, is_combined=True, use_buckets=False):
        f_keys_to_plans = {}
        f_keys_to_dists = {}
        f_keys_to_buckets = {}
        # Compute the expectation via Monte Carlo methods
        nr_remaining_plans = 0
        for arm_idx in self.plan_trees:
            if arm_idx not in removed_plans:
                nr_remaining_plans += 1
                plan = self.plan_trees[arm_idx]
                for f_node in plan.root.f_nodes:
                    f_key = f_node.f_key
                    left_key = f_key.split(":")[0].split("-")
                    right_key = f_key.split(":")[1].split("-")
                    f_key = frozenset(left_key).union(frozenset(right_key))
                    if f_key not in f_keys_to_plans:
                        f_keys_to_plans[f_key] = []
                        f_keys_to_dists[f_key] = (f_node.d_mean, f_node.d_std)
                        f_keys_to_buckets[f_key] = f_node.buckets
                        # f_keys_to_dists[f_key] = (f_node.d_mean, f_node.b_std)
                    f_keys_to_plans[f_key].append(arm_idx)
        if is_combined:
            reverse_dict = dict()
            for f_key, f_set in f_keys_to_plans.items():
                if frozenset(f_set) not in reverse_dict:
                    reverse_dict[frozenset(f_set)] = (f_key, f_keys_to_dists[f_key])
                else:
                    union_set = reverse_dict[frozenset(f_set)][0].union(frozenset(f_key))
                    f_mean = reverse_dict[frozenset(f_set)][1][0] + f_keys_to_dists[f_key][0]
                    f_std = np.sqrt(reverse_dict[frozenset(f_set)][1][0] ** 2 + f_keys_to_dists[f_key][1] ** 2)
                    reverse_dict[frozenset(f_set)] = (union_set, (f_mean, f_std))
            combined_dict = dict()
            combined_dist = dict()
            for f_set, f_keys in reverse_dict.items():
                combined_dict[f_keys[0]] = f_set
                combined_dist[f_keys[0]] = f_keys[1]
            f_keys_to_plans = combined_dict
            f_keys_to_dists = combined_dist
        # print("Print all plans")
        # for f_key in f_keys_to_plans:
        #     print(f_key, f_keys_to_plans[f_key], f_keys_to_buckets[f_key])
        # large_f_keys = sorted([f_key for f_key in f_keys_to_plans if len(f_keys_to_plans[f_key]) < nr_remaining_plans],
        #                       key=lambda k: f_keys_to_dists[k][1], reverse=True)[:top_k]
        # print(f_keys_to_plans)
        # print(f_keys_to_buckets)
        large_f_keys = sorted([f_key for f_key in f_keys_to_plans if len(f_keys_to_plans[f_key]) < nr_remaining_plans],
                              key=lambda k: max(list(f_keys_to_buckets[k].keys())) -
                                            min(list(f_keys_to_buckets[k].keys())), reverse=True)[:top_k]
        # large_f_keys = set()
        # for arm_idx in self.plan_trees:
        #     large_key = None
        #     if arm_idx not in removed_plans:
        #         plan = self.plan_trees[arm_idx]
        #         for f_node in plan.root.f_nodes:
        #             f_key = f_node.f_key
        #             left_key = f_key.split(":")[0].split("-")
        #             right_key = f_key.split(":")[1].split("-")
        #             f_key = frozenset(left_key).union(frozenset(right_key))
        #             if len(f_keys_to_plans[f_key]) < nr_remaining_plans:
        #                 vals = list(f_keys_to_buckets[f_key].keys())
        #                 delta = max(vals) - min(vals)
        #                 if large_key is None or delta > large_key[1]:
        #                     large_key = (f_key, delta)
        #         large_f_keys.add(large_key[0])
        # for large_key in large_f_keys:
        #     print(large_key, f_keys_to_plans[large_key],
        #           f_keys_to_buckets[large_key], sum([x * y for x, y in f_keys_to_buckets[large_key].items()]))
        f_keys_to_ranges = {}
        if use_buckets:
            for k in large_f_keys:
                value_list = list(f_keys_to_buckets[k].keys())
                probs_list = [f_keys_to_buckets[k][v] for v in value_list]
                plan_list = [k for _ in range(len(value_list))]
                f_keys_to_ranges[k] = list(zip(value_list, probs_list, plan_list))
        else:
            for k in large_f_keys:
                distribution = NormalDist(mu=f_keys_to_dists[k][0], sigma=f_keys_to_dists[k][1])
                upper = f_keys_to_dists[k][0] + 3 * f_keys_to_dists[k][1]
                min_val = -1 * upper / (2 * nr_blocks - 1)
                lower = max(f_keys_to_dists[k][0] - 3 * f_keys_to_dists[k][1], min_val)
                stride = max((upper - lower) / nr_blocks, 0)
                ranges = np.arange(lower, upper, stride, dtype=float)
                cdf_list = np.array([distribution.cdf(x) for x in ranges])
                probs_list = cdf_list[1:] - cdf_list[:-1]
                probs_list = probs_list / np.sum(probs_list)
                value_list = (ranges[1:] + ranges[:-1]) / 2
                if value_list[0] < 0:
                    value_list[0] = 0
                plan_list = [k for _ in range(len(value_list))]
                f_keys_to_ranges[k] = list(zip(value_list, probs_list, plan_list))
        f_keys_constants = {f_key: f_keys_to_plans[f_key] for f_key in f_keys_to_plans
                            if f_key not in f_keys_to_ranges and len(f_keys_to_plans[f_key]) < nr_remaining_plans}
        return f_keys_to_plans, f_keys_to_dists, f_keys_to_ranges, f_keys_constants

    def minimize_cost_ilp(self, removed_plans, max_parallelism=-1, top_k=3, nr_blocks=20):
        (f_keys_to_plans, f_keys_to_dists,
         f_keys_to_ranges, f_keys_constants) = self.generate_ranges(removed_plans, top_k,
                                                                    2, is_combined=False, use_buckets=True)
        # if True:
        try:
            # Create a new model
            m = gp.Model("minimize_cost")
            # m.Params.LogToConsole = 0
            # Create plan variables
            plan_vars = {arm_idx: m.addVar(vtype=GRB.BINARY, name="p_" + str(arm_idx))
                         for arm_idx in self.plan_trees if arm_idx not in removed_plans}
            plan_mean_dict = {arm_idx: 0 for arm_idx in self.plan_trees if arm_idx not in removed_plans}
            for f_key in f_keys_to_dists:
                for arm_idx in f_keys_to_plans[f_key]:
                    plan_mean_dict[arm_idx] += f_keys_to_dists[f_key][0]
            M = max(list(plan_mean_dict.values())) * 10
            PRHS = len(plan_vars) - 1
            # f_keys_constants = {f_key: f_keys_to_plans[f_key] for f_key in f_keys_to_plans
            #                     if f_key not in f_keys_to_ranges and len(f_keys_to_plans[f_key]) < len(plan_vars)}
            # Create intermediate variables
            intermediate_vars = {f_key: m.addVar(vtype=GRB.BINARY, name="i_" + str(f_key))
                                 for f_key in f_keys_to_plans if len(f_keys_to_plans[f_key]) < len(plan_vars)}
            # Create block variables for each intermediate result
            # blocks_vars = {f_key: {idx: m.addVar(vtype=GRB.BINARY, name=f"{str(f_key)}_{idx}")
            #                        for idx, b in enumerate(f_keys_to_ranges[f_key])}
            #                for f_key in f_keys_to_ranges}
            # Create minimum variable for each combination of blocks
            all_combinations = list(itertools.product(*list(f_keys_to_ranges.values())))
            combination_vars = {idx: m.addVar(vtype=GRB.INTEGER, name=f"z_{idx}")
                                for idx, f_key in enumerate(all_combinations)}
            extra_vars = {idx: {arm_idx: m.addVar(vtype=GRB.BINARY, name=f"z_{idx}_y_{arm_idx}")
                                for arm_idx in plan_vars}
                          for idx, f_key in enumerate(all_combinations)}

            # Constraint 1: parallelism constraint
            if max_parallelism > 0:
                m.addConstr(gp.quicksum([plan_vars[arm_idx] for arm_idx in plan_vars])
                            <= max_parallelism, "c_max_parallelism")

            # Constraint 2: intermediate result in the plan is selected <=> a plan is selected
            for arm_idx, arm_var in plan_vars.items():
                f_keys_in_plan = [intermediate_vars[f_key] for f_key in f_keys_to_plans if arm_idx in
                                  f_keys_to_plans[f_key] and f_key in intermediate_vars]
                nr_im = len(f_keys_in_plan)
                m.addConstr(gp.quicksum(f_keys_in_plan) >= nr_im * arm_var, "cp_" + str(arm_idx))
            for f_key, f_var in intermediate_vars.items():
                plan_vars_in_f_var = [plan_vars[arm_idx] for arm_idx in f_keys_to_plans[f_key] if arm_idx in plan_vars]
                nr_plans = len(plan_vars_in_f_var)
                m.addConstr(gp.quicksum(plan_vars_in_f_var) <= nr_plans * f_var, "ci_" + str(f_key))
                m.addConstr(gp.quicksum(plan_vars_in_f_var) >= f_var, "ci2_" + str(f_key))

            # Constraint 3: blocks are selected <=> the intermediate is selected
            # for f_key, f_block_vars in blocks_vars.items():
            #     block_var_list = list(f_block_vars.values())
            #     nr_blocks = len(block_var_list)
            #     m.addConstr(gp.quicksum(block_var_list) == nr_blocks * intermediate_vars[f_key], "cb_" + str(f_key))

            # Constraint 4: for each combination of range blocks, solve min(min(p1, p2, ..., pn))
            probabilities = []
            for idx, combination in enumerate(all_combinations):
                probability = 1
                for block in combination:
                    probability *= block[1]
                probabilities.append(probability)
                plan_cost = {arm_idx: 0 for arm_idx in plan_vars}
                # Add the constant intermediate results
                for f_key, plan_set in f_keys_constants.items():
                    for arm_idx in plan_set:
                        plan_cost[arm_idx] += f_keys_to_dists[f_key][0]
                # Add the variable intermediate results
                for range_tuple in combination:
                    f_key = range_tuple[2]
                    plan_set = f_keys_to_plans[f_key]
                    for arm_idx in plan_set:
                        plan_cost[arm_idx] += range_tuple[0]
                for arm_idx, p_var in plan_vars.items():
                    plan_cost_const = plan_cost[arm_idx] - M
                    m.addConstr((plan_cost_const * p_var + M) <= (combination_vars[idx] +
                                10 * M * extra_vars[idx][arm_idx]), f"cc_{idx}_p_{arm_idx}")
                    m.addConstr(plan_cost_const * p_var >= plan_cost_const,
                                f"tt_{idx}_p_{arm_idx}")
                extra_vars_list = list(extra_vars[idx].values())
                m.addConstr(gp.quicksum(extra_vars_list) == PRHS, f"cy_{idx}")
            # m.addConstr(plan_vars[0] == 1, f"test_0")
            # Set objective
            m.setObjective(gp.quicksum([c_var * probabilities[idx]
                                        for idx, c_var in combination_vars.items()]), GRB.MINIMIZE)
            m.update()
            m.Params.TimeLimit = 60
            m.optimize()
            # Optimize model
            print('Obj: %g' % m.objVal)
            self.nr_selected_ims = float(m.objVal) + sum([f_keys_to_dists[f_key][0]
                                                          for f_key, plan_set in f_keys_to_plans.items()
                                                          if len(plan_set) == len(plan_vars)])
            optimal_arms = [arm_idx for arm_idx in plan_vars if plan_vars[arm_idx].X > 0.5]
            print(optimal_arms)

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))
            m.computeIIS()
            m.write(f"model_{self.postgres.query_name}.ilp")

        except AttributeError as e:
            print('Encountered an attribute error: ' + str(e))
            m.computeIIS()
            m.write(f"model_{self.postgres.query_name}.ilp")
        return optimal_arms

    def block_based_distorted_objective(self, removed_plans, top_k=3, nr_blocks=20):
        optimal_arms = []
        (f_keys_to_plans, f_keys_to_dists,
         f_keys_to_ranges, f_keys_constants) = self.generate_ranges(removed_plans, top_k, nr_blocks)
        best_utility = sys.maxsize
        plan_list = [arm_idx for arm_idx in self.plan_trees if arm_idx not in removed_plans]
        if self.verbose:
            # target_keys = [f_key for f_key in f_keys_to_plans if
            #                1 in f_keys_to_plans[f_key] and 3 in f_keys_to_plans[f_key]]
            # print({f_key: f_keys_to_dists[f_key] for f_key in target_keys})
            print("f keys to plans: ", f_keys_to_plans)
            print("f keys to dists: ", f_keys_to_dists)
            print("f keys to ranges: ", f_keys_to_ranges)
        all_combinations = list(itertools.product(*list(f_keys_to_ranges.values())))
        while len(optimal_arms) < min(self.max_para, len(plan_list)):
            if len(optimal_arms) == 0:
                plan_exp = {arm_idx: 0 for arm_idx in plan_list}
                for f_key, plan_set in f_keys_to_plans.items():
                    if len(plan_set) < len(plan_list):
                        exp = f_keys_to_dists[f_key][0]
                        for arm_idx in plan_set:
                            plan_exp[arm_idx] += exp
                unique_dict = {arm_idx: 0 for arm_idx in plan_list}
                best_arm = sorted(plan_exp.keys(), key=lambda x: plan_exp[x])[0]
                optimal_arms.append(best_arm)
                for f_key in f_keys_to_ranges:
                    plan_set = f_keys_to_plans[f_key]
                    for arm_idx in plan_set:
                        unique_dict[arm_idx] += sum([x[0] * x[1] for x in f_keys_to_ranges[f_key]])
                    # if best_arm in plan_set:
                    #     variable_utility += sum([x[0] * x[1] for x in f_keys_to_ranges[f_key]])
                for f_key in f_keys_constants:
                    plan_set = f_keys_to_plans[f_key]
                    for arm_idx in plan_set:
                        unique_dict[arm_idx] += f_keys_to_dists[f_key][0]
                    # if best_arm in plan_set:
                    #     variable_utility += f_keys_to_dists[f_key][0]
                best_utility = unique_dict[best_arm]
                if self.verbose:
                    print("Best arm:", best_arm, unique_dict, plan_exp)
            else:
                best_arm = None
                next_arms = {frozenset(optimal_arms).union([arm_idx]): 0
                             for arm_idx in plan_list if arm_idx not in optimal_arms}
                for next_arm in next_arms:
                    for combination in all_combinations:
                        all_prob = 1
                        local_plan_cost = {arm_idx: 0 for arm_idx in next_arm}
                        for f_key in f_keys_constants:
                            for arm_idx in next_arm:
                                if arm_idx in f_keys_to_plans[f_key]:
                                    local_plan_cost[arm_idx] += f_keys_to_dists[f_key][0]
                        for value, prob, f_key in combination:
                            all_prob *= prob
                            for arm_idx in next_arm:
                                if arm_idx in f_keys_to_plans[f_key]:
                                    local_plan_cost[arm_idx] += value

                        min_plan = min(list(local_plan_cost.keys()), key=lambda k: local_plan_cost[k])
                        min_value = all_prob * local_plan_cost[min_plan]
                        next_arms[next_arm] += min_value
                next_arm_list = list(next_arms.keys())
                next_arm_list.reverse()
                local_best_arm = sorted(next_arm_list, key=lambda x: next_arms[x])[0]
                local_utility = next_arms[local_best_arm] + self.penalty * len(optimal_arms)
                if self.verbose:
                    print(next_arms, local_utility, best_utility)
                if local_utility < best_utility:
                    best_utility = local_utility
                    best_arm = local_best_arm
                    optimal_arms = optimal_arms + [x for x in local_best_arm if x not in optimal_arms]

            if best_arm is None:
                break
        self.nr_selected_ims = float(best_utility) + sum([f_keys_to_dists[f_key][0]
                                                          for f_key, plan_set in f_keys_to_plans.items()
                                                          if len(plan_set) == len(plan_list)])
        return sorted(optimal_arms)

    def dist_free_distorted_objective(self, removed_plans, top_k=3, nr_blocks=20):
        optimal_arms = []
        (f_keys_to_plans, f_keys_to_dists,
         f_keys_to_ranges, f_keys_constants) = self.generate_ranges(removed_plans, top_k,
                                                                    2, is_combined=False, use_buckets=True)
        self.f_keys_to_plans = f_keys_to_plans
        self.f_keys_to_dists = f_keys_to_dists
        self.f_keys_to_ranges = f_keys_to_ranges
        self.f_keys_constants = f_keys_constants
        best_utility = sys.maxsize
        plan_list = [arm_idx for arm_idx in self.plan_trees if arm_idx not in removed_plans]
        if self.verbose:
            print("f keys to plans: ", f_keys_to_plans)
            print("f keys to dists: ", f_keys_to_dists)
            print("f keys to ranges: ", f_keys_to_ranges)

        # global_constant_cost = {arm_idx: 0 for arm_idx in plan_list}
        # for f_key in f_keys_constants:
        #     for arm_idx in f_keys_to_plans[f_key]:
        #         global_constant_cost[arm_idx] += f_keys_to_dists[f_key][0]
        # combination_dict = {arm_idx: np.full(len(all_combinations), global_constant_cost[arm_idx])
        #                     for arm_idx in plan_list}
        # for arm_idx in plan_list:

        while len(optimal_arms) < min(self.max_para, len(plan_list)):
            if len(optimal_arms) == 0:
                plan_exp = {arm_idx: 0 for arm_idx in plan_list}
                for f_key, plan_set in f_keys_to_plans.items():
                    if len(plan_set) < len(plan_list):
                        exp = f_keys_to_dists[f_key][0]
                        for arm_idx in plan_set:
                            plan_exp[arm_idx] += exp
                unique_dict = {arm_idx: 0 for arm_idx in plan_list}
                best_arm = sorted(plan_exp.keys(), key=lambda x: plan_exp[x])[0]
                for f_key in f_keys_to_ranges:
                    plan_set = f_keys_to_plans[f_key]
                    for arm_idx in plan_set:
                        unique_dict[arm_idx] += sum([x[0] * x[1] for x in f_keys_to_ranges[f_key]])
                    # if best_arm in plan_set:
                    #     variable_utility += sum([x[0] * x[1] for x in f_keys_to_ranges[f_key]])
                for f_key in f_keys_constants:
                    plan_set = f_keys_to_plans[f_key]
                    for arm_idx in plan_set:
                        unique_dict[arm_idx] += f_keys_to_dists[f_key][0]
                    # if best_arm in plan_set:
                    #     variable_utility += f_keys_to_dists[f_key][0]
                # best_arm = 0
                best_utility = unique_dict[best_arm]
                optimal_arms.append(best_arm)
                print("Best arm:", best_arm, unique_dict, plan_exp)
            else:
                best_arm = None
                next_arms = {frozenset(optimal_arms).union([arm_idx]): 0
                             for arm_idx in plan_list if arm_idx not in optimal_arms}
                for next_arm in next_arms:
                    arm_ranges = [r for k, r in f_keys_to_ranges.items() if
                                  len(next_arm.intersection(f_keys_to_plans[k])) > 0]
                    all_combinations = list(itertools.product(*arm_ranges))
                    local_constant_cost = {arm_idx: 0 for arm_idx in next_arm}
                    for f_key in f_keys_constants:
                        for arm_idx in next_arm:
                            if arm_idx in f_keys_to_plans[f_key]:
                                local_constant_cost[arm_idx] += f_keys_to_dists[f_key][0]
                    for combination in all_combinations:
                        all_prob = 1
                        local_plan_cost = local_constant_cost.copy()
                        for value, prob, f_key in combination:
                            all_prob *= prob
                            for arm_idx in next_arm:
                                if arm_idx in f_keys_to_plans[f_key]:
                                    local_plan_cost[arm_idx] += value
                        min_plan = min(list(local_plan_cost.keys()), key=lambda k: local_plan_cost[k])
                        min_value = all_prob * local_plan_cost[min_plan]
                        next_arms[next_arm] += min_value
                next_arm_list = list(next_arms.keys())
                next_arm_list.reverse()
                local_best_arm = sorted(next_arm_list, key=lambda x: next_arms[x])[0]
                local_utility = next_arms[local_best_arm] # + self.penalty * len(optimal_arms)
                if self.verbose:
                    print(next_arms, local_utility, best_utility)
                if local_utility < best_utility:
                    best_utility = local_utility
                    best_arm = local_best_arm
                    optimal_arms = optimal_arms + [x for x in local_best_arm if x not in optimal_arms]

            if best_arm is None:
                break
        self.nr_selected_ims = float(best_utility) + sum([f_keys_to_dists[f_key][0]
                                                          for f_key, plan_set in f_keys_to_plans.items()
                                                          if len(plan_set) == len(plan_list)])
        return sorted(optimal_arms)

    def expected_minimum_normal_variables(self, removed_plans, top_k=3, nr_blocks=20):
        optimal_arms = []
        f_keys_to_plans = {}
        f_keys_to_dists = {}
        plan_trees = {k: v for k, v in self.plan_trees.items() if k not in removed_plans}
        full_set = frozenset(plan_trees.keys())
        plan_to_results = {}
        for arm_idx in plan_trees:
            plan = plan_trees[arm_idx].root
            plan_to_results[arm_idx] = 0
            for f_node in plan.f_nodes:
                f_key = f_node.f_key
                plan_to_results[arm_idx] += f_node.d_mean
                if f_key not in f_keys_to_plans:
                    f_keys_to_plans[f_key] = []
                    f_keys_to_dists[f_key] = (f_node.d_mean, f_node.d_std)
                    # f_keys_to_dists[f_key] = (f_node.d_mean, f_node.b_std)
                f_keys_to_plans[f_key].append(arm_idx)

        plan_sets_to_f_keys = {}
        plan_sets_to_dist = {}
        reduced_mean = 0
        for f_key in f_keys_to_plans:
            plan_key = frozenset(f_keys_to_plans[f_key])
            if plan_key != full_set:
                if plan_key not in plan_sets_to_f_keys:
                    plan_sets_to_f_keys[plan_key] = []
                    plan_sets_to_dist[plan_key] = [0, 0]
                plan_sets_to_f_keys[plan_key].append(f_key)
                plan_sets_to_dist[plan_key][0] += f_keys_to_dists[f_key][0]
                plan_sets_to_dist[plan_key][1] += (f_keys_to_dists[f_key][1] ** 2)
            else:
                reduced_mean += f_keys_to_dists[f_key][0]
        for plan_idx in plan_to_results:
            plan_to_results[plan_idx] -= reduced_mean
        for plan_key in plan_sets_to_dist:
            plan_sets_to_dist[plan_key][1] = plan_sets_to_dist[plan_key][1] ** 0.5

        large_keys = sorted(list(plan_sets_to_dist.keys()), key=lambda k: plan_sets_to_dist[k][1], reverse=True)[:top_k]
        plan_sets_to_ranges = {}
        for k in plan_sets_to_dist:
            distribution = NormalDist(mu=plan_sets_to_dist[k][0], sigma=plan_sets_to_dist[k][1])
            lower = int(plan_sets_to_dist[k][0] - 2 * plan_sets_to_dist[k][1])
            upper = plan_sets_to_dist[k][0] + 2 * plan_sets_to_dist[k][1]
            stride = max(int(np.round((upper - lower) / nr_blocks)), 1)
            # stride = 10
            ranges = np.arange(lower, upper, stride, dtype=int)
            cdf_list = np.array([distribution.cdf(x) for x in ranges])
            probs_list = cdf_list[1:] - cdf_list[:-1]
            probs_list = probs_list / np.sum(probs_list)
            value_list = (ranges[1:] + ranges[:-1]) / 2
            value_list = value_list.clip(min=0)
            plan_list = [k for _ in range(len(value_list))]
            plan_sets_to_ranges[k] = list(zip(value_list, probs_list, plan_list))

        constant_plan_cost = {arm_idx: 0 for arm_idx in plan_trees}
        for plan_key in plan_sets_to_dist:
            if plan_key not in large_keys:
                for plan_idx in plan_key:
                    constant_plan_cost[plan_idx] += plan_sets_to_dist[plan_key][0]

        best_utility = sys.maxsize
        if self.verbose:
            target_keys = [f_key for f_key in f_keys_to_plans if 1 in f_keys_to_plans[f_key] and 3 in f_keys_to_plans[f_key]]
            print({f_key: f_keys_to_dists[f_key] for f_key in target_keys})
            print("f keys to plans: ", f_keys_to_plans)
            print("f keys to dists: ", f_keys_to_dists)
            print("Plan sets to dists: ", plan_sets_to_dist)
        while len(optimal_arms) < min(3, len(plan_trees)):
            if len(optimal_arms) == 0:
                best_arm = sorted(plan_to_results.keys(), key=lambda x: plan_to_results[x])[0]
                optimal_arms.append(best_arm)
                variable_utility = 0
                for plan_key in plan_sets_to_ranges:
                    if best_arm in plan_key:
                        variable_utility += sum([x[0] * x[1] for x in plan_sets_to_ranges[plan_key]])
                best_utility = variable_utility
                if self.verbose:
                    print("Best arm:", best_arm, best_utility)
            else:
                best_arm = None
                next_arms = {frozenset(optimal_arms).union([arm_idx]): 0
                             for arm_idx in plan_trees if arm_idx not in optimal_arms}

                for next_arm in next_arms:
                    local_plan_sets_to_ranges = {k: v for k, v in plan_sets_to_ranges.items()
                                                 if len(k.intersection(next_arm)) > 0}
                    local_constant_cost = {arm_idx: 0 for arm_idx in next_arm}
                    variable_sets = set(local_plan_sets_to_ranges.keys())
                    for plan_key in local_plan_sets_to_ranges:
                        if next_arm.issubset(plan_key):
                            # local_constant_value = sum([x[0] * x[1] for x in local_plan_sets_to_ranges[plan_key]])
                            local_constant_value = plan_sets_to_dist[plan_key][0]
                            for arm_idx in local_constant_cost:
                                local_constant_cost[arm_idx] += local_constant_value
                            variable_sets.remove(plan_key)
                    local_sorted_keys = sorted(list(variable_sets), key=lambda k: plan_sets_to_dist[k][1], reverse=True)
                    local_large_keys = local_sorted_keys[:top_k]
                    for plan_key in local_sorted_keys[top_k:]:
                        # local_constant_value = sum([x[0] * x[1] for x in local_plan_sets_to_ranges[plan_key]])
                        local_constant_value = plan_sets_to_dist[plan_key][0]
                        for arm_idx in plan_key:
                            if arm_idx in local_constant_cost:
                                local_constant_cost[arm_idx] += local_constant_value
                    local_large_ranges = {k: local_plan_sets_to_ranges[k] for k in local_large_keys}
                    all_combinations = list(itertools.product(*list(local_large_ranges.values())))
                    for combination in all_combinations:
                        all_prob = 1
                        local_plan_cost = {arm_idx: cost for arm_idx, cost in local_constant_cost.items()
                                           if arm_idx in next_arm}
                        for value, prob, plan_key in combination:
                            if len(plan_key.intersection(next_arm)) > 0:
                                all_prob *= prob
                                for plan_idx in plan_key:
                                    if plan_idx in local_plan_cost:
                                        local_plan_cost[plan_idx] += value
                        min_plan = min(list(local_plan_cost.keys()), key=lambda k: local_plan_cost[k])
                        min_value = all_prob * local_plan_cost[min_plan]
                        next_arms[next_arm] += min_value
                local_best_arm = sorted(next_arms.keys(), key=lambda x: next_arms[x])[0]
                local_utility = next_arms[local_best_arm] + self.penalty * len(optimal_arms)
                if self.verbose:
                    print(next_arms, local_utility, best_utility)
                if local_utility < best_utility:
                    best_utility = local_utility
                    best_arm = local_best_arm
                    optimal_arms = optimal_arms + [x for x in local_best_arm if x not in optimal_arms]

            if best_arm is None:
                break
        return sorted(optimal_arms)

    def monte_carlo_method(self, removed_plans):
        optimal_arms = []
        best_arm = None
        best_utility = sys.maxsize
        intermediate_sampling = {}
        intermediate_dists = {}
        plan_sizes = {}
        total_samples = 1000
        # Compute the expectation via Monte Carlo methods
        for p in self.plan_trees:
            if p not in removed_plans:
                plan = self.plan_trees[p]
                plan_size = np.zeros(total_samples)
                for f_node in plan.root.f_nodes:
                    if f_node.f_key not in intermediate_sampling:
                        mu = f_node.d_mean
                        sigma = f_node.d_std
                        # sigma = f_node.b_std
                        a = np.random.normal(mu, sigma, total_samples)
                        a[a < 0] = 0
                        intermediate_sampling[f_node.f_key] = a
                        intermediate_dists[f_node.f_key] = (mu, sigma)
                        plan_size = plan_size + a
                    else:
                        plan_size = plan_size + intermediate_sampling[f_node.f_key]
                plan_sizes[p] = plan_size
        # print(intermediate_dists)
        while len(optimal_arms) < 3:
            for arm_idx in self.plan_trees:
                if arm_idx in optimal_arms or arm_idx in removed_plans:
                    continue
                current_arms = optimal_arms + [arm_idx]
                current_plans = [p for p in self.plan_trees if p in current_arms]
                # min_samples = np.min(np.array([plan_costs[p] for p in plan_costs if p in current_arms]), axis=0)
                min_sizes = np.min(np.array([plan_sizes[p] for p in plan_sizes if p in current_arms]), axis=0)
                # expected_speedup = (self.nr_threads / len(current_plans) + self.nr_threads - 2) / (self.nr_threads - 1)
                # utility = np.mean(min_samples)
                result_size = np.mean(min_sizes) + len(current_plans)
                print(current_arms, result_size)
                if result_size < best_utility:
                    best_utility = result_size
                    best_arm = arm_idx
            if best_arm is None:
                break
            optimal_arms.append(best_arm)
            best_arm = None

        return optimal_arms

    def monte_carlo_method_predicate(self, removed_plans):
        optimal_arms = []
        best_arm = None
        best_utility = sys.maxsize
        predicate_sampling = {}
        predicate_dists = {}
        plan_sizes = {}
        total_samples = 1000
        plan_trees = {k: v for k, v in self.plan_trees.items() if k not in removed_plans}
        print(plan_trees.keys())
        # Compute the expectation via Monte Carlo methods
        for p, plan in plan_trees.items():
            plan_size = np.zeros(total_samples)
            for f_key, p_dict in plan.p_visitor.intermediate_to_predicates.items():
                f_key_size = np.array([1] * total_samples)
                for p_key, p_dist in p_dict.items():
                    if p_dist[1] == 0:
                        f_key_size = f_key_size * p_dist[0]
                        continue
                    if p_key not in predicate_sampling:
                        mu = p_dist[0]
                        sigma = p_dist[1]
                        a = np.random.normal(mu, sigma, total_samples)
                        a[a < 0] = 0
                        a[a > 1] = 1
                        predicate_sampling[p_key] = a
                        predicate_dists[p_key] = (mu, sigma)
                        f_key_size = f_key_size * a
                    else:
                        f_key_size = f_key_size * predicate_sampling[p_key]
                plan_size += f_key_size
            plan_sizes[p] = plan_size

        while len(optimal_arms) < 3:
            for arm_idx in self.plan_trees:
                if arm_idx in optimal_arms or arm_idx in removed_plans:
                    continue
                current_arms = optimal_arms + [arm_idx]
                current_plans = [p for p in self.plan_trees if p in current_arms]

                min_sizes = np.min(np.array([plan_sizes[p] for p in plan_sizes if p in current_arms]), axis=0)
                expected_speedup = (self.nr_threads / len(current_plans) + self.nr_threads - 2) / (self.nr_threads - 1)
                result_size = np.mean(min_sizes) + len(current_plans)
                # print(current_arms, result_size)
                if result_size < best_utility:
                    best_utility = result_size
                    best_arm = arm_idx
            if best_arm is None:
                break
            optimal_arms.append(best_arm)
            best_arm = None

        return optimal_arms
