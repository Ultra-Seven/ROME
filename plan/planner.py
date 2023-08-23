import itertools
import sys
from itertools import chain, combinations
from statistics import NormalDist

import numpy as np
import gurobipy as gp
from gurobipy import GRB


class Planner:
    def __init__(self, plan_trees, postgres, nr_threads=24,
                 topk=3, nr_blocks=10,
                 method="max_results", max_para=3):
        self.nr_selected_ims = -1
        self.plan_trees = plan_trees
        self.postgres = postgres
        self.nr_threads = nr_threads
        self.policy = method
        self.optimized_policy = "monte_carlo"
        self.verbose = True
        self.topk = int(topk)
        self.max_para = max_para
        self.nr_blocks = int(nr_blocks)

    def optimal_arms(self, plan_trees):
        # return self.greedy_max_accurate_plan(plan_trees)
        if self.policy == "max_results":
            return self.maximize_unique_results(self.max_para)
        if self.policy == "max_im_ilp":
            return self.maximize_unique_im_ilp()
        if self.policy == "max_im_ilp_parallel":
            return self.maximize_unique_im_ilp(self.max_para)
        elif self.policy == "fix plan":
            return [5]
        elif self.policy == "probability_model":
            return self.probability_model(plan_trees)
        elif self.policy == "predicate":
            return self.predicate_model(plan_trees)
        else:
            return self.probability_model(plan_trees)

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
        while len(uncovered) > 0 and (max_parallelism == -1 or len(optimal_arms) < max_parallelism):
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

    def probability_model(self, plan_trees, nr_threads=24):
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
            optimal_arms = self.monte_carlo_method(removed_plans)
        else:
            optimal_arms = self.expected_minimum_normal_variables(removed_plans, self.topk, self.nr_blocks)
        return optimal_arms

    def predicate_model(self, plan_trees, nr_threads=24):
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
        else:
            optimal_arms = self.expected_minimum_normal_variables(removed_plans)
        return optimal_arms

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
                    # f_keys_to_dists[f_key] = (f_node.d_mean, f_node.d_std)
                    f_keys_to_dists[f_key] = (f_node.d_mean, f_node.b_std)
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
                local_utility = next_arms[local_best_arm] + 0.1 * len(optimal_arms)
                if self.verbose:
                    print(next_arms)
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
