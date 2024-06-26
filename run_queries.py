import argparse
import os
import signal
import sys
from itertools import combinations
from time import time, sleep
import subprocess

import psycopg2
from scipy.stats import chi2_contingency
import csv
import itertools

import numpy as np
import pandas as pd

from db.postgres import Postgres
from plan.plan_tree import PlanTree
from plan.planner import Planner
import psutil

from utils.pg_utils import create_connection
from utils.sql_utils import extract_tables, get_join_predicates

query_dir = sys.argv[1]
database = sys.argv[2]
output_file = sys.argv[3]
PG_CONNECTION_STR = f"dbname={database} user=postgres host=localhost"
nr_connections = 3
topk = sys.argv[4] if len(sys.argv) > 4 else 5
blocks = sys.argv[5] if len(sys.argv) > 5 else 5
uncertainty = 0.5
penalty = 0.01

_ALL_OPTIONS = [
    "enable_nestloop", "enable_hashjoin", "enable_mergejoin",
    "enable_seqscan", "enable_indexscan", "enable_indexonlyscan"
]
flag_combinations_list = list(itertools.product([0, 1], repeat=len(_ALL_OPTIONS)))

def get_args():
    parser = argparse.ArgumentParser(
        prog='ParaRQO',
        description='Robust Query Optimization with Parallel Execution')
    parser.add_argument('--query_dir', type=str, default='queries',
                        help='Directory containing the queries')
    parser.add_argument('--database', type=str, default='imdb',
                        help='database name')
    parser.add_argument('--output_file', type=str, default='test',
                        help='output file name')
    parser.add_argument('--topk', type=int, default=5,
                        help='top k plans to consider')
    parser.add_argument('--blocks', type=int, default=10,
                        help='number of blocks to consider')
    parser.add_argument('--uncertainty', type=float, default=0.5,
                        help='uncertainty to consider')
    parser.add_argument('--penalty', type=float, default=0.1,
                        help='penalty to consider')
    parser.add_argument('--method_name', type=str, default="pararqo",
                        help='method name')
    parser.add_argument('--nr_parallelism', type=int, default=3,
                        help='number of parallelism')
    parser.add_argument('--nr_runs', type=int, default=3,
                        help='number of experimental runs')
    parser.add_argument('--use_psql', type=int, default=0,
                        help='whether to use psql plan or not')
    parser.add_argument('--solver', type=str, default="greedy",
                        help='whether to use psql plan or not')
    parser.add_argument('--is_full', type=int, default=0,
                        help='whether to use psql plan or not')
    return parser.parse_args()


def _arm_idx_to_hints(arm_idx):
    hints = []
    for option in _ALL_OPTIONS:
        hints.append(f"SET {option} TO off;")

    if arm_idx == 0:
        for option in _ALL_OPTIONS:
            hints.append(f"SET {option} TO on;")
    elif arm_idx == 1:
        hints.append("SET enable_hashjoin TO on;")
        hints.append("SET enable_indexonlyscan TO on;")
        hints.append("SET enable_indexscan TO on;")
        hints.append("SET enable_mergejoin TO on;")
        hints.append("SET enable_seqscan TO on;")
    elif arm_idx == 2:
        hints.append("SET enable_hashjoin TO on;")
        hints.append("SET enable_indexonlyscan TO on;")
        hints.append("SET enable_nestloop TO on;")
        hints.append("SET enable_seqscan TO on;")
    elif arm_idx == 3:
        hints.append("SET enable_hashjoin TO on;")
        hints.append("SET enable_indexonlyscan TO on;")
        hints.append("SET enable_seqscan TO on;")
    elif arm_idx == 4:
        hints.append("SET enable_hashjoin TO on;")
        hints.append("SET enable_indexonlyscan TO on;")
        hints.append("SET enable_indexscan TO on;")
        hints.append("SET enable_nestloop TO on;")
        hints.append("SET enable_seqscan TO on;")
    elif arm_idx == 5:
        hints = []
    elif arm_idx == 6:
        return hints
    else:
        raise Exception("RegBlocker only supports the first 5 arms")
    return hints

def _arm_idx_to_perm_hints(arm_idx):
    flag_combination = flag_combinations_list[arm_idx]
    hints = [f"SET {f} TO {'on' if flag_combination[f_idx] == 1 else 'off'};"
             for f_idx, f in enumerate(_ALL_OPTIONS)]
    return hints


def parse_postgres_plan(plan):
    plan = plan[0]["Plan"]
    cost = plan["Total Cost"]
    return cost


def find_arms(sql, postgres, full=False,
              method_name="max_results", nr_parallelism=3,
              query_name="default", solver="greedy", conn=None):
    plan_trees = {}
    if full:
        nr_arms = 2 ** len(_ALL_OPTIONS)
    else:
        nr_arms = 6
    for x in range(nr_arms):
        if full:
            hints = _arm_idx_to_perm_hints(x)
        else:
            hints = _arm_idx_to_hints(x)
        new_sql = sql
        if not new_sql.endswith(";"):
            new_sql = new_sql + ";"

        # hints.append("SET max_parallel_workers_per_gather = 0;")
        target_sql = "\n".join(hints) + "\nEXPLAIN (COSTS, VERBOSE, FORMAT JSON) " + new_sql
        result = subprocess.run(['psql', '-h', 'localhost',
                                 '-U', 'postgres', '-d', database, '-XqAt', '-c', target_sql],
                                stdout=subprocess.PIPE)
        plan_trees[x] = PlanTree(sql, result, postgres,
                                 visualization=False, pid=x,
                                 query_name=query_name)
    planner = Planner(plan_trees, postgres, 24,
                      topk, blocks, method_name, nr_parallelism,
                      solver=solver, penalty=penalty, conn=conn)
    plan_start = time()
    optimal_arms = planner.optimal_arms(plan_trees)
    plan_end = time()
    return optimal_arms, plan_end - plan_start, planner.nr_selected_ims, planner


def check_arms(sql, postgres):
    plan_trees = {}
    alias_to_table = extract_tables(sql)
    for x in range(nr_arms):
        hints = _arm_idx_to_hints(x)
        new_sql = sql
        if not new_sql.endswith(";"):
            new_sql = new_sql + ";"

        target_sql = "SET max_parallel_workers_per_gather = 0;\n" \
                     "SET statement_timeout = 60000;\n" + \
                     "\n".join(hints) + "\nEXPLAIN (COSTS, VERBOSE, ANALYZE, FORMAT JSON) " + new_sql
        result = subprocess.run(['psql', '-h', 'localhost',
                                 '-U', 'postgres', '-d', database, '-XqAt', '-c', target_sql],
                                stdout=subprocess.PIPE)
        try:
            plan_trees[x] = PlanTree(sql, result, postgres, visualization=False)
        except:
            print(f"Time out for arm {x}")

    return plan_trees


def decode_arms_by_overlap(table_roots, candidate_arms):
    # Find the arm with the minimal cost
    min_cost = sys.maxsize
    min_arm = -1
    factor_keys = set()
    for x, plan_tree in table_roots.items():
        cost = plan_tree.root.total_cost
        if cost < min_cost:
            min_cost = cost
            min_arm = x
    candidate_arms.append(min_arm)
    for k in table_roots[min_arm].root.cost_variables.keys():
        factor_keys.add(k)
    while len(candidate_arms) < nr_connections:
        max_results = -1
        max_arm = -1
        for x, plan_tree in table_roots.items():
            if x not in candidate_arms:
                new_results = len([n for n in list(table_roots[x].root.cost_variables.keys()) if n not in factor_keys])
                if new_results > max_results:
                    max_results = new_results
                    max_arm = x
        candidate_arms.append(max_arm)
        for k in table_roots[max_arm].root.cost_variables.keys():
            factor_keys.add(k)
    candidate_arms = sorted(candidate_arms)
    return candidate_arms


def decide_arms_by_probs(table_roots, candidate_arms):
    # Find the arm with the minimal cost
    best_arms = None
    combination_list = list(combinations(list(range(nr_arms)), nr_connections))
    best_prob = -1
    for c in combination_list:
        all_wrong_prob = 1
        for arm_idx in c:
            plan_root = table_roots[arm_idx].root
            all_correct_prob = 1
            for f_node in plan_root.f_nodes:
                prob = f_node.get_prob()
                all_correct_prob = all_correct_prob * prob
            plan_wrong_prob = 1 - all_correct_prob
            all_wrong_prob = all_wrong_prob * plan_wrong_prob
        one_plan_correct = 1 - all_wrong_prob
        if one_plan_correct > best_prob:
            best_prob = one_plan_correct
            best_arms = c
    candidate_arms = sorted(list(best_arms))
    print(candidate_arms)
    return candidate_arms


def traverse_plan_node(node):
    # Add information to the list
    f_key = str(node.f_key) if node.f_key is not None else (str(node.filter) if hasattr(node, "filter")
                                                            else (str(node.relation_name)
                                                                  if hasattr(node, "relation_name") else "None"))
    infor = [str(node.node_type), f_key, str(node.plan_rows), str(node.actual_rows),
             str(node.total_cost), str(node.cost)]
    return_list = [infor]
    if len(node.children) == 0:
        return return_list
    else:
        for child in node.children:
            return_list += traverse_plan_node(child)
        return return_list


def run_query(sql, nr_threads, arms, full=False, use_psql=0, hint_dict=None):
    start = time()
    try:
        if not sql.endswith(";"):
            sql = sql + ";"

        procs_list = []
        for x in range(len(arms)):
            if hint_dict is None:
                if full:
                    hints = _arm_idx_to_perm_hints(arms[x])
                else:
                    hints = _arm_idx_to_hints(arms[x])
            else:
                hints = [hint_dict[arms[x]]]
            execute_sql = f"\nSET statement_timeout = 60000;\n"
            execute_sql = f"SET max_parallel_workers = {nr_threads};\n" \
                          f"SET max_parallel_workers_per_gather = {nr_threads};" \
                          f"\nSET statement_timeout = 60000;\n"
            # execute_sql = f"SET max_parallel_workers = 8;\n" \
            #               f"SET max_parallel_workers_per_gather = 2;" \
            #               f"\nSET statement_timeout = 60000;\n"
            execute_sql += ("\n".join(hints) + "\n")
            execute_sql += sql
            if hint_dict is not None:
                worker = subprocess.Popen(['psql', '-h', 'localhost', '-U', 'postgres', '-d',
                                           database, '-c', "LOAD 'pg_hint_plan';",
                                           '-c', execute_sql], stdout=subprocess.PIPE)
            else:
                worker = subprocess.Popen(['psql', '-h', 'localhost', '-U', 'postgres', '-d',
                                           database, '-c', execute_sql], stdout=subprocess.PIPE)
            print(f"Registering process {worker.pid}")
            procs_list.append(psutil.Process(worker.pid))
        if use_psql == 1:

            execute_sql = f"\nSET statement_timeout = 60000;\n"
            # execute_sql = f"SET max_parallel_workers = {nr_threads};\n" \
            #               f"SET max_parallel_workers_per_gather = {nr_threads};" \
            #               f"\nSET statement_timeout = 60000;\n"
            if len(arms) == 0:
                execute_sql += f"SET max_parallel_workers = {nr_threads};\n" \
                              f"SET max_parallel_workers_per_gather = {nr_threads};\n"
            execute_sql += sql
            worker = subprocess.Popen(['psql', '-h', 'localhost', '-U', 'postgres', '-d',
                                       database, '-c', execute_sql], stdout=subprocess.PIPE)
            print(f"Registering psql process {worker.pid}")
            procs_list.append(psutil.Process(worker.pid))

        def on_terminate(proc):
            print("process {} terminated".format(proc) + ": " + str(time() - start))
            for proc in procs_list:
                if psutil.pid_exists(proc.pid):
                    print(f"Killing process {proc.pid}")
                    os.kill(proc.pid, signal.SIGKILL)
        while True:
            gone, alive = psutil.wait_procs(procs_list, timeout=70, callback=on_terminate)
            if len(gone) > 0:
                stop = time()
                for alive_proc in alive:
                    print(alive_proc.pid)
                    os.kill(alive_proc.pid, signal.SIGKILL)
                break

        # print("Restarting: ", result.stdout.decode('utf-8'))
        remaining_procs = []
        for proces in psutil.process_iter():
            try:
                process_args_list = proces.cmdline()
                if any(["parallel worker" in x for x in process_args_list]):
                    remaining_procs.append(psutil.Process(proces.pid))
            except Exception as e:
                continue
        while True:
            gone, alive = psutil.wait_procs(remaining_procs, timeout=60, callback=on_terminate)
            for alive_proc in alive:
                os.kill(alive_proc.pid, signal.SIGKILL)
            if len(alive) == 0:
                break
        result = subprocess.Popen(['pg_ctl', '-D', '/export/pgdata/', 'restart'],
                                  stdout=subprocess.PIPE)
        remaining_procs = []
        for proces in psutil.process_iter():
            try:
                process_args_list = proces.cmdline()
                if any(["pg_ctl" in x for x in process_args_list]):
                    remaining_procs.append(psutil.Process(proces.pid))
            except Exception as e:
                continue
        while True:
            gone, alive = psutil.wait_procs(remaining_procs, timeout=60, callback=on_terminate)
            for alive_proc in alive:
                os.kill(alive_proc.pid, signal.SIGKILL)
            if len(alive) == 0:
                break
        print("Restarting done!")
    except Exception as e:
        print(e)
        stop = time()

    return stop - start


def run_bouquet_query(sql, arms, conn, scale=2.):
    start = time()
    nr_plans_check = 0
    success = 0
    sql = sql.strip()
    try:
        if not sql.endswith(";"):
            sql = sql + ";"
        max_len = min(len(arms), 7)
        timeout_list = [1000, 2000, 4000, 8000, 16000, 29000]
        if max_len == 1:
            timeout_list[max_len - 1] = 60000
        else:
            timeout_list[max_len - 1] = (60000 - sum(timeout_list[:(max_len-1)]))
        for idx in range(max_len):
            hints = arms[idx]
            timeout = timeout_list[idx]
            # execute_sql = f"\nSET statement_timeout = {timeout};\n"
            execute_sql = hints + "\n" + sql
            # execute_sql += sql
            nr_plans_check = idx + 1
            try:
                set_sql = (f"SET statement_timeout = {timeout};\nSET max_parallel_workers = 24;"
                           f"\nSET max_parallel_workers_per_gather = 24;")
                print("Running: " + hints)
                result = subprocess.run(['psql', '-h', 'localhost',
                                         '-U', 'postgres', '-d', database, '-XqAt',
                                         '-c', "LOAD 'pg_hint_plan';", '-c',
                                         set_sql, '-c', execute_sql],
                                        stdout=subprocess.PIPE)
                result_str = result.stdout.decode('utf-8')
                print("Results: " + result_str)
                if result_str.strip() == '':
                    continue
                else:
                    success = 1
                    print("Successfully executed query with hint: " + hints)
                    break
            except Exception as e:
                print(e)
                continue
        stop = time()
        def on_terminate(proc):
            print("process {} terminated".format(proc) + ": " + str(time() - start))
        remaining_procs = []
        for proces in psutil.process_iter():
            try:
                process_args_list = proces.cmdline()
                if any(["parallel worker" in x for x in process_args_list]):
                    remaining_procs.append(psutil.Process(proces.pid))
            except Exception as e:
                continue
        while True:
            gone, alive = psutil.wait_procs(remaining_procs, timeout=60, callback=on_terminate)
            for alive_proc in alive:
                os.kill(alive_proc.pid, signal.SIGKILL)
            if len(alive) == 0:
                break
        result = subprocess.Popen(['pg_ctl', '-D', '/export/pgdata/', 'restart'],
                                  stdout=subprocess.PIPE)
        remaining_procs = []
        for proces in psutil.process_iter():
            try:
                process_args_list = proces.cmdline()
                if any(["pg_ctl" in x for x in process_args_list]):
                    remaining_procs.append(psutil.Process(proces.pid))
            except Exception as e:
                continue
        while True:
            gone, alive = psutil.wait_procs(remaining_procs, timeout=60, callback=on_terminate)
            for alive_proc in alive:
                os.kill(alive_proc.pid, signal.SIGKILL)
            if len(alive) == 0:
                break
        print("Restarting done!")
    except Exception as e:
        print(e)
        stop = time()

    return stop - start, nr_plans_check, success


def run_baseline_query(sql, nr_threads):
    start = time()
    try:
        if not sql.endswith(";"):
            sql = sql + ";"

        procs_list = []
        hints = _arm_idx_to_hints(1)
        execute_sql = f"SET max_parallel_workers = {nr_threads};\n" \
                      f"SET max_parallel_workers_per_gather = {nr_threads};" \
                      f"\nSET statement_timeout = 60000;\n"
        execute_sql += ("\n".join(hints) + "\n")
        execute_sql += sql
        worker = subprocess.Popen(['psql', '-h', 'localhost', '-U', 'postgres', '-d',
                                   database, '-c', execute_sql], stdout=subprocess.PIPE)
        procs_list.append(psutil.Process(worker.pid))

        def on_terminate(proc):
            # print("process {} terminated".format(proc))
            for proc in procs_list:
                if psutil.pid_exists(proc.pid):
                    # print(f"Killing process {proc.pid}")
                    os.kill(proc.pid, signal.SIGKILL)
        while True:
            gone, alive = psutil.wait_procs(procs_list, timeout=60, callback=on_terminate)
            if len(gone) > 0:
                for alive_proc in alive:
                    print(alive_proc.pid)
                    os.kill(alive_proc.pid, signal.SIGKILL)
                break
    except Exception as e:
        print(e)
    stop = time()
    return stop - start


def benchmark(method_name, nr_parallelism, use_psql=False,
              nr_runs=3, solver="greedy", is_full=False):
    queries = []
    BASELINE = False
    postgres = Postgres("localhost", 5432, "postgres", "postgres", database)
    postgres.uncertainty = uncertainty
    postgres.nr_blocks = blocks
    if method_name == "create_db":
        create_larger_database(database, postgres, 4)
    postgres.close()
    for fp in os.listdir(query_dir):
        # if fp.endswith(".sql"):
        if fp.endswith(".sql") and fp in ["28a.sql"]:
            with open(os.path.join(query_dir, fp)) as f:
                query = f.read()
                queries.append((fp, query))
    if database == "so":
        unsupported_queries = ["q10-", "q13-", "q14-", "q15-", "q16-"]
        queries = [q for q in queries if not any([q[0].startswith(u) for u in unsupported_queries])]
    queries = sorted(queries, key=lambda x: x[0])
    print("Read", len(queries), "queries.")
    total_threads = 24
    is_full = is_full
    conn = None

    for x in range(nr_runs):
        with open(f"results/{output_file}_{x}.csv", 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(["Idx", "Threads", "Query", "Time", "Arms", "PlanTime", "Nr_IMs"])
            for idx, (fp, q) in enumerate(queries):
                query_name = fp.split("/")[-1].split(".")[0]
                if True:
                # try:
                    # run_query(q, total_threads, [4])
                    # run_query(q, total_threads, [5])
                    if BASELINE:
                        q_time = run_baseline_query(q, total_threads)
                        spamwriter.writerow([str(idx), 24, fp,
                                             str(q_time), "", 0, 0])
                        print(idx, 24, fp, q_time, "", 0, flush=True)
                    else:
                        postgres.set_sql_query(q, query_name)
                        if use_psql:
                            arms, plan_time, nr_ims, planner = find_arms(q, postgres, full=is_full,
                                                                method_name=method_name,
                                                                nr_parallelism=nr_parallelism-1,
                                                                query_name=query_name,
                                                                solver=solver)
                        else:
                            arms, plan_time, nr_ims, planner = find_arms(q, postgres, full=is_full,
                                                                method_name=method_name,
                                                                nr_parallelism=nr_parallelism,
                                                                query_name=query_name,
                                                                solver=solver, conn=conn)
                        if method_name == "bouquet":
                            q_time, nr_plans_check, success = run_bouquet_query(q, arms, conn, scale=1.5)
                            spamwriter.writerow([str(idx), success, fp,
                                                 str(q_time), str(len(arms)),
                                                 str(plan_time), str(nr_plans_check)])
                            print(idx, success, fp, q_time, str(len(arms)), str(nr_plans_check), flush=True)
                        elif method_name == "rome_bouquet":
                            hint_dict = {}
                            plan_trees = {}
                            for hint_idx, hint in enumerate(arms):
                                hint_dict[hint_idx] = hint
                                plan_trees[hint_idx] = PlanTree(postgres.sql, planner.bouquet_plans[hint], postgres,
                                                                visualization=False,
                                                                pid=hint_idx,
                                                                query_name=query_name)

                            new_planner = Planner(plan_trees, postgres, 24,
                                                  topk, blocks, "probability_model", nr_parallelism,
                                                  solver=solver, penalty=penalty, conn=conn)
                            optimal_arms = new_planner.optimal_arms(plan_trees)
                            arm_nr_threads = (total_threads // len(optimal_arms))
                            print(fp, optimal_arms)
                            # q_time = run_query(q, arm_nr_threads, arms, full=is_full, use_psql=use_psql)
                            q_time = run_query(q, arm_nr_threads, optimal_arms, full=is_full,
                                               use_psql=use_psql, hint_dict=hint_dict)
                            # q_time = run_baseline_query(q, total_threads)
                            # query_name = fp.split("/")[-1].split(".")[0]
                            spamwriter.writerow([str(idx), str(arm_nr_threads), fp,
                                                 str(q_time), "-".join([str(x) for x in optimal_arms]),
                                                 str(plan_time), str(nr_ims)])
                            print(idx, arm_nr_threads, fp, q_time, optimal_arms, nr_ims, flush=True)
                        elif method_name == "bouquet_topk":
                            hint_dict = {}
                            for hint_idx, hint in enumerate(arms):
                                hint_dict[hint_idx] = hint

                            optimal_arms = list(range(min(nr_parallelism, len(arms))))
                            arm_nr_threads = (total_threads // len(optimal_arms))
                            print(fp, optimal_arms)
                            q_time = run_query(q, arm_nr_threads, optimal_arms, full=is_full,
                                               use_psql=use_psql, hint_dict=hint_dict)
                            spamwriter.writerow([str(idx), str(arm_nr_threads), fp,
                                                 str(q_time), "-".join([str(x) for x in optimal_arms]),
                                                 str(plan_time), str(nr_ims)])
                            print(idx, arm_nr_threads, fp, q_time, optimal_arms, nr_ims, flush=True)
                        else:
                            arm_nr_threads = (total_threads // len(arms))
                            print(fp, arms)
                            # q_time = run_query(q, arm_nr_threads, arms, full=is_full, use_psql=use_psql)
                            q_time = run_query(q, arm_nr_threads, arms, full=is_full, use_psql=use_psql)
                            # q_time = run_baseline_query(q, total_threads)
                            # query_name = fp.split("/")[-1].split(".")[0]
                            spamwriter.writerow([str(idx), str(arm_nr_threads), fp,
                                                 str(q_time), "-".join([str(x) for x in arms]),
                                                 str(plan_time), str(nr_ims)])
                            print(idx, arm_nr_threads, fp, q_time, arms, nr_ims, flush=True)
                # except Exception as e:
                #     print(e)
                #     print(f"Unsupported query {fp}")
    if conn:
        conn.close()

def selectivity_analysis():
    query_dir = sys.argv[1]
    queries = []
    postgres = Postgres("localhost", 5432, "postgres", "postgres", database)
    postgres.close()
    for fp in os.listdir(query_dir):
        if fp.endswith(".sql"):
            with open(os.path.join(query_dir, fp)) as f:
                query = f.read()
                queries.append((fp, query))
    queries = sorted(queries, key=lambda x: x[0])
    print("Read", len(queries), "queries.")

    with open("results/selectivity_seqs.csv", 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(["Query", "Arm", "Operator", "Intermediate", "Plan_Row", "Row", "Plan_Cost", "Cost"])
        for idx, (fp, q) in enumerate(queries):
            plan_trees = check_arms(q, postgres)
            for plan_idx in plan_trees:
                plan_root = plan_trees[plan_idx].root
                nodes_list = traverse_plan_node(plan_root)
                query_name = fp.split("/")[-1].split(".")[0]
                for node in nodes_list:
                    spamwriter.writerow([query_name, str(plan_idx), node[0], node[1], node[2],
                                         node[3], node[4], node[5]])
            print(idx, fp, flush=True)


def bench_optimal(full=False):
    queries = []
    postgres = Postgres("localhost", 5432, "postgres", "postgres", database)
    postgres.close()
    if full:
        nr_arms = 2 ** len(_ALL_OPTIONS)
    else:
        nr_arms = 6
    for fp in os.listdir(query_dir):
        if fp.endswith(".sql"):
            with open(os.path.join(query_dir, fp)) as f:
                query = f.read()
                queries.append((fp, query))
    queries = sorted(queries, key=lambda x: x[0])
    print("Read", len(queries), "queries.")
    total_threads = 24

    with open(f"results/{database}_optimal.txt", 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(["Idx", "Threads", "Query", "Time", "Arm"])
        for idx, (fp, q) in enumerate(queries):
            arm_time_dict = {}
            for arm_idx in range(nr_arms):
                print(fp, arm_idx)
                q_time = run_query(q, total_threads, [arm_idx])
                arm_time_dict[arm_idx] = q_time
            optimal_time = min(arm_time_dict.values())
            optimal_arm = [key for key in arm_time_dict if arm_time_dict[key] == optimal_time]
            spamwriter.writerow([str(idx), str(total_threads), fp, str(optimal_time), optimal_arm[0]])
            print(idx, total_threads, fp, q_time, optimal_arm[0], flush=True)


def compare_cardinality(query_name):
    error_df = pd.read_csv(f"results/selectivity_seqs.csv", sep=",", quotechar="|")
    error_df = error_df[error_df["Query"] == query_name]
    error_df = error_df.groupby(["Arm"])
    for arm_idx, arm_df in error_df:
        arm_df = arm_df[(arm_df["Operator"] == "Hash Join") | (arm_df["Operator"] == "Nested Loop")
                        | (arm_df["Operator"] == "Merge Join")]
        actual_rows = 0
        plan_rows = 0
        for intermediate, i_df in arm_df.groupby(["Intermediate"]):
            actual_rows += i_df["Row"].iloc[0]
            plan_rows += i_df["Plan_Row"].iloc[0]
        print(arm_idx, actual_rows, plan_rows)


def check_statistics():
    postgres = Postgres("localhost", 5432, "postgres", "postgres", database)
    ci_person_id = postgres.stats_table["cast_info"]["person_id"]
    histogram_bound = [int(x) for x in ci_person_id["histogram_bounds"][1:-1].split(",")]
    mi_movie_id = postgres.stats_table["movie_info"]["movie_id"]
    cc_movie_id = postgres.stats_table["complete_cast"]["movie_id"]
    mk_keyword_id = postgres.stats_table["movie_keyword"]["keyword_id"]

    mi_info_type_id = postgres.stats_table["movie_info"]["info_type_id"]
    it1_id = postgres.stats_table["info_type"]["id"]

    cur = postgres.connection.cursor()
    cur.execute("SELECT mi.movie_id, mi.info_type_id FROM movie_info AS mi "
                "WHERE mi.note like '%internet%' AND mi.info like 'USA:% 200%'")
    results = cur.fetchall()
    results = [(t[0], t[1]) for t in results]

    postgres.close()


def variance_analysis():
    stats_df = pd.read_csv(f"results/selectivity_seqs.csv", sep=",", quotechar="|")
    actual_rows = np.array(stats_df["Row"])
    plan_rows = np.array(stats_df["Plan_Row"])
    errors = actual_rows / plan_rows
    stats_df["Error"] = list(errors)
    query_df = stats_df.groupby(["Query"])
    selectivity_list = []
    plan_selectivity_list = []
    for query, q_df in query_df:
        selectivity = []
        plan_selectivity = []
        with open(os.path.join(query_dir, (query + ".sql"))) as f:
            sql = f.read()
            alias_to_tables = extract_tables(sql)
        for index, row in q_df.iterrows():
            if row["Operator"] in ["Hash Join", "Nested Loop", "Merge Join"]:
                intermediate_key = row["Intermediate"]
                left_tables = intermediate_key.split(":")[0].split("-")
                right_tables = intermediate_key.split(":")[1].split("-")
                if len(left_tables) == 1:
                    left_table = left_tables[0]
                    check_str = "\(" + left_table + "." + "|" + alias_to_tables[left_table]
                    scan_df = q_df[(q_df["Operator"] == "Seq Scan")]
                    scan_df = scan_df[scan_df["Intermediate"].str.contains(check_str)]
                    left_card = scan_df["Row"].iloc[0] if len(scan_df) > 0 else None
                    left_plan_card = scan_df["Plan_Row"].iloc[0] if len(scan_df) > 0 else None
                else:
                    left_card = None
                    for index_2, row_2 in q_df.iterrows():
                        intermediate_key_2 = row_2["Intermediate"]
                        intermediate_set = frozenset(intermediate_key_2.replace(":", "-").split("-"))
                        if intermediate_set == frozenset(left_tables):
                            left_card = row_2["Row"]
                            left_plan_card = row_2["Plan_Row"]
                            break
                if len(right_tables) == 1:
                    right_table = right_tables[0]
                    check_str = "\(" + right_table + "." + "|" + alias_to_tables[right_table]
                    scan_df = q_df[(q_df["Operator"] == "Seq Scan")]
                    scan_df = scan_df[scan_df["Intermediate"].str.contains(check_str)]
                    right_card = scan_df["Row"].iloc[0] if len(scan_df) > 0 else None
                    right_plan_card = scan_df["Plan_Row"].iloc[0] if len(scan_df) > 0 else None
                else:
                    right_card = None
                    for index_2, row_2 in q_df.iterrows():
                        intermediate_key_2 = row_2["Intermediate"]
                        intermediate_set = frozenset(intermediate_key_2.replace(":", "-").split("-"))
                        if intermediate_set == frozenset(right_tables):
                            right_card = row_2["Row"]
                            right_plan_card = row_2["Plan_Row"]
                            break
                if left_card is not None and right_card is not None and left_card > 0 and right_card > 0:
                    selectivity.append(row["Row"] / (right_card * left_card))
                    plan_selectivity.append(row["Plan_Row"] / (right_plan_card * left_plan_card))
                elif left_card == 0 or right_card == 0:
                    selectivity.append(1)
                    plan_selectivity.append(1)
                else:
                    selectivity.append(-1)
                    plan_selectivity.append(-1)
            else:
                selectivity.append(1)
                plan_selectivity.append(1)

        selectivity_list += selectivity
        plan_selectivity_list += plan_selectivity
    stats_df["Selectivity"] = selectivity_list
    stats_df["Plan_Selectivity"] = plan_selectivity_list
    variance_df = stats_df[((stats_df["Operator"] == "Hash Join") | (stats_df["Operator"] == "Nested Loop")
                           | (stats_df["Operator"] == "Merge Join")) & (stats_df["Selectivity"] != -1)]
    variance_df = variance_df.groupby(["Intermediate"])
    for intermediate, i_df in variance_df:
        if len(i_df) == 1:
            selectivity_var = -1
            plan_selectivity_var = -1
        else:
            selectivity_var = i_df["Selectivity"].var()
            plan_selectivity_var = i_df["Plan_Selectivity"].var()
        print(intermediate, selectivity_var, i_df["Error"].mean())


def correlation_analysis(table_name):
    postgres = Postgres("localhost", 5432, "postgres", "postgres", database)

    cur = postgres.connection.cursor()
    cur.execute(f"SELECT * FROM {table_name} AS mi")
    columns = [c.name for c in cur.description]
    column_data = {c.name: [] for c in cur.description if c.type_code in [23, 16, 1082, 1083, 1700] and c.name != "id"}
    results = cur.fetchall()
    for t in results:
        for x in range(len(t)):
            if columns[x] in column_data:
                if t[x] is None:
                    column_data[columns[x]].append(-1)
                else:
                    column_data[columns[x]].append(t[x])
    comb = combinations(list(column_data.keys()), 2)
    data = pd.DataFrame(column_data)
    for c in comb:
        # correlation = spearmanr(column_data[c[0]], column_data[c[1]])
        # print(c, correlation)
        # Cross tabulation between GENDER and APPROVE_LOAN
        crosstab_result = pd.crosstab(index=data[c[0]], columns=data[c[1]])

        # Performing Chi-sq test
        chisq_result = chi2_contingency(crosstab_result)
        print(c, chisq_result)

    postgres.close()


def stress_analysis():
    # flags = ["enable_hashjoin", "enable_mergejoin", "enable_nestloop",
    #              "enable_material", "enable_partitionwise_join",
    #              "enable_bitmapscan", "enable_memoize", "enable_parallel_hash",
    #              "enable_seqscan", "enable_gathermerge", "enable_indexonlyscan",
    #              "enable_indexscan"]

    flags = _ALL_OPTIONS

    queries = []
    postgres = Postgres("localhost", 5432, "postgres", "postgres", database)
    postgres.close()
    for fp in os.listdir(query_dir):
        if fp.endswith(".sql"):
            with open(os.path.join(query_dir, fp)) as f:
                query = f.read()
                queries.append((fp, query))
    queries = sorted(queries, key=lambda x: x[0])
    lst = list(itertools.product([0, 1], repeat=len(flags)))
    print("Read", len(queries), "queries.")
    overlapped_list = []
    for idx, (fp, q) in enumerate(queries):
        postgres.set_sql_query(q, fp.split("/")[-1].split(".")[0])
        arm_im_results = {}
        covered_im_results = {}
        for arm_idx, flag_combination in enumerate(lst):
            flag_settings = [f"SET {f} TO {'on' if flag_combination[f_idx] == 1 else 'off'};"
                             for f_idx, f in enumerate(flags)]
            new_sql = q
            if not new_sql.endswith(";"):
                new_sql = new_sql + ";"

            # hints.append("SET max_parallel_workers_per_gather = 0;")
            target_sql = "\n".join(flag_settings) + "\nEXPLAIN (COSTS, VERBOSE, FORMAT JSON) " + new_sql
            result = subprocess.run(['psql', '-h', 'localhost',
                                     '-U', 'postgres', '-d', database, '-XqAt', '-c', target_sql],
                                    stdout=subprocess.PIPE)
            # if x == 1 or x == 4:
            #     plan_trees[x] = PlanTree(result, postgres, visualization=False)
            plan_tree = PlanTree(q, result, postgres, visualization=False)
            f_nodes = set([f_node.f_key for f_node in plan_tree.root.f_nodes])
            same_plan = False
            for arm_im_idx, im_result in arm_im_results.items():
                if f_nodes.issubset(im_result):
                    same_plan = True
                    break
            if not same_plan:
                arm_im_results[arm_idx] = f_nodes
                for im in f_nodes:
                    if im not in covered_im_results:
                        covered_im_results[im] = []
                    covered_im_results[im].append(arm_idx)
        nr_overlapped = len([im for im in covered_im_results if len(covered_im_results[im]) > 1])
        overlapped_list.append(nr_overlapped)
    print(np.mean(nr_overlapped))


def create_larger_database(db_name, postgres, nr_times):
    new_db_name = f"{db_name}_{nr_times}"
    new_pg_conn = psycopg2.connect(
        dbname=new_db_name,
        user="postgres",
        password="postgres",
        host="localhost",
        port=5432
    )
    for table_name in postgres.table_sizes:
        cur = new_pg_conn.cursor()
        pg_cur = postgres.connection.cursor()
        pg_cur.execute("SELECT * FROM " + table_name + ";")
        results = pg_cur.fetchall()
        start_row = 1
        for x in range(nr_times):
            for idx, r in enumerate(results):
                new_r = list(r)
                new_r[0] = start_row
                new_r = [f"$${v.replace('$', '')}$$" if isinstance(v, str) else
                         ("NULL" if v is None else str(v)) for v in new_r]
                new_r_str = "(" + (", ".join(new_r)) + ")"
                new_sql = "INSERT INTO " + table_name + " VALUES " + new_r_str
                cur.execute(new_sql)
                start_row += 1
        print("Finished", table_name)
        new_pg_conn.commit()
    sys.exit(0)


def intermediate_result(sql_path):
    sql = open(sql_path).read()
    sql = sql.replace(";", "")
    postgres = Postgres("localhost", 5432, "postgres", "postgres", database)
    postgres.set_sql_query(sql, sql_path.split("/")[-1].split(".")[0])
    new_sql = "SELECT COUNT(*)"
    alias_set = {"mc",  "cn", "miidx", "it", "ct", "mi", "it2", "t", "kt"}
    predicates = sql.split("WHERE")[-1].strip().split(" AND ")
    join_predicates = get_join_predicates(sql)
    unary_predicates = [p for p in predicates if p not in join_predicates]
    table_clause = [postgres.alias_to_tables[a] + " AS " + a for a in alias_set]
    new_sql += " FROM " + ", ".join(table_clause)
    new_predicates = []
    for unary_predicate in unary_predicates:
        u_alias = unary_predicate.split(" ")[0].split(".")[0]
        if u_alias in alias_set:
            new_predicates.append(unary_predicate)
    for join_predicate in join_predicates:
        left_alias = join_predicate.split(" ")[0].split(".")[0]
        right_alias = join_predicate.split(" ")[2].split(".")[0]
        if left_alias in alias_set and right_alias in alias_set:
            new_predicates.append(join_predicate)
    new_sql += " WHERE " + " AND ".join(new_predicates)
    cur = postgres.connection.cursor()
    cur.execute(new_sql)
    results = cur.fetchone()[0]
    print(new_sql)
    print(results)
    postgres.close()


if __name__ == "__main__":
    args = get_args()
    query_dir = args.query_dir
    database = args.database
    output_file = args.output_file
    PG_CONNECTION_STR = f"dbname={database} user=postgres host=localhost"
    nr_connections = 3
    topk = args.topk
    blocks = args.blocks
    uncertainty = args.uncertainty
    penalty = args.penalty
    benchmark(args.method_name, args.nr_parallelism, args.use_psql,
              args.nr_runs, args.solver, args.is_full)
    # intermediate_result(f"{query_dir}/13a.sql")
    # bench_optimal()
    # selectivity_analysis()
    # compare_cardinality("28c")
    # check_statistics()
    # variance_analysis()
    # correlation_analysis("cast_info")
    # stress_analysis()
