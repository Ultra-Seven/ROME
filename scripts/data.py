import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as st

from utils.sql_utils import get_join_predicates

query_list = ["q1-", "q2-", "q3-", "q4-", "q5-", "q6-", "q7-", "q8-", "q9-", "q11-", "q12-"]

def check_valid(s):
    for q in query_list:
        if s.startswith(q):
            return True
    return False


def mean_confidence_interval(data, confidence=0.95):
    h = st.t.interval(confidence, len(data)-1, loc=np.mean(data), scale=st.sem(data))
    return h[1] - np.mean(data)


def performance_evaluation():
    nr_runs = 3
    methods = ["greedy_intermediate_results",
               "greedy_intermediate_results_para3",
               "greedy_intermediate_results_2",
               "fix_plan_0",
               "default_imdb_fix_plan_0",
               "fix_plan_1",
               "fix_plan_2",
               "fix_plan_3",
               "fix_plan_4",
               "fix_plan_5",
               "greedy_predicate_model",
               "greedy_result_model",
               "greedy_cardmodel_mc_e0.01",
               "greedy_cardmodel_mc_e0.1",
               "greedy_cardmodel_mc_e0.2",
               "greedy_cardmodel_mc_e0.5",
               "greedy_cardmodel_mc_e0.8",
               "greedy_cardmodel_mc_e1",
               "greedy_cardmodel_mc_e10",

               "greedy_cardmodel_mc_p0.1",
               "greedy_cardmodel_mc_p2",
               "greedy_cardmodel_mc_p5",
               "greedy_cardmodel_mc_p10",
               "greedy_cardmodel_k1b10",
               "greedy_cardmodel_k3b10",
               "greedy_cardmodel_k5b10",
               "greedy_cardmodel_k7b10",
               "greedy_cardmodel_k3b5",
               "greedy_cardmodel_k3b20",
               "greedy_cardmodel_k3b30",
               "greedy_cardmodel_k3b50",
               # "greedy_cardmodel_k1b10p0.5",
               # "greedy_cardmodel_k3b10p0.5",
               # "greedy_cardmodel_k5b10p0.5",
               # "greedy_cardmodel_k7b10p0.5",
               # "greedy_cardmodel_k3b5p0.5",
               # "greedy_cardmodel_k3b50p0.5",
               # "greedy_cardmodel_k3b100p0.5",
               #
               # "greedy_cardsq_k5b10e0.5",
               # "greedy_cardsq_k5b10e1",
               # "greedy_cardsq_k5b10e10",
               #
               # "greedy_imresults",
               # "greedy_imresults_ilp",
               # "greedy_imresults_ilp_para3",
               # "greedy_imresults_ilp_para8"，
               "greedy_combine_k1b10",
               "greedy_combine_k3b10",
               "greedy_combine_k5b10",
               "greedy_combine_k7b10",
               "greedy_combine_k3b5",
               "greedy_combine_k3b20",
               "greedy_combine_k3b30",
               "greedy_combine_k3b50",

               "greedy_combine_k3b10u001",
               "greedy_combine_k3b10u01",
               "greedy_combine_k3b10u05",
               "greedy_combine_k3b10u1",
               "greedy_combine_k3b10u10",
               "greedy_combine_k3b10p001",
               "greedy_combine_k3b10p01",
               "greedy_combine_k3b10p1",
               "greedy_combine_k3b10p5",
               "greedy_combine_k3b10p10",

               "greedy_so_combine_k5b10",
               "greedy_so_combine_mpd",

               "greedy_predicate_k1b10",

               # Vary parameters
               # "greedy_poirange_k1b10",
               # "greedy_poirange_k3b10",
               # "greedy_poirange_k5b10",
               # "greedy_poirange_k7b10",
               # "greedy_poirange_k3b5",
               # "greedy_poirange_k3b20",
               # "greedy_poirange_k3b30",
               # "greedy_poirange_k3b50",
               # "greedy_imdb_poirange_mpd",

               "greedy_poirange_para1_mpd",
               "greedy_poirange_para2_mpd",
               "greedy_poirange_para4_mpd",
               "greedy_poirange_para5_mpd",
               "greedy_poirange_para6_mpd",
               "greedy_poirange_para1_pm",
               "greedy_poirange_para2_pm",
               "greedy_poirange_para4_pm",
               "greedy_poirange_para5_pm",
               "greedy_poirange_para6_pm",
               "greedy_top3_least",
               "imdb_bouquet",
               "imdb_bouquet_topk",
               "imdb_rome_bouquet"
               ]
    data = {}
    for method in methods:
        data[method] = [pd.read_csv("/Users/tracy/Documents/Research/bak_ParaRQO/results/performance/"
                                    + method + "_" + str(run) + ".csv")
                        for run in range(nr_runs)]
    optimal = [{"Time": np.min(np.array([data["fix_plan_" + str(0)][run]["Time"],
                                         data["fix_plan_" + str(1)][run]["Time"],
                                         data["fix_plan_" + str(2)][run]["Time"],
                                         data["fix_plan_" + str(3)][run]["Time"],
                                         data["fix_plan_" + str(4)][run]["Time"],
                                         data["fix_plan_" + str(5)][run]["Time"]]),
                               axis=0)} for run in range(nr_runs)]
    with open("/Users/tracy/Documents/Research/bak_ParaRQO/results/performance/imdb_optimal.csv", "w") as f:
        f.write("Idx,Threads,Query,Time,Arm\n")
        queries = list(data["fix_plan_0"][0]["Query"])
        for i, q in enumerate(queries):
            performance_list = [data["fix_plan_" + str(x)][0]["Time"][i] for x in range(6)]
            min_idx = np.argmin(performance_list)
            f.write(f"{i},24,{q},{np.min(performance_list)},{min_idx}\n")

    data["imdb_optimal"] = optimal
    print(f"Method\t\t\t\tPerformance")
    for method in data:
        performance = np.array([data[method][x]["Time"] for x in range(nr_runs)])
        avg_performance = np.mean(performance, axis=0)
        avg_value = np.sum(avg_performance)
        sum_performance = np.sum(performance, axis=1)
        h = mean_confidence_interval(sum_performance)
        method_name = "_".join(method.split("_")[1:])
        print(f"{method_name},{avg_value},{h}")

    with open("/Users/tracy/Documents/Research/bak_ParaRQO/results/query_performance.csv", "w") as f:
        f.write("Query,Method,Performance,Interval\n")
        for method in data:
            if method in ["imdb_optimal", "greedy_combine_k3b10"]:
                performance = np.array([data[method][x]["Time"] for x in range(nr_runs)])
                interval = np.array([mean_confidence_interval(y) for y in performance.T])
                queries = list(data["fix_plan_0"][0]["Query"])
                avg_performance = np.mean(performance, axis=0)

                method_name = "_".join(method.split("_")[1:])
                for idx, p in enumerate(avg_performance):
                    query_name = queries[idx].split(".")[0]
                    f.write(f"{query_name},{method_name},{p},{interval[idx]}\n")

                # query_names = ",".join([q.split(".")[0] for m, q in enumerate(queries) if m % 10 == 0])
                #
                # print(query_names)


def time_vs_quality():
    # methods = ["greedy_imresults", "greedy_imresults_ilp_para3", "greedy_imresults_ilp_para8"]
    # methods = ["greedy_full_ilppara3", "greedy_full_ilppara8",
    #            "greedy_full_imresultspara3", "greedy_full_imresultspara8",
    #            "greedy_psql_ilppara2", "greedy_psql_ilppara4", "greedy_psql_ilppara6", "greedy_psql_ilppara8",
    #            "greedy_psql_imresultspara2", "greedy_psql_imresultspara4", "greedy_psql_imresultspara6",
    #            "greedy_psql_imresultspara8"]
    model_name = "pm"
    if model_name == "pm":
        methods = ["greedy_pm_ilppara2", "greedy_pm_ilppara4",
                   "greedy_pm_ilppara6", "greedy_pm_ilppara8",
                   "greedy_poirange_greedypara2", "greedy_poirange_greedypara4",
                   "greedy_poirange_greedypara6", "greedy_poirange_greedypara8"
                   ]
    else:
        methods = ["greedy_psql_ilppara2", "greedy_psql_ilppara4",
                   "greedy_psql_ilppara6", "greedy_psql_ilppara8",
                   "greedy_psql_imresultspara2", "greedy_psql_imresultspara4",
                   "greedy_psql_imresultspara6", "greedy_psql_imresultspara8"
                   ]
    data = {}
    nr_runs = 3
    for method in methods:
        data[method] = [pd.read_csv("/Users/tracy/Documents/Research/bak_ParaRQO/results/performance/"
                                    + method + "_" + str(run) + ".csv")
                        for run in range(nr_runs)]
    print(f"Method\tPerformance\tInterval\tIMs Quality\tPlan Time\tNr Plans")
    perf_dict = {"Greedy": {}, "ILP": {}}
    for method in data:
        sys_name = "ILP" if "ilp" in method else "Greedy"
        parallelism = int(method.split("para")[-1])
        performance = np.array([data[method][x]["Time"] for x in range(nr_runs)])
        avg_performance = np.mean(performance, axis=0)
        avg_value = np.sum(avg_performance)
        sum_performance = np.sum(performance, axis=1)
        h = mean_confidence_interval(sum_performance)
        method_name = "_".join(method.split("_")[1:])
        quality = np.array([data[method][x]["Nr_IMs"] for x in range(nr_runs)])
        nr_plans = np.array([[len(str(a).split("-")) for a in data[method][x]["Arms"]] for x in range(nr_runs)])
        avg_nr_plans = np.mean(np.mean(nr_plans, axis=0))

        plan_time = np.array([data[method][x]["PlanTime"] for x in range(nr_runs)])
        avg_quality = np.mean(np.mean(quality, axis=0))
        avg_plan_time = np.mean(plan_time, axis=0)
        all_plan_time = np.sum(avg_plan_time)
        if parallelism not in perf_dict[sys_name]:
            perf_dict[sys_name][parallelism] = {}
        perf_dict[sys_name][parallelism]["Performance"] = avg_value
        perf_dict[sys_name][parallelism]["Quality"] = avg_quality
        perf_dict[sys_name][parallelism]["Planning"] = all_plan_time
        perf_dict[sys_name][parallelism]["Interval"] = h
        print(f"{method_name}\t{avg_value}\t{h}\t{avg_quality}\t{all_plan_time}\t{avg_nr_plans}")
    for x in [2, 4, 6, 8]:
        base = perf_dict["ILP"][x]["Quality"]
        for sys_name in ["ILP", "Greedy"]:
            perf_dict[sys_name][x]["Quality"] = abs(float(perf_dict[sys_name][x]["Quality"] - base))

    with open(f"/Users/tracy/Documents/Research/bak_ParaRQO/results/performance/quality_{model_name}.csv", "w") as f:
        f.write("Method,Parallelism,Performance,Quality,Planning,Interval\n")
        for sys_name in ["ILP", "Greedy"]:
            for x in [2, 4, 6, 8]:
                f.write(f"{sys_name},{x},{perf_dict[sys_name][x]['Performance']},"
                        f"{perf_dict[sys_name][x]['Quality']},"
                        f"{perf_dict[sys_name][x]['Planning']},"
                        f"{perf_dict[sys_name][x]['Interval']}\n")

def im_analysis_per_query():
    directory = "figs/"
    queries = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a directory
        if os.path.isdir(f):
            queries.append(filename)
    queries = sorted(queries)
    im_optimal = pd.read_csv("results/im_optimal_0.csv")
    para_rqo = pd.read_csv("/Users/tracy/Documents/Research/bak_ParaRQO/results/performance"
                           "/greedy_cardmodel_k5b10_1.csv")
    fix_time_dict = {
        "0": pd.read_csv("/Users/tracy/Documents/Research/bak_ParaRQO/results/performance"
                        "/fix_plan_0_1.csv"),
        "1": pd.read_csv("/Users/tracy/Documents/Research/bak_ParaRQO/results/performance"
                        "/fix_plan_1_1.csv"),
        "2": pd.read_csv("/Users/tracy/Documents/Research/bak_ParaRQO/results/performance"
                        "/fix_plan_2_1.csv"),
        "3": pd.read_csv("/Users/tracy/Documents/Research/bak_ParaRQO/results/performance"
                        "/fix_plan_3_1.csv"),
        "4": pd.read_csv("/Users/tracy/Documents/Research/bak_ParaRQO/results/performance"
                        "/fix_plan_4_1.csv"),
        "5": pd.read_csv("/Users/tracy/Documents/Research/bak_ParaRQO/results/performance"
                        "/fix_plan_5_1.csv")
    }
    with open("results/im_analysis.csv", "w") as f:
        f.write("Query,Sys,EC,TC\n")
        # Default method is plan = 0
        for query in queries:
            try:
                # Default method is plan = 0
                default_path = os.path.join(directory, query, "im_0.txt")
                default_arm = pd.read_csv(default_path, sep="|", header=None)
                default_arm.columns = ["IM", "EC", "TC"]
                default_tcs = sum(list(default_arm["TC"]))
                default_ecs = sum(list(default_arm["EC"]))
                # IM optimal
                im_idx = im_optimal[im_optimal["Query"] == query+".sql"]["Arms"].values[0]
                im_optimal_path = os.path.join(directory, query, f"im_{im_idx}.txt")
                im_optimal_arm = pd.read_csv(im_optimal_path, sep="|", header=None)
                im_optimal_arm.columns = ["IM", "EC", "TC"]
                im_optimal_tcs = sum(list(im_optimal_arm["TC"]))
                im_optimal_ecs = sum(list(im_optimal_arm["EC"]))
                # ParaRQO
                para_idx_list = para_rqo[para_rqo["Query"] == query+".sql"]["Arms"].values[0].split("-")
                best_arm_idx = -1
                best_time = sys.maxsize
                for para_idx in para_idx_list:
                    fix_time = fix_time_dict[para_idx][fix_time_dict[para_idx]["Query"] == query+".sql"]["Time"].values[0]
                    if fix_time < best_time:
                        best_time = fix_time
                        best_arm_idx = para_idx
                best_path = os.path.join(directory, query, f"im_{best_arm_idx}.txt")
                best_arm = pd.read_csv(best_path, sep="|", header=None)
                best_arm.columns = ["IM", "EC", "TC"]
                best_tcs = sum(list(best_arm["TC"]))
                best_ecs = sum(list(best_arm["EC"]))
                f.write(f"{query},Postgres,1,1\n")
                f.write(f"{query},OIM,{im_optimal_ecs/default_ecs},{im_optimal_tcs/default_tcs}\n")
                f.write(f"{query},ParaRQO,{best_ecs/default_ecs},{best_tcs/default_tcs}\n")
            except Exception as e:
                print(query)


def analyze_per_query_performance():
    data_df = pd.read_csv("/Users/tracy/Documents/Research/bak_ParaRQO/results/query_performance.csv")
    queries = sorted(list(set(data_df["Query"])))
    pm_df = data_df[data_df["Method"].str.contains("poirange_k3b10")]
    optimal_df = data_df[data_df["Method"].str.contains("optimal")]
    bao_df = data_df[data_df["Method"].str.contains("BAO1")]
    pm_performance = pm_df["Performance"].values
    optimal_performance = optimal_df["Performance"].values
    bao_performance = bao_df["Performance"].values

    optimal_ratio = pm_performance / optimal_performance
    bao_ratio = pm_performance / bao_performance
    optimal_regression = 0
    optimal_improvement = 0
    optimal_same = 0
    bao_regression = 0
    bao_improvement = 0
    bao_same = 0
    bound = 0.2
    with open("/Users/tracy/Documents/Research/bak_ParaRQO/results/query_performance_ratio.csv", "w") as f:
        f.write("Query,Method,Ratio\n")
        for idx, query in enumerate(queries):
            f.write(f"{query},Optimal,{optimal_ratio[idx]}\n")
            if optimal_ratio[idx] > 1 + bound:
                optimal_regression += 1
            elif optimal_ratio[idx] < 1 - bound:
                optimal_improvement += 1
            else:
                optimal_same += 1
        for idx, query in enumerate(queries):
            f.write(f"{query},BAO,{bao_ratio[idx]}\n")
            if bao_ratio[idx] > 1 + bound:
                bao_regression += 1
            elif bao_ratio[idx] < 1 - bound:
                bao_improvement += 1
            else:
                bao_same += 1

    bin_edges = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3, 3.5])
    hist, bin_edges = np.histogram(optimal_ratio, bins=bin_edges)
    total = sum(hist)
    hist = [h / total for h in hist]
    hist.append(0)
    print(list(zip(bin_edges, hist)))
    bin_edges = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    hist, bin_edges = np.histogram(bao_ratio, bins=bin_edges)
    hist = [h / total for h in hist]
    hist.append(0)
    print(list(zip(bin_edges, hist)))

    print(f"Optimal stats: {optimal_improvement}, {optimal_regression}, {optimal_same}")
    print(f"BAO stats: {bao_improvement}, {bao_regression}, {bao_same}")


def analyze_per_query_result():
    data_df = pd.read_csv("results/im_analysis.csv")
    queries = sorted(list(set(data_df["Query"])))
    pm_df = data_df[data_df["Sys"].str.contains("ParaRQO")]
    oim_df = data_df[data_df["Sys"].str.contains("OIM")]
    default_df = data_df[data_df["Sys"].str.contains("Postgres")]
    pm_performance = pm_df["TC"].values
    oim_performance = oim_df["TC"].values
    default_performance = default_df["TC"].values

    oim_ratio = pm_performance / oim_performance
    default_ratio = pm_performance / default_performance
    oim_regression = 0
    oim_improvement = 0
    oim_same = 0
    default_regression = 0
    default_improvement = 0
    default_same = 0
    bound = 0.2
    for idx, query in enumerate(queries):
        if oim_ratio[idx] > 1 + bound:
            oim_regression += 1
        elif oim_ratio[idx] < 1 - bound:
            oim_improvement += 1
        else:
            oim_same += 1
    for idx, query in enumerate(queries):
        if default_ratio[idx] > 1 + bound:
            default_regression += 1
        elif default_ratio[idx] < 1 - bound:
            default_improvement += 1
        else:
            default_same += 1

    bin_edges = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6])
    hist, bin_edges = np.histogram(oim_ratio, bins=bin_edges)
    hist = [h / sum(hist) for h in hist]

    print(zip(hist, bin_edges))
    bin_edges = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5])
    hist, bin_edges = np.histogram(default_ratio, bins=bin_edges)
    hist = [h / sum(hist) for h in hist]
    hist.append(0)
    print(zip(hist, bin_edges))

    print(f"OIM stats: {oim_improvement}, {oim_regression}, {oim_same}")
    print(f"Default stats: {default_improvement}, {default_regression}, {default_same}")


def evaluate_so_performance():
    nr_runs = 3
    methods = [
        "so_cardmodel_k5b10",
        "so_max_results",
        "so_postgres",
        "greedy_stack_poirange",
        "greedy_stack_poirange_mpd",
        "so_optimal",
        "greedy_stack_top3_least",
        "stack_bouquet",
        "stack_bouquet_topk",
    ]
    data = {}
    query_list = ["q1-", "q2-", "q3-", "q4-", "q5-", "q6-", "q7-", "q8-", "q9-", "q11-", "q12-"]

    def check_valid(s):
        for q in query_list:
            if s.startswith(q):
                return True
        return False
    for method in methods:
        data[method] = [pd.read_csv("/Users/tracy/Documents/Research/bak_ParaRQO/results/performance/"
                                    + method + "_" + str(run) + ".csv")
                        for run in range(nr_runs)]
        for idx, df in enumerate(data[method]):
            df['Valid'] = list(
                map(lambda x: check_valid(x), df['Query']))
            df = df[df['Valid']]
            data[method][idx] = df

    print(f"Method\t\t\t\tPerformance")
    for method in data:
        performance = np.array([data[method][x]["Time"] for x in range(nr_runs)])
        avg_performance = np.mean(performance, axis=0)
        avg_value = np.sum(avg_performance)
        sum_performance = np.sum(performance, axis=1)
        h = mean_confidence_interval(sum_performance)
        method_name = "_".join(method.split("_")[1:])
        nr_timeouts = np.mean([len([x for x in p if x >= 59]) for p in performance])
        print(f"{method_name},{avg_value},{h}, {nr_timeouts}")


def parallel_overhead(nr_runs=3):
    methods = ["fix_plan_0",
               "fix_plan_1",
               "fix_plan_2",
               "fix_plan_3",
               "fix_plan_4",
               "fix_plan_5",
               "greedy_combine_k3b10"
               ]
    data = {}
    for method in methods:
        data[method] = [pd.read_csv("/Users/tracy/Documents/Research/bak_ParaRQO/results/performance/"
                                    + method + "_" + str(run) + ".csv")
                        for run in range(nr_runs)]
    print(f"Method\t\t\t\tPerformance")
    pm_arms = list(data["greedy_combine_k3b10"][0]["Arms"])
    performance = np.array([data["greedy_combine_k3b10"][x]["Time"] for x in range(nr_runs)])
    avg_performance = np.mean(performance, axis=0)

    # Ideal time: iterate query
    ideal_time = np.zeros(len(avg_performance))
    sum1 = 0
    sum2 = 0
    sum3 = 0
    s = 0
    for idx, arm in enumerate(pm_arms):
        arm_idx_list = arm.split("-")
        min_time = sys.maxsize
        min1 = sys.maxsize
        min2 = sys.maxsize
        min3 = sys.maxsize
        for arm_idx in arm_idx_list:
            arm_time = np.mean([data["fix_plan_" + arm_idx][0]["Time"][idx],
                                data["fix_plan_" + arm_idx][1]["Time"][idx],
                                data["fix_plan_" + arm_idx][2]["Time"][idx]])
            arm1 = data["fix_plan_" + arm_idx][0]["Time"][idx]
            min1 = min(min1, arm1)
            arm2 = data["fix_plan_" + arm_idx][1]["Time"][idx]
            min2 = min(min2, arm2)
            arm3 = data["fix_plan_" + arm_idx][2]["Time"][idx]
            min3 = min(min3, arm3)
            if arm_time < min_time:
                min_time = arm_time
        ideal_time[idx] = min_time
        sum1 += min1
        sum2 += min2
        sum3 += min3
        s += np.mean([min1, min2, min3])

    ideal_sum = np.sum(ideal_time)
    h = mean_confidence_interval([sum1, sum2, sum3])
    avg_sum = np.sum(avg_performance)
    mean_speedup = avg_sum / ideal_sum
    print(f"{mean_speedup},{avg_sum},{ideal_sum},{s},{h}")

    # Find the optimal arms
    optimal_arms = []
    optimal_time = []
    has_optimal_speedup = []
    no_optimal_speedup = []
    default_time_list_optimal = []
    default_time_list_no_optimal = []
    for idx, arm in enumerate(pm_arms):
        best_arm = -1
        best_time = sys.maxsize
        default_time = 0
        for arm_idx in range(6):
            arm_time = np.mean([data["fix_plan_" + str(arm_idx)][0]["Time"][idx],
                                data["fix_plan_" + str(arm_idx)][1]["Time"][idx],
                                data["fix_plan_" + str(arm_idx)][2]["Time"][idx]])
            if arm_time < best_time:
                best_time = arm_time
                best_arm = arm_idx
            default_time = data["fix_plan_0"][0]["Time"][idx]
        optimal_time.append(best_time)
        optimal_arms.append(str(best_arm))
        if str(best_arm) in arm.split("-"):
            has_optimal_speedup.append(avg_performance[idx])
            default_time_list_optimal.append(default_time)
        else:
            no_optimal_speedup.append(avg_performance[idx])
            default_time_list_no_optimal.append(default_time)
    mean_optimal_speedup = np.mean(has_optimal_speedup)
    mean_bad_speedup = np.mean(no_optimal_speedup)
    print(np.sum(default_time_list_optimal) / np.sum(has_optimal_speedup),
          np.sum(default_time_list_no_optimal) / np.sum(no_optimal_speedup),
          len(has_optimal_speedup) / len(optimal_arms), len(no_optimal_speedup) / len(optimal_arms))

    # Test different selection methods
    # 1. Random
    random_list = []
    for idx, arm in enumerate(pm_arms):
        arm_idx_list = arm.split("-")
        random_arms = np.random.choice([0, 1, 2, 3, 4, 5], 3)
        random_arm = -1
        random_time = sys.maxsize
        for arm_idx in random_arms:
            arm_time = np.mean([data["fix_plan_" + str(arm_idx)][0]["Time"][idx],
                                data["fix_plan_" + str(arm_idx)][1]["Time"][idx],
                                data["fix_plan_" + str(arm_idx)][2]["Time"][idx]])
            if arm_time < random_time:
                random_time = arm_time
                random_arm = arm_idx
        random_list.append(random_time)
    print(sum(random_list))


def so_parallel_overhead(nr_runs=3):
    methods = ["so_fix_plan_0",
               "so_fix_plan_1",
               "so_fix_plan_2",
               "so_fix_plan_3",
               "so_fix_plan_4",
               "so_fix_plan_5",
               "greedy_stack_poirange"
               ]
    data = {}
    for method in methods:
        data[method] = [pd.read_csv("/Users/tracy/Documents/Research/bak_ParaRQO/results/performance/"
                                    + method + "_" + str(run) + ".csv")
                        for run in range(3)]
        for idx, df in enumerate(data[method]):
            df['Valid'] = list(
                map(lambda x: check_valid(x), df['Query']))
            df = df[df['Valid']]
            data[method][idx] = df
    print(f"Method\t\t\t\tPerformance")
    pm_arms = list(data["greedy_stack_poirange"][0]["Arms"])
    performance = np.array([data["greedy_stack_poirange"][x]["Time"] for x in range(nr_runs)])
    avg_performance = np.mean(performance, axis=0)

    # Ideal time: iterate query
    ideal_time = np.zeros(len(avg_performance))
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for idx, arm in enumerate(pm_arms):
        arm_idx_list = arm.split("-")
        min_time = sys.maxsize
        min1 = sys.maxsize
        min2 = sys.maxsize
        min3 = sys.maxsize
        for arm_idx in arm_idx_list:
            arm_time = np.mean([data["so_fix_plan_" + arm_idx][0]["Time"][idx],
                                data["so_fix_plan_" + arm_idx][1]["Time"][idx],
                                data["so_fix_plan_" + arm_idx][2]["Time"][idx]])
            arm1 = data["so_fix_plan_" + arm_idx][0]["Time"][idx]
            min1 = min(min1, arm1)
            arm2 = data["so_fix_plan_" + arm_idx][1]["Time"][idx]
            min2 = min(min2, arm2)
            arm3 = data["so_fix_plan_" + arm_idx][2]["Time"][idx]
            min3 = min(min3, arm3)
            if arm_time < min_time:
                min_time = arm_time
        ideal_time[idx] = min_time
        sum1 += min1
        sum2 += min2
        sum3 += min3
    h = mean_confidence_interval([sum1, sum2, sum3])
    ideal_sum = np.sum(ideal_time)
    avg_sum = np.sum(avg_performance)
    mean_speedup = avg_sum / ideal_sum
    print(f"{mean_speedup},{avg_sum},{ideal_sum},{h}")



def so_breakdown():
    methods = ["so_optimal",
               "greedy_stack_poirange",
               "so_postgres"
               ]
    data = {}

    for method in methods:
        data[method] = [pd.read_csv("/Users/tracy/Documents/Research/bak_ParaRQO/results/performance/"
                                    + method + "_" + str(run) + ".csv")
                        for run in range(3)]
        for idx, df in enumerate(data[method]):
            df['Valid'] = list(
                map(lambda x: check_valid(x), df['Query']))
            df = df[df['Valid']]
            data[method][idx] = df

    optimal_arms = list(data["so_optimal"][0]["Arm"])
    pm_arms = list(data["greedy_stack_poirange"][0]["Arms"])
    performance = np.array([data["greedy_stack_poirange"][x]["Time"] for x in range(3)])
    avg_performance = np.mean(performance, axis=0)
    default_performance = np.array([data["so_postgres"][x]["Time"] for x in range(3)])

    avg_default_performance = np.mean(default_performance, axis=0)
    has_optimal_speedup = []
    default_time_list_optimal = []
    no_optimal_speedup = []
    default_time_list_no_optimal = []
    for idx, arm in enumerate(pm_arms):
        if str(optimal_arms[idx]) in arm.split("-"):
            has_optimal_speedup.append(avg_performance[idx])
            default_time_list_optimal.append(avg_default_performance[idx])
        else:
            no_optimal_speedup.append(avg_performance[idx])
            default_time_list_no_optimal.append(avg_default_performance[idx])
    print(np.sum(default_time_list_optimal) / np.sum(has_optimal_speedup),
            np.sum(default_time_list_no_optimal) / np.sum(no_optimal_speedup),
            len(has_optimal_speedup) / len(pm_arms), len(no_optimal_speedup) / len(pm_arms))

def vary_size(nr_runs=3):
    methods = [
        "greedy_combine_k3b10",
        "fix_plan_0",
        "greedy_imdb2_poirange",
        "imdb2_fix_plan_0",
        "greedy_imdb4_poirange",
        "imdb4_fix_plan_0",
        "greedy_imdb6_poirange",
        "imdb6_fix_plan_0",
        "greedy_imdb10_poirange",
        "imdb10_fix_plan_0"
    ]
    data = {}
    for method in methods:
        data[method] = [pd.read_csv("/Users/tracy/Documents/Research/bak_ParaRQO/results/performance/"
                                    + method + "_" + str(run) + ".csv")
                        for run in range(nr_runs)]
    print(f"Method\t\t\t\tPerformance")
    for method in data:
        performance = np.array([data[method][x]["Time"] for x in range(nr_runs)])
        nr_timeouts = np.mean([len([x for x in p if x >= 60]) for p in performance])
        avg_performance = np.mean(performance, axis=0)
        avg_value = np.sum(avg_performance)
        sum_performance = np.sum(performance, axis=1)
        h = mean_confidence_interval(sum_performance)
        method_name = method.replace("greedy_", "")
        print(f"{method_name},{avg_value},{h}, {nr_timeouts}")


def breakdown_query():
    data_df = pd.read_csv("/Users/tracy/Documents/Research/bak_ParaRQO/results/query_performance.csv")
    queries = sorted(list(set(data_df["Query"])))
    pm_df = data_df[data_df["Method"].str.contains("combine_k3b10")]
    optimal_df = data_df[data_df["Method"].str.contains("optimal")]
    bao_df = data_df[data_df["Method"].str.contains("BAO1")]
    pm_performance = pm_df["Performance"].values
    optimal_performance = optimal_df["Performance"].values
    bao_performance = bao_df["Performance"].values

    like_queries = []
    not_like_queries = []
    like_speedups = []
    not_like_speedups = []

    for idx, query in enumerate(queries):
        sql = open(f"/Users/tracy/Documents/Research/ParaRQO/queries/imdb/{query}.sql").read()
        if " like " in sql.lower():
            like_queries.append(query)
            like_speedups.append(bao_performance[idx] / pm_performance[idx])
        else:
            not_like_queries.append(query)
            not_like_speedups.append(bao_performance[idx] / pm_performance[idx])

    num_dict = {}
    unary_predicates_dict = {}
    with open("/Users/tracy/Documents/Research/bak_ParaRQO/results/ep_speedup.csv", "w") as f:
        f.write("NrPredicates,Speedup\n")
        for idx, query in enumerate(queries):
            sql = open(f"/Users/tracy/Documents/Research/ParaRQO/queries/imdb/{query}.sql").read()
            sql = sql.replace(" where ", " WHERE ")
            sql = sql.replace(" and ", " AND ")
            sql = sql.replace(" between ", " BETWEEN ")
            sql = sql.replace(";", "")
            count = 0
            if " like " in sql.lower():
                count += 1
            if " > " in sql:
                count += 1
            if " < " in sql:
                count += 1
            if " >= " in sql:
                count += 1
            if " <= " in sql:
                count += 1
            if " BETWEEN " in sql.upper():
                count += 1
            if count not in num_dict:
                num_dict[count] = []
            sp = bao_performance[idx] / pm_performance[idx]
            num_dict[count].append(sp)
            predicates = sql.split("WHERE")[-1].strip().split(" AND ")
            join_predicates = get_join_predicates(sql)
            potential_unary_predicates = [p.strip() for p in predicates if p.strip() not in join_predicates]
            unary_predicates = []

            for nidx, u in enumerate(potential_unary_predicates):
                if len(unary_predicates) > 0 and " BETWEEN " in unary_predicates[-1]:
                    unary_predicates[-1] = unary_predicates[-1] + " AND " + u
                else:
                    unary_predicates.append(u)
            nr_unary_predicates = len(unary_predicates)

            if nr_unary_predicates not in unary_predicates_dict:
                unary_predicates_dict[nr_unary_predicates] = []
            unary_predicates_dict[nr_unary_predicates].append(sp)

            f.write(f"{count},{sp}\n")

    # for num in sorted(num_dict.keys()):
    #     if len(num_dict[num]) == 1:
    #         h = 0
    #     else:
    #         h = mean_confidence_interval(num_dict[num])
    #     print(f"({num}, {np.mean(num_dict[num])}) +- ({h}, {h})")
    #
    # for num in sorted(unary_predicates_dict.keys()):
    #     if len(unary_predicates_dict[num]) == 1:
    #         h = 0
    #     else:
    #         h = mean_confidence_interval(unary_predicates_dict[num])
    #     print(f"({num}, {np.mean(unary_predicates_dict[num])}) +- ({h}, {h})")


def calculate_variance():
    methods = ["fix_plan_0",
               "fix_plan_1",]
    data = {}
    for method in methods:
        data[method] = pd.read_csv("/Users/tracy/Documents/Research/bak_ParaRQO/results/performance/"
                                   + method + "_0.csv")
    for method in data:
        performance = np.array(data[method]["Time"])
        method_name = "_".join(method.split("_")[1:])
        avg_performance = np.mean(performance)
        var_performance = np.var(performance) ** 0.5
        print(f"{method_name},{avg_performance},{var_performance}")


if __name__ == '__main__':
    performance_evaluation()
    evaluate_so_performance()
    # time_vs_quality()
    # im_analysis_per_query()
    # analyze_per_query_performance()
    # analyze_per_query_result()
    # parallel_overhead()
    # so_breakdown()
    # so_parallel_overhead()
    # vary_size()
    # breakdown_query()
    # calculate_variance()
