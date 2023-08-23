import numpy as np
import pandas as pd
import scipy.stats as st


def mean_confidence_interval(data, confidence=0.95):
    h = st.t.interval(confidence, len(data)-1, loc=np.mean(data), scale=st.sem(data))
    return h[1] - np.mean(data)


def performance_evaluation():
    nr_runs = 3
    methods = ["greedy_intermediate_results",
               "greedy_intermediate_results_para3",
               "greedy_intermediate_results_2",
               "fix_plan_0",
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
               "greedy_cardmodel_k1b10p0.5",
               "greedy_cardmodel_k3b10p0.5",
               "greedy_cardmodel_k5b10p0.5",
               "greedy_cardmodel_k7b10p0.5",
               "greedy_cardmodel_k3b5p0.5",
               "greedy_cardmodel_k3b50p0.5",
               "greedy_cardmodel_k3b100p0.5",

               "greedy_cardsq_k5b10e0.5",
               "greedy_cardsq_k5b10e1",
               "greedy_cardsq_k5b10e10",

               "greedy_imresults",
               "greedy_imresults_ilp",
               "greedy_imresults_ilp_para3",
               "greedy_imresults_ilp_para8"
               ]
    data = {}
    for method in methods:
        data[method] = [pd.read_csv("results/performance/" + method + "_" + str(run) + ".csv")
                        for run in range(nr_runs)]
    optimal = [{"Time": np.min(np.array([data["fix_plan_" + str(0)][run]["Time"],
                                         data["fix_plan_" + str(1)][run]["Time"],
                                         data["fix_plan_" + str(2)][run]["Time"],
                                         data["fix_plan_" + str(3)][run]["Time"],
                                         data["fix_plan_" + str(4)][run]["Time"],
                                         data["fix_plan_" + str(5)][run]["Time"]]),
                               axis=0)} for run in range(nr_runs)]
    data["imdb_optimal"] = optimal
    print(f"Method\t\t\t\tPerformance")
    for method in data:
        performance = np.array([data[method][x]["Time"] for x in range(nr_runs)])
        avg_performance = np.mean(performance, axis=0)
        avg_value = np.sum(avg_performance)
        sum_performance = np.sum(performance, axis=1)
        h = mean_confidence_interval(sum_performance)
        method_name = "_".join(method.split("_")[1:])
        print(f"{method_name}\t{avg_value}\t{h}")

    # with open("results/query_performance.csv", "w") as f:
    #     f.write("Query,Method,Performance,Interval\n")
    #     for method in data:
    #         if method in ["fix_plan_0", "fix_plan_1", "fix_plan_2", "fix_plan_3", "fix_plan_4",
    #                       "fix_plan_5", "imdb_optimal", "greedy_cardmodel_k7b10p0.5"]:
    #             performance = np.array([data[method][x]["Time"] for x in range(nr_runs)])
    #             interval = np.array([mean_confidence_interval(y) for y in performance.T])
    #             queries = list(data["fix_plan_0"][0]["Query"])
    #             avg_performance = np.mean(performance, axis=0)
    #
    #             method_name = "_".join(method.split("_")[1:])
    #             for idx, p in enumerate(avg_performance):
    #                 query_name = queries[idx].split(".")[0]
    #                 f.write(f"{query_name},{method_name},{p},{interval[idx]}\n")
    #
    #             query_names = ",".join([q.split(".")[0] for m, q in enumerate(queries) if m % 10 == 0])
    #
    #             print(query_names)


def time_vs_quality():
    # methods = ["greedy_imresults", "greedy_imresults_ilp_para3", "greedy_imresults_ilp_para8"]
    methods = ["greedy_full_ilppara3", "greedy_full_ilppara8",
               "greedy_full_imresultspara3", "greedy_full_imresultspara8",
               "greedy_psql_ilppara2", "greedy_psql_ilppara4", "greedy_psql_ilppara6", "greedy_psql_ilppara8",
               "greedy_psql_imresultspara2", "greedy_psql_imresultspara4", "greedy_psql_imresultspara6", "greedy_psql_imresultspara8",]
    data = {}
    nr_runs = 3
    for method in methods:
        data[method] = [pd.read_csv("results/performance/" + method + "_" + str(run) + ".csv")
                        for run in range(nr_runs)]
    print(f"Method\tPerformance\tInterval\tIMs Quality\tPlan Time\tNr Plans")
    for method in data:
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
        print(f"{method_name}\t{avg_value}\t{h}\t{avg_quality}\t{all_plan_time}\t{avg_nr_plans}")


if __name__ == '__main__':
    # performance_evaluation()
    time_vs_quality()
