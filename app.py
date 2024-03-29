import asyncio
import os
import subprocess
import sys
import re
import aiopg
import pandas as pd
import psycopg2
import streamlit as st
from time import time
from psycopg2.extras import NamedTupleCursor
from streamlit_echarts import st_echarts

from db.postgres import Postgres

# Use the non-interactive Agg backend, which is recommended as a
# thread-safe backend.
# See https://matplotlib.org/3.3.2/faq/howto_faq.html#working-with-threads.
import matplotlib as mpl

from plan.plan_tree import PlanTree
from plan.planner import Planner
_ALL_OPTIONS = [
        "enable_nestloop", "enable_hashjoin", "enable_mergejoin",
        "enable_seqscan", "enable_indexscan", "enable_indexonlyscan"
    ]
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

sys.path.append('/')
mpl.use("agg")

##############################################################################
# Workaround for the limited multi-threading support in matplotlib.
# Per the docs, we will avoid using `matplotlib.pyplot` for figures:
# https://matplotlib.org/3.3.2/faq/howto_faq.html#how-to-use-matplotlib-in-a-web-application-server.
# Moreover, we will guard all operations on the figure instances by the
# class-level lock in the Agg backend.
##############################################################################
from matplotlib.backends.backend_agg import RendererAgg

_lock = RendererAgg.lock

# -- Set page config
apptitle = 'ROME Demo'

st.set_page_config(page_title=apptitle,  layout="wide")

# Title the app
st.title('ROME: Robust Optimization via Multi-Plan Execution')
st.markdown(
    """
    <style>
    p {
        font-size: large !important;
    }
    
    td {
        font-size: large !important;
    }
    
    .st-bl {
        font-size: Large !important;
    }   

    </style>
    """,
    unsafe_allow_html=True,
)
# st.markdown("""
#  * Use the menu at left to select dataset and choose system parameters
#  * Write down the query in the text box
#  * The system will return the query result using alternative plans
# """)


st.sidebar.markdown("## Select Dataset and Parameters")


# -- Set time by GPS or event
select_dataset = st.sidebar.selectbox('Dataset', ['Join Order Benchmark', 'Stackoverflow', 'Upload your own dataset'])
database = "imdb" if select_dataset == 'Join Order Benchmark' else "so"
query_dict = None
dataset_queries = None
if select_dataset == 'Join Order Benchmark' or select_dataset == 'Stackoverflow':
    query_dir = f"./queries/{database}"
    query_names = []
    for fp in os.listdir(query_dir):
        if fp.endswith(".sql"):
            with open(os.path.join(query_dir, fp)) as f:
                query = f.read()
                query_names.append((fp, query))
    if database == "so":
        unsupported_queries = ["q10-", "q13-", "q14-", "q15-", "q16-"]
        query_names = [q for q in query_names if not any([q[0].startswith(u) for u in unsupported_queries])]
    query_names = sorted(query_names, key=lambda x: x[0])
    query_names = sorted(query_names, key=lambda x: int(x[0].split(".")[0].split("-")[0].replace("q", "")[0:2]))
    query_dict = {q[0].split(".")[0]: q[1] for q in query_names}
    dataset_queries = st.sidebar.selectbox('Queries', [q[0].split(".")[0] for q in query_names])

else:
    fileUploadLabel = "Upload your dataset in csv"
    uploadedFile = st.sidebar.file_uploader(fileUploadLabel, type=['csv'],
                                            accept_multiple_files=False, key="fileUploader")
    if uploadedFile is not None:
        uploaded_df = pd.read_csv(uploadedFile)
        table_name = "test"
        print(uploaded_df)
    # st.stop()

select_solver = st.sidebar.selectbox('Plan Selector', ['Default', 'MPD', 'PM', 'Manual'])
if query_dict is not None and dataset_queries is not None:
    query = st.text_area('Query', query_dict[dataset_queries].strip(), height=200)
else:
    query = st.text_area('Query', "SELECT * FROM test", height=200)

rerun = False
def click_button():
    rerun = True
    for key in st.session_state.keys():
        del st.session_state[key]


# st.sidebar.button('Rerun', on_click=click_button)

# -- Create sidebar for plot controls
st.sidebar.markdown('## Set System Parameters')
if select_solver == "Default":
    nr_plans = 1
elif select_solver == "MPD" or select_solver == "PM":
    # Nr. alternative plans
    nr_plans = st.sidebar.slider('Nr. parallel plans', 1, 10, 3)
check_boxes = []
if select_solver == "PM":
    # Nr. intermediate results
    k = st.sidebar.slider('Nr. intermediate results', 1, 10, 3)
    # Nr. blocks
    b = st.sidebar.slider('Nr. blocks', 10, 50, 10, step=10)
elif select_solver == "Manual":
    k = 3
    b = 10
    check_boxes = st.sidebar.multiselect(
        'Select the plans to execute',
        ['Plan 0', 'Plan 1', 'Plan 2', 'Plan 3', 'Plan 4', 'Plan 5'], ['Plan 1'])
    nr_plans = len(check_boxes)
else:
    k = 3
    b = 10
# Nr. blocks
timeout = st.sidebar.slider('Timeout', 10, 120, 30, step=10)

PG_CONNECTION_STR = f"dbname={database} user=postgres password=postgres host=localhost port=5432"
# -- Create a text element and let the reader know the data is loading.
# strain_load_state = st.text('Connecting to the database...')
try:
    postgres = Postgres("localhost", 5432, "postgres", "postgres", database)
    postgres.close()
except:
    st.warning('Failed to connect to the database. Please check the connection parameters and try again.')
    st.stop()

# strain_load_state.text(f'Connected to {database}...done!')


async def run_query(idx, arm_idx, connections, running_flags, sql, nr_threads, explain=False):
    async with aiopg.connect(PG_CONNECTION_STR) as con:
        connections[idx] = await con.get_backend_pid()
        async with con.cursor(cursor_factory=NamedTupleCursor) as cursor:
            hints = _arm_idx_to_hints(arm_idx)
            execute_sql = f"SET max_parallel_workers = {nr_threads};\n" \
                          f"SET max_parallel_workers_per_gather = {nr_threads};" \
                          f"\nSET statement_timeout = {timeout * 1000};\n"
            execute_sql += ("\n".join(hints) + "\n")
            if explain:
                execute_sql += ("EXPLAIN (ANALYZE, COSTS, VERBOSE, FORMAT JSON) " + sql)
            else:
                execute_sql += sql
            start = time()
            try:
                running_flags[idx] = True
                await cursor.execute(execute_sql)
                results = await cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                end = time()
                running_flags[idx] = None
                for other_id, connection_pid in enumerate(connections):
                    if connection_pid != connections[idx] and running_flags[other_id] is not None:
                        print(f"Terminating {connection_pid}", flush=True)
                        await cursor.execute(f'SELECT pg_terminate_backend({connection_pid});')
                return results, columns, end - start
            except (asyncio.exceptions.CancelledError,
                    psycopg2.errors.QueryCanceled,
                    Exception) as e:
                end = time()
                if end - start > timeout // 1000:
                    print("Terminate due to timeout", e, flush=True)
                    return None, None, end - start
                else:
                    print("Terminate due to internal error", e, flush=True)
                    return None, None, end - start


def find_arms(sql, postgres, method_name="max_results",
              query_name="test", conn=None):
    plan_trees = {}
    nr_arms = 6
    solver = "greedy" if "ILP" not in method_name else "ilp"
    method = method_name.split("-")[0]
    if method == "MPD" and solver == "ilp":
        method = "max_im_ilp_parallel"
    elif method == "PM":
        method = "probability_model"
    elif method == "MPD" and solver == "greedy":
        method = "max_results"
    elif method == "Default":
        method = "Default"
    elif method == "Manual":
        method = "Manual"
    else:
        method = "probability_model"
    for x in range(nr_arms):
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
    if method == "Default":
        return [0], 0, 0, plan_trees
    if method == "Manual":
        selected_arms = sorted([int(c.split(" ")[1]) for c in check_boxes])
        return selected_arms, 0, 0, plan_trees
    planner = Planner(plan_trees, postgres, 24,
                      k, b, method, nr_plans,
                      solver=solver, penalty=0.01, conn=conn)
    if method == "probability_model":
        st.session_state['planner'] = planner
    plan_start = time()
    optimal_arms = planner.optimal_arms(plan_trees)
    plan_end = time()
    return optimal_arms, plan_end - plan_start, planner.nr_selected_ims, plan_trees


async def rome_query_execution(q, nr_process, selected_arms, explain=False):
    connections = [None for _ in range(len(selected_arms))]
    running_flags = [None for _ in range(len(selected_arms))]
    nr_threads = 24 // nr_process
    executor_list = [run_query(idx, selected_arms[idx], connections, running_flags, q, nr_threads, explain=explain)
                     for idx in range(len(selected_arms))]
    try:
        results = await asyncio.gather(*executor_list)
    except Exception as e:
        return [(None, None, 0) for _ in range(len(selected_arms))]
    return results

query_name = "test"
postgres.set_sql_query(query, query_name)
if ('plan_trees' in st.session_state and 'query_name' in st.session_state
        and st.session_state['query_name'] == dataset_queries
        and 'method' in st.session_state and st.session_state['method'] == select_solver
        and (select_solver != "Manual" or 'check_box' not in st.session_state
             or st.session_state['check_box'] != check_boxes) and not rerun):
    plan_trees, plan_time, nr_ims, arms = st.session_state['plan_trees'], st.session_state['plan_time'], \
                                          st.session_state['nr_ims'], st.session_state['arms']
    rerun = False
else:
    arms, plan_time, nr_ims, plan_trees = find_arms(query, postgres, method_name=select_solver)
    st.session_state['plan_trees'] = plan_trees
    st.session_state['plan_time'] = plan_time
    st.session_state['nr_ims'] = nr_ims
    st.session_state['arms'] = arms
    st.session_state['query_name'] = dataset_queries
    st.session_state['method'] = select_solver
    st.session_state['check_box'] = check_boxes
    if "actual_results" in st.session_state:
        del st.session_state['actual_results']
    rerun = True
arms = sorted(arms)
col1, col2 = st.columns([0.6, 0.4])
font_size = 20
label_size = 20
height = "450px"
with col1:
    st.subheader("Plan Selection")

    # tab1, tab2, tab3 = st.tabs(['Estimated Nr. Rows', 'Actual Nr. Rows', 'Nr. Intermediate Results'])
    # tabs = st.selectbox('Plan Selection', ['Estimated Nr. Rows', 'Actual Nr. Rows', 'Nr. Intermediate Results'])

    tabs = st.radio(
        "",
        ["Estimated Nr. Rows", "Nr. Plans Per Results", "Std of Cardinality Estimation"], horizontal=True)
    if tabs == "Estimated Nr. Rows":
        plan_data = {arm: plan_trees[arm].to_dict(plan_trees[arm].root) for arm in arms}
        min_val = sys.maxsize
        max_val = 0
        for arm in arms:
            f_nodes = plan_trees[arm].root.f_nodes
            for f_node in f_nodes:
                nr_rows = f_node.plan_rows
                if nr_rows < min_val:
                    min_val = nr_rows
                if nr_rows > max_val:
                    max_val = nr_rows
        legend_data = []
        for arm in arms:
            if arm == 0:
                legend_data.append({
                        "name": f"Plan {arm}",
                        "icon": 'rectangle'
                        # "icon": 'image://https://t3.ftcdn.net/jpg/04/89/08/82/240_F_489088251_p6pGD1QqJVD3g6mcSGpn9s3fyhLjKSEg.jpg',
                        # "icon": 'circle'
                    })
            else:
                legend_data.append({
                    "name": f"Plan {arm}",
                    "icon": 'rectangle'
                })
        opts_1 = {
            "tooltip": {"trigger": "item", "triggerOn": "mousemove", "formatter": "{b}: {c}"},
            "legend": {
                "top": '1%',
                "left": '2%',
                "orient": 'horizontal',
                "data": legend_data,
                # "width": "50",
                # "height": "50",
            },
            "visualMap": {
                "left": 'right',
                "top": 'top',
                "min": min_val,
                "max": max_val,
                "inRange": {
                    "color": [
                        '#313695',
                        '#4575b4',
                        '#74add1',
                        '#abd9e9',
                        '#e0f3f8',
                        '#ffffbf',
                        '#fee090',
                        '#fdae61',
                        '#f46d43',
                        '#d73027',
                        '#a50026'
                    ]
                },
                "text": ['High', 'Low'],
                "calculable": True
            },
            "series": [
                {
                    "type": "tree",
                    "name": f"Plan {arm}",
                    "data": [plan_data[arm]],
                    "top": f"12%",
                    "left": f"{2 + 100 // len(arms) * arm_idx}%",
                    "bottom": f"10%",
                    "right": f"{100 - (100 // len(arms) * (arm_idx + 1) - 2)}%",
                    "symbol": "circle",
                    # "roam": True,
                    "itemStyle": {"borderWidth": 0},
                    "symbolSize": label_size,
                    "orient": 'vertical',
                    "label": {
                        "position": "top",
                        # "rotate": -90,
                        "verticalAlign": "middle",
                        "align": "middle",
                        "fontSize": font_size,
                        "overflow": "break",
                        "formatter": "{c}"
                    },
                    "leaves": {
                        "label": {
                            "position": "bottom",
                            # "rotate": -90,
                            "verticalAlign": "middle",
                            "align": "middle",
                            "formatter": "{b}"
                        }
                    },
                    "expandAndCollapse": False,
                    "animationDuration": 550,
                    "animationDurationUpdate": 750,
                } for arm_idx, arm in enumerate(arms)],
        }
        st_echarts(opts_1, height=height, width="90%")

    elif tabs == "Actual Nr. Rows":
        if "actual_results" in st.session_state:
            actual_plan_data = st.session_state['actual_results']
        else:
            actual_plan_data = asyncio.run(rome_query_execution(query, nr_plans, arms, explain=True))
            st.session_state['actual_results'] = actual_plan_data
        plan_trees = {}
        for thread_ctr, r in enumerate(actual_plan_data):
            if r[0] is not None:
                query_plan = list(r[0][0][0])
                plan_trees[arms[thread_ctr]] = PlanTree(query, r[0], postgres, visualization=False,
                                                        pid=thread_ctr, query_name=query_name, query_plan=query_plan)
            else:
                plan_trees[arms[thread_ctr]] = None
        actual_data = {}
        for arm in arms:
            if plan_trees[arm] is not None:
                actual_data[arm] = plan_trees[arm].to_dict(plan_trees[arm].root, key="actual_rows")
            else:
                actual_data[arm] = None
        min_val = sys.maxsize
        max_val = 0
        for arm in arms:
            f_nodes = plan_trees[arm].root.f_nodes
            for f_node in f_nodes:
                nr_rows = f_node.plan_rows
                if nr_rows < min_val:
                    min_val = nr_rows
                if nr_rows > max_val:
                    max_val = nr_rows
        opts_2 = {
            "tooltip": {"trigger": "item", "triggerOn": "mousemove", "formatter": "{b}: {c}"},
            "legend": {
                "top": '1%',
                "left": '2%',
                "orient": 'horizontal',
                "data": [
                    {
                        "name": f"Plan {arm}",
                        "icon": 'rectangle'
                    }
                    for arm in arms
                ],
            },
            "visualMap": {
                "left": 'right',
                "top": 'top',
                "min": min_val,
                "max": max_val,
                "inRange": {
                    "color": [
                        '#313695',
                        '#4575b4',
                        '#74add1',
                        '#abd9e9',
                        '#e0f3f8',
                        '#ffffbf',
                        '#fee090',
                        '#fdae61',
                        '#f46d43',
                        '#d73027',
                        '#a50026'
                    ]
                },
                "text": ['High', 'Low'],
                "calculable": True
            },
            "series": [
                {
                    "type": "tree",
                    "name": f"Plan {arm}",
                    "data": [actual_data[arm]],
                    "top": f"12%",
                    "left": f"{2 + 100 // len(arms) * arm_idx}%",
                    "bottom": f"10%",
                    "right": f"{100 - (100 // len(arms) * (arm_idx + 1) - 2)}%",
                    "symbol": "circle",
                    # "roam": True,
                    "itemStyle": {"borderWidth": 0},
                    "symbolSize": label_size,
                    "orient": 'vertical',
                    "label": {
                        "position": "top",
                        # "rotate": -90,
                        "verticalAlign": "middle",
                        "align": "middle",
                        "fontSize": font_size,
                        "overflow": "break",
                        "formatter": "{c}"
                    },
                    "leaves": {
                        "label": {
                            "position": "bottom",
                            # "rotate": -90,
                            "verticalAlign": "middle",
                            "align": "middle",
                            "formatter": "{b}"
                        }
                    },
                    "expandAndCollapse": False,
                    "animationDuration": 550,
                    "animationDurationUpdate": 750,
                } for arm_idx, arm in enumerate(arms)],
        }
        st_echarts(opts_2, height=height, width="90%")

    elif tabs == "Nr. Plans Per Results":
        f_nodes_count = {}
        min_val = sys.maxsize
        max_val = 0
        for arm in arms:
            for f_node in plan_trees[arm].root.f_nodes:
                f_key = f_node.f_key
                if f_key not in f_nodes_count:
                    f_nodes_count[f_key] = 0
                f_nodes_count[f_key] += 1
        im_plan_data = {arm: plan_trees[arm].to_dict(plan_trees[arm].root) for arm in arms}
        for arm in arms:
            arm_data = im_plan_data[arm]
            stack = [arm_data]
            while len(stack) > 0:
                node = stack.pop()
                if node["name"] in f_nodes_count:
                    node["value"] = f_nodes_count[node["name"]]
                else:
                    node["value"] = len(arms)
                if node["value"] < min_val:
                    min_val = node["value"]
                if node["value"] > max_val:
                    max_val = node["value"]
                for child in node["children"]:
                    stack.append(child)
        opts_3 = {
            "tooltip": {"trigger": "item", "triggerOn": "mousemove", "formatter": "{b}: {c}"},
            "legend": {
                "top": '1%',
                "left": '2%',
                "orient": 'horizontal',
                "data": [
                    {
                        "name": f"Plan {arm}",
                        "icon": 'rectangle'
                    }
                    for arm in arms
                ],
            },
            "visualMap": {
                "left": 'right',
                "top": 'top',
                "min": min_val,
                "max": max_val,
                "inRange": {
                    "color": [
                        '#313695',
                        '#4575b4',
                        '#74add1',
                        '#abd9e9',
                        '#e0f3f8',
                        '#ffffbf',
                        '#fee090',
                        '#fdae61',
                        '#f46d43',
                        '#d73027',
                        '#a50026'
                    ]
                },
                "text": ['High', 'Low'],
                "calculable": True
            },
            "series": [
                {
                    "type": "tree",
                    "name": f"Plan {arm}",
                    "data": [im_plan_data[arm]],
                    "top": f"12%",
                    "left": f"{2 + 100 // len(arms) * arm_idx}%",
                    "bottom": f"10%",
                    "right": f"{100 - (100 // len(arms) * (arm_idx + 1) - 2)}%",
                    "symbol": "circle",
                    # "roam": True,
                    "itemStyle": {"borderWidth": 0},
                    "symbolSize": label_size,
                    "orient": 'vertical',
                    "label": {
                        "position": "top",
                        # "rotate": -90,
                        "verticalAlign": "middle",
                        "align": "middle",
                        "fontSize": font_size,
                        "overflow": "break",
                        "formatter": "{c}"
                    },
                    "leaves": {
                        "label": {
                            "position": "bottom",
                            # "rotate": -90,
                            "verticalAlign": "middle",
                            "align": "middle",
                            "formatter": "{b}"
                        }
                    },
                    "expandAndCollapse": False,
                    "animationDuration": 550,
                    "animationDurationUpdate": 750,
                } for arm_idx, arm in enumerate(arms)],
        }
        st_echarts(opts_3, height=height, width="90%", key="im_plan")

    elif tabs == "Std of Cardinality Estimation":
        variance_dict = {}
        min_val = sys.maxsize
        max_val = 0
        if select_solver == "PM":
            planner = st.session_state['planner']
            print(planner.f_keys_to_ranges, flush=True)
            for f_key in planner.f_keys_to_ranges:
                for val, prob, k in planner.f_keys_to_ranges[f_key]:
                    variance_dict[f_key] = variance_dict.get(f_key, 0) + prob * val
                variance_dict[f_key] = variance_dict[f_key] ** 0.5

        im_plan_data = {arm: plan_trees[arm].to_dict(plan_trees[arm].root) for arm in arms}
        for arm in arms:
            arm_data = im_plan_data[arm]
            stack = [arm_data]
            while len(stack) > 0:
                node = stack.pop()
                node_key = frozenset(node["name"].replace(":", "-").split("-"))
                if node_key in variance_dict:
                    node["value"] = int(variance_dict[node_key])
                else:
                    node["value"] = 0
                if node["value"] < min_val:
                    min_val = node["value"]
                if node["value"] > max_val:
                    max_val = node["value"]
                for child in node["children"]:
                    stack.append(child)
        opts_4 = {
            "tooltip": {"trigger": "item", "triggerOn": "mousemove", "formatter": "{b}: {c}"},
            "legend": {
                "top": '1%',
                "left": '2%',
                "orient": 'horizontal',
                "data": [
                    {
                        "name": f"Plan {arm}",
                        "icon": 'rectangle'
                    }
                    for arm in arms
                ],
            },
            "visualMap": {
                "left": 'right',
                "top": 'top',
                "min": min_val,
                "max": max_val,
                "inRange": {
                    "color": [
                        '#313695',
                        '#4575b4',
                        '#74add1',
                        '#abd9e9',
                        '#e0f3f8',
                        '#ffffbf',
                        '#fee090',
                        '#fdae61',
                        '#f46d43',
                        '#d73027',
                        '#a50026'
                    ]
                },
                "text": ['High', 'Low'],
                "calculable": True
            },
            "series": [
                {
                    "type": "tree",
                    "name": f"Plan {arm}",
                    "data": [im_plan_data[arm]],
                    "top": f"12%",
                    "left": f"{2 + 100 // len(arms) * arm_idx}%",
                    "bottom": f"10%",
                    "right": f"{100 - (100 // len(arms) * (arm_idx + 1) - 2)}%",
                    "symbol": "circle",
                    # "roam": True,
                    "itemStyle": {"borderWidth": 0},
                    "symbolSize": label_size,
                    "orient": 'vertical',
                    "label": {
                        "position": "top",
                        # "rotate": -90,
                        "verticalAlign": "middle",
                        "align": "middle",
                        "fontSize": font_size,
                        "overflow": "break",
                        "formatter": "{c}"
                    },
                    "leaves": {
                        "label": {
                            "position": "bottom",
                            # "rotate": -90,
                            "verticalAlign": "middle",
                            "align": "middle",
                            "formatter": "{b}"
                        }
                    },
                    "expandAndCollapse": False,
                    "animationDuration": 550,
                    "animationDurationUpdate": 750,
                } for arm_idx, arm in enumerate(arms)],
        }
        st_echarts(opts_4, height=height, width="90%", key="variance_plan")
with col2:
    # Execute the query
    if not rerun:
        multi_results = st.session_state['results']
    else:
        multi_results = asyncio.run(rome_query_execution(query, nr_plans, arms))
        st.session_state['results'] = multi_results
        st.session_state['query_name'] = dataset_queries
        st.session_state['method'] = select_solver
    query_results = None
    columns = None
    query_time = sys.maxsize
    best_plan = -1
    for thread_ctr, r in enumerate(multi_results):
        if r[0] is not None:
            query_results = r[0]
            query_time = min(r[2], query_time)
            columns = r[1]
            if query_time == r[2]:
                best_plan = arms[thread_ctr]

    st.subheader("Query Results")
    if query_results is not None:
        query_dict = {col: [] for col in columns}
        for row in query_results:
            for idx, col in enumerate(columns):
                query_dict[col].append(row[idx])
        df = pd.DataFrame(query_dict)
        st.table(df)
        st.subheader("Query Performance")
        st.markdown(f"Planning time: **:green[{plan_time:.2f}]** seconds. "
                    f"Query execution time: **:red[{query_time:.2f}]** seconds.")
        # st.markdown(f"Query execution time: **:red[{query_time:.2f}]** seconds.")
        optimal_hints = _arm_idx_to_hints(best_plan)
        hint_dict = {h: "" for h in _ALL_OPTIONS}
        for hint in optimal_hints:
            hint_dict[hint.split(" ")[1]] = hint
        st.markdown(f"The optimal plan is **:blue[Plan {best_plan}]** with hints:")
        for h_key in _ALL_OPTIONS:
            st.markdown(hint_dict[h_key])
    else:
        if query_time > timeout // 1000:
            st.warning(f"Query execution timeout after {timeout} seconds")
        else:
            st.warning("Query execution failed")


st.subheader("About the ROME")
st.markdown("""
You can see how this works in the [Quickview Jupyter Notebook](https://github.com/losc-tutorial/quickview) or 
[see the code](https://github.com/Ultra-Seven/ParaRQO).
""")