import json
import re
import subprocess
import psycopg2

from utils.sql_utils import get_join_predicates, extract_tables


def create_connection(dbname, user, password, host, port):
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )
    return conn


def close_connection(conn):
    conn.close()


def get_psql_param(conn, parameter):
    cur = conn.cursor()
    cur.execute(f"SHOW {parameter}")
    return cur.fetchone()[0]


def get_all_tables(conn):
    cur = conn.cursor()
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    results = cur.fetchall()
    return [t[0] for t in results]


def get_table_size(conn, table):
    cur = conn.cursor()
    cur.execute(f"select pg_relation_size('{table}')")
    return cur.fetchone()[0]


def get_table_estimates(conn, table):
    cur = conn.cursor()
    cur.execute(f"SELECT reltuples AS estimate FROM pg_class where relname = '{table}'")
    return cur.fetchone()[0]


def get_stats_table(conn, table_names):
    cur = conn.cursor()
    table_sql = ", ".join(list(map(lambda x: f"'{x}'", table_names)))
    cur.execute(f"select * from pg_stats where tablename IN ({table_sql})")
    columns = [desc[0] for desc in cur.description]
    results = cur.fetchall()
    stats_table = {}
    for result in results:
        stats_dict = {col: result[idx] for idx, col in enumerate(columns)}
        table_name = stats_dict["tablename"]
        att_name = stats_dict["attname"]
        if table_name not in stats_table:
            stats_table[table_name] = {}
        stats_table[table_name][att_name] = stats_dict
    return stats_table


def get_predicate_selectivity(postgres, joined_tables, next_table, connected_cols=None):
    sql = postgres.sql
    database = postgres.database
    predicates = sql.split(" WHERE ")[-1].strip().split(" AND ")
    all_alias = set(joined_tables).union({next_table})
    predicates = [p.strip() for p in predicates]
    all_join_predicates = get_join_predicates(sql)
    join_predicates = [p for p in all_join_predicates
                       if set([cols.split(".")[0].strip()
                               for cols in p.split("=")]).issubset(joined_tables) or
                       frozenset([cols.strip() for cols in p.split("=")]) == connected_cols]

    unary_predicates = [p for p in predicates if p not in all_join_predicates and
                        re.findall(r"\w+.\w+", p)[0].split(".")[0] in all_alias]
    alias_to_tables = extract_tables(sql)

    target_sql = "SELECT * FROM " + " CROSS JOIN ".join([(alias_to_tables[a] + " AS " + a)
                                                         for a in joined_tables.union({next_table})]) \
                 + " WHERE " + " AND ".join(unary_predicates + join_predicates)

    target_sql = "SET join_collapse_limit TO 1;\n" + \
                 "\nEXPLAIN (COSTS, VERBOSE, FORMAT JSON) " + target_sql
    result = subprocess.run(['psql', '-h', 'localhost',
                             '-U', 'postgres', '-d', database, '-XqAt', '-c', target_sql],
                            stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    nr_rows = int(json.loads(result)[0]["Plan"]["Plan Rows"])
    table_rows = int(postgres.table_records[alias_to_tables[next_table]])
    return nr_rows / table_rows


def get_base_table_selectivity(postgres):
    sql = postgres.sql.strip()
    if sql[-1] == ";":
        sql = sql[:-1]
    sql = sql.replace("and", "AND")
    sql = sql.replace("And", "AND")
    sql = sql.replace("between", "BETWEEN")
    sql = sql.replace("Between", "BETWEEN")
    sql = sql.replace("\n", " ")
    sql = sql.replace("\t", " ")
    sql = sql.replace(" where ", " WHERE ")
    sql = sql.replace(" from ", " FROM ")
    predicates = sql.split(" WHERE ")[-1].strip().split(" AND ")
    cursor = 0
    while cursor < len(predicates):
        if len(re.findall(r"\w+\.\w+\s*BETWEEN", predicates[cursor])) > 0:
            predicate = " AND ".join(predicates[cursor: cursor+2])
            predicates = predicates[:cursor] + [predicate] + predicates[cursor+2:]
        cursor += 1
    database = postgres.database
    alias_to_tables = extract_tables(sql)
    print(sql, alias_to_tables)
    alias_to_unary = {k: [] for k in alias_to_tables.keys()}
    alias_to_rows = {k: 1 for k in alias_to_tables.keys()}
    all_join_predicates = get_join_predicates(sql)
    unary_predicates = [p for p in predicates if p not in all_join_predicates]
    for unary_predicate in unary_predicates:
        alias = re.findall(r"\w+\.\w+", unary_predicate)[0].split(".")[0]
        alias_to_unary[alias].append(unary_predicate)

    for a in alias_to_tables.keys():
        where_clause = "" if len(alias_to_unary[a]) == 0 else " WHERE " + " AND ".join(alias_to_unary[a])
        # target_sql = "SELECT * FROM " + alias_to_tables[a] + " AS " + a + where_clause
        target_sql = "SELECT * FROM " + alias_to_tables[a] + " AS " + a
        target_sql = "EXPLAIN (COSTS, VERBOSE, FORMAT JSON) " + target_sql
        result = subprocess.run(['psql', '-h', 'localhost', '-U', 'postgres', '-d', database, '-XqAt', '-c', target_sql],
                                stdout=subprocess.PIPE)
        result = result.stdout.decode('utf-8')
        nr_rows = int(json.loads(result)[0]["Plan"]["Plan Rows"])
        alias_to_rows[a] = nr_rows
    return alias_to_rows
