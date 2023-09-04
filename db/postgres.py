from utils.pg_utils import create_connection, get_psql_param, get_all_tables, get_table_size, get_table_estimates, \
    get_stats_table, get_base_table_selectivity
from utils.sql_utils import extract_tables


class Postgres:
    def __init__(self, host, port, user, password, database):
        self.alias_to_rows = None
        self.stats_table = None
        self.sql = None
        self.query_name = None
        self.alias_to_tables = None
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connection = create_connection(database, user, password, host, port)
        self.cursor = None
        self.parameters = {}
        self.table_sizes = {}
        self.table_records = {}
        self.initialize_parameters()
        table_names = get_all_tables(self.connection)
        for table_name in table_names:
            self.table_sizes[table_name] = get_table_size(self.connection, table_name)
            self.table_records[table_name] = get_table_estimates(self.connection, table_name)
        self.retrieve_stats()
        print("Initialized Postgres connection")
        print(f"Parameters: {self.parameters}")

    def initialize_parameters(self):
        self.parameters["block_size"] = int(get_psql_param(self.connection, "block_size"))
        self.parameters["seq_page_cost"] = float(get_psql_param(self.connection, "seq_page_cost"))
        self.parameters["cpu_tuple_cost"] = float(get_psql_param(self.connection, "cpu_tuple_cost"))
        self.parameters["cpu_operator_cost"] = float(get_psql_param(self.connection, "cpu_operator_cost"))
        self.parameters["random_page_cost"] = float(get_psql_param(self.connection, "random_page_cost"))
        self.parameters["cpu_index_tuple_cost"] = float(get_psql_param(self.connection, "cpu_index_tuple_cost"))
        self.parameters["parallel_tuple_cost"] = float(get_psql_param(self.connection, "parallel_tuple_cost"))
        self.parameters["parallel_setup_cost"] = float(get_psql_param(self.connection, "parallel_setup_cost"))

    def retrieve_stats(self):
        self.stats_table = get_stats_table(self.connection, list(self.table_sizes.keys()))

    def set_sql_query(self, sql, query_name):
        self.sql = sql
        self.query_name = query_name
        self.alias_to_rows = get_base_table_selectivity(self)
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
        self.alias_to_tables = extract_tables(sql)

    def close(self):
        self.connection.close()
