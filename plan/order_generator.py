import random
import copy
from utils.sql_utils import extract_tables, generate_from_clause


class Generator:
    def __init__(self, sql):
        self.tables = extract_tables(sql)
        self.sql = sql

    def generate(self, k):
        from_clause = self.sql.split(" FROM ")[-1].split(" WHERE")[0]
        new_sqls = []
        relations = list(self.tables.keys())
        for x in range(k):
            new_order = relations.copy()
            new_sql = copy.copy(self.sql)
            random.shuffle(new_order)
            new_from = generate_from_clause(new_order, self.tables)
            new_sqls.append(new_sql.replace(from_clause, new_from))
        return new_sqls

