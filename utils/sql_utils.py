import re


def extract_tables(sql):
    table_to_alias = {}
    # table_items = sql.split(" FROM ")[-1].split(" WHERE")[0].strip().split(", ")
    table_items = []
    for clause in sql.split(" FROM ")[1:]:
        for t in clause.split(" WHERE")[0].strip().split(", "):
            table_items.append(t.strip())
    for item in table_items:
        item_list = item.strip().split()
        if " AS " in item:
            table, alias = item.split(" AS ")
            table_to_alias[alias.strip()] = table.strip()
        elif len(item_list) > 1:
            table, alias = item_list
            table_to_alias[alias.strip()] = table.strip()
        else:
            table_to_alias[item.strip()] = item.strip()
    return table_to_alias


def generate_from_clause(alias_list, alias_dict):
    from_clause = ""
    for alias in alias_list:
        if from_clause == "":
            from_clause += f"{alias_dict[alias]} AS {alias}"
        elif "CROSS JOIN" not in from_clause:
            right_join = f"{alias_dict[alias]} AS {alias}"
            from_clause = f"{from_clause} CROSS JOIN {right_join}"
        else:
            right_join = f"{alias_dict[alias]} AS {alias}"
            from_clause = f"({from_clause}) CROSS JOIN {right_join}"
    return from_clause


def get_join_predicates(sql):
    join_1 = re.findall(r"\w+\.\w+\s?=\s?\w+\.\w+", sql)
    join_1 += re.findall(r"\w+\.\w+\s?>\s?\w+\.\w+", sql)
    join_1 += re.findall(r"\w+\.\w+\s?<\s?\w+\.\w+", sql)
    join_1 += re.findall(r"\w+\.\w+\s?>=\s?\w+\.\w+", sql)
    join_1 += re.findall(r"\w+\.\w+\s?<=\s?\w+\.\w+", sql)
    return join_1
