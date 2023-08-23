from db.postgres import Postgres
from operators.hash import Hash
from operators.node import Node
from utils.sql_utils import get_join_predicates
from visitors.visitor import Visitor


def is_join_predicate(predicate, left_tables, right_tables):
    alias = [term.strip().split(".")[0] for term in predicate.split("=")]
    if (alias[0] in left_tables and alias[1] in right_tables) or (alias[1] in left_tables and alias[0] in right_tables):
        return True
    else:
        return False


class PredicateVisitor(Visitor):
    def __init__(self, postgres: Postgres, sql: str):
        self.postgres = postgres
        self.sql = sql
        self.join_predicates = get_join_predicates(sql)
        self.predicates_variables = []
        self.intermediate_to_predicates = {}
        self.predicate_dists = {}

    def visit_hash_join_node(self, element: Node) -> None:
        left_node_variables = self.predicates_variables[-2]
        right_node_variables = self.predicates_variables[-1]
        left_child = element.children[0]
        right_child = element.children[1]
        left_tables = left_child.tables
        right_tables = right_child.tables
        predicates_list = [p for p in self.join_predicates
                           if is_join_predicate(p, left_tables, right_tables)]
        # is_pkfk = sum([int(".id" not in p) for p in predicates_list]) == 0
        is_pkfk = 0
        uncertainty = element.UNCERTAINTY if element.f_key != "mi:it1" else 10
        if is_pkfk:
            predicates_dist = {"load": (element.f_mean, 0)}
        else:
            predicates_dist = {" AND ".join(predicates_list):
                                   (element.f_mean, 0 if is_pkfk else uncertainty * element.f_mean)}
        load_f = 1
        if "load" in left_node_variables:
            load_f = load_f * left_node_variables["load"][0]
        if "load" in right_node_variables:
            load_f = load_f * right_node_variables["load"][0]
        if "load" in predicates_dist:
            load_f = load_f * predicates_dist["load"][0]
        predicates_dist.update(left_node_variables)
        predicates_dist.update(right_node_variables)
        if "load" in predicates_dist:
            predicates_dist["load"] = (load_f, 0)
        self.intermediate_to_predicates[element.f_key] = predicates_dist
        self.predicates_variables = self.predicates_variables[:-2] + [predicates_dist]

    def visit_merge_join_node(self, element: Node) -> None:
        left_node_variables = self.predicates_variables[-2]
        right_node_variables = self.predicates_variables[-1]
        left_child = element.children[0]
        right_child = element.children[1]
        left_tables = left_child.tables
        right_tables = right_child.tables
        predicates_list = [p for p in self.join_predicates
                           if is_join_predicate(p, left_tables, right_tables)]
        # is_pkfk = sum([int(".id" not in p) for p in predicates_list]) == 0
        is_pkfk = 0
        uncertainty = element.UNCERTAINTY if element.f_key != "mi:it1" else 10
        if is_pkfk:
            predicates_dist = {"load": (element.f_mean, 0)}
        else:
            predicates_dist = {" AND ".join(predicates_list):
                                   (element.f_mean, 0 if is_pkfk else uncertainty * element.f_mean)}
        load_f = 1
        if "load" in left_node_variables:
            load_f = load_f * left_node_variables["load"][0]
        if "load" in right_node_variables:
            load_f = load_f * right_node_variables["load"][0]
        if "load" in predicates_dist:
            load_f = load_f * predicates_dist["load"][0]
        predicates_dist.update(left_node_variables)
        predicates_dist.update(right_node_variables)
        if "load" in predicates_dist:
            predicates_dist["load"] = (load_f, 0)
        self.intermediate_to_predicates[element.f_key] = predicates_dist
        self.predicates_variables = self.predicates_variables[:-2] + [predicates_dist]

    def visit_nested_loop_node(self, element: Node) -> None:
        left_node_variables = self.predicates_variables[-2]
        right_node_variables = self.predicates_variables[-1]
        left_child = element.children[0]
        right_child = element.children[1]
        left_tables = left_child.tables
        right_tables = right_child.tables
        predicates_list = [p for p in self.join_predicates
                           if is_join_predicate(p, left_tables, right_tables)]
        # is_pkfk = sum([int(".id" not in p) for p in predicates_list]) == 0
        is_pkfk = 0
        uncertainty = element.UNCERTAINTY if element.f_key != "mi:it1" else 10
        if is_pkfk:
            predicates_dist = {"load": (element.f_mean, 0)}
        else:
            predicates_dist = {" AND ".join(predicates_list):
                                   (element.f_mean, 0 if is_pkfk else uncertainty * element.f_mean)}
        load_f = 1
        if "load" in left_node_variables:
            load_f = load_f * left_node_variables["load"][0]
        if "load" in right_node_variables:
            load_f = load_f * right_node_variables["load"][0]
        if "load" in predicates_dist:
            load_f = load_f * predicates_dist["load"][0]
        predicates_dist.update(left_node_variables)
        predicates_dist.update(right_node_variables)
        if "load" in predicates_dist:
            predicates_dist["load"] = (load_f, 0)
        self.intermediate_to_predicates[element.f_key] = predicates_dist
        self.predicates_variables = self.predicates_variables[:-2] + [predicates_dist]


    def visit_hash_node(self, element: Node) -> None:
        pass

    def visit_aggregate_node(self, element: Node) -> None:
        pass

    def visit_sort_node(self, element: Node) -> None:
        pass

    def visit_gather_node(self, element: Node) -> None:
        pass

    def visit_materialize_node(self, element: Node) -> None:
        pass

    def visit_seq_scan_node(self, element: Node) -> None:
        predicates_dist = {}
        if hasattr(element, "filter"):
            if element.filter.count("=") == 1:
                predicates_dist = {"load": (element.f_mean, 0)}
            else:
                predicates_dist = {element.filter: (element.f_mean, element.UNCERTAINTY * element.f_mean)}
        elif element.f_mean < 1:
            predicates_dist = {"load": (element.f_mean, 0)}
        if "load" not in predicates_dist:
            predicates_dist["load"] = (1, 0)
        predicates_dist["load"] = (predicates_dist["load"][0] * element.card_product, 0)
        if len(element.children) == 0:
            self.predicates_variables.append(predicates_dist)

    def visit_index_scan_node(self, element: Node) -> None:
        predicates_dist = {}
        if hasattr(element, "filter"):
            if element.filter.count("=") == 1:
                predicates_dist = {"load": (element.f_mean, 0)}
            else:
                predicates_dist = {element.filter: (element.f_mean, element.UNCERTAINTY * element.f_mean)}
        elif element.f_mean < 1:
            predicates_dist = {"load": (element.f_mean, 0)}
        if "load" not in predicates_dist:
            predicates_dist["load"] = (1, 0)
        predicates_dist["load"] = (predicates_dist["load"][0] * element.card_product, 0)
        if len(element.children) == 0:
            self.predicates_variables.append(predicates_dist)

    def visit_bitmap_index_scan_node(self, element: Node) -> None:
        predicates_dist = {}
        if hasattr(element, "filter"):
            if element.filter.count("=") == 1:
                predicates_dist = {"load": (element.f_mean, 0)}
            else:
                predicates_dist = {element.filter: (element.f_mean, element.UNCERTAINTY * element.f_mean)}
        elif element.f_mean < 1:
            predicates_dist = {"load": (element.f_mean, 0)}
        if "load" not in predicates_dist:
            predicates_dist["load"] = (1, 0)
        predicates_dist["load"] = (predicates_dist["load"][0] * element.card_product, 0)
        if len(element.children) == 0:
            self.predicates_variables.append(predicates_dist)

    def visit_bitmap_heap_scan_node(self, element: Node) -> None:
        predicates_dist = {}
        if hasattr(element, "filter"):
            if element.filter.count("=") == 1:
                predicates_dist = {"load": (element.f_mean, 0)}
            else:
                predicates_dist = {element.filter: (element.f_mean, element.UNCERTAINTY * element.f_mean)}
        elif element.f_mean < 1:
            predicates_dist = {"load": (element.f_mean, 0)}
        if "load" not in predicates_dist:
            predicates_dist["load"] = (1, 0)
        predicates_dist["load"] = (predicates_dist["load"][0] * element.card_product, 0)
        if len(element.children) == 0:
            self.predicates_variables.append(predicates_dist)

    def visit_index_only_scan_node(self, element: Node) -> None:
        predicates_dist = {}
        if hasattr(element, "filter"):
            if element.filter.count("=") == 1:
                predicates_dist = {"load": (element.f_mean, 0)}
            else:
                predicates_dist = {element.filter: (element.f_mean, element.UNCERTAINTY * element.f_mean)}
        elif element.f_mean < 1:
            predicates_dist = {"load": (element.f_mean, 0)}
        if "load" not in predicates_dist:
            predicates_dist["load"] = (1, 0)
        predicates_dist["load"] = (predicates_dist["load"][0] * element.card_product, 0)
        if len(element.children) == 0:
            self.predicates_variables.append(predicates_dist)
