from db.postgres import Postgres
from operators.hash import Hash
from operators.node import Node
from visitors.visitor import Visitor


class IntermediateVisitor(Visitor):
    def __init__(self, postgres: Postgres):
        self.postgres = postgres
        self.join_keys = {}
        self.factor_variables = []

    def visit_hash_join_node(self, element: Node) -> None:
        left_node = element.children[0]
        right_node = element.children[1] if element.children[1].node_type != "Hash" \
            else element.children[1].children[0]
        factor = element.cost / (left_node.plan_rows + right_node.plan_rows)
        left_child_intermediate = left_node.f_key
        right_child_intermediate = right_node.f_key
        node_variables = {}

        if left_child_intermediate is not None:
            if left_child_intermediate not in node_variables:
                node_variables[left_child_intermediate] = 0
            node_variables[left_child_intermediate] += factor
        else:
            if "c" not in node_variables:
                node_variables["c"] = 0
            node_variables["c"] += (factor * left_node.plan_rows)
        if right_child_intermediate is not None:
            if right_child_intermediate not in node_variables:
                node_variables[right_child_intermediate] = 0
            node_variables[right_child_intermediate] += factor
        else:
            if "c" not in node_variables:
                node_variables["c"] = 0
            node_variables["c"] += (factor * right_node.plan_rows)
        for key, value in node_variables.items():
            if key not in self.join_keys:
                self.join_keys[key] = {"coefficient": 0, "variables": []}
            self.join_keys[key]["coefficient"] += value
        self.factor_variables = self.factor_variables[:-2] + [node_variables]

    def visit_merge_join_node(self, element: Node) -> None:
        k_tuple = self.postgres.parameters["cpu_tuple_cost"]
        k_op = self.postgres.parameters["cpu_operator_cost"]
        left_child_intermediate = element.children[0].f_key
        right_child_intermediate = element.children[1].f_key
        node_variables = {}
        if left_child_intermediate is not None:
            if left_child_intermediate not in node_variables:
                node_variables[left_child_intermediate] = 0
            node_variables[left_child_intermediate] += (k_tuple + k_op)

        if right_child_intermediate is not None:
            if right_child_intermediate not in node_variables:
                node_variables[right_child_intermediate] = 0
            node_variables[right_child_intermediate] += k_op
        if left_child_intermediate is None and right_child_intermediate is None:
            if "c" not in node_variables:
                node_variables["c"] = 0
            node_variables["c"] += element.cost
        elif left_child_intermediate is not None and right_child_intermediate is None:
            if "c" not in node_variables:
                node_variables["c"] = 0
            node_variables["c"] += max(0, element.cost -
                                                      (k_tuple + k_op) * element.children[0].plan_rows)
        elif left_child_intermediate is None and right_child_intermediate is not None:
            if "c" not in node_variables:
                node_variables["c"] = 0
            node_variables["c"] += max(0, element.cost -
                                       k_op * element.children[1].plan_rows)
        for key, value in node_variables.items():
            if key not in self.join_keys:
                self.join_keys[key] = {"coefficient": 0, "variables": []}
            self.join_keys[key]["coefficient"] += value
        self.factor_variables = self.factor_variables[:-2] + [node_variables]

    def visit_nested_loop_node(self, element: Node) -> None:
        k_tuple = self.postgres.parameters["cpu_tuple_cost"]
        left_child_intermediate = element.children[0].f_key
        right_child_intermediate = element.children[1].f_key
        join_intermediate = element.f_key
        node_variables = {}
        if left_child_intermediate is not None:
            if left_child_intermediate not in node_variables:
                node_variables[left_child_intermediate] = 0
            if element.children[1].node_type == "Materialize":
                node_variables[left_child_intermediate] += 0.0125
            elif element.children[1].c_std == 0:
                node_variables[left_child_intermediate] += element.children[1].cost
            else:
                product_key = left_child_intermediate + "*" + right_child_intermediate
                if product_key not in node_variables:
                    node_variables[product_key] = 0
                node_variables[product_key] += 1
        elif right_child_intermediate is not None:
            nr_left_plans = element.children[0].plan_rows
            right_node_variables = self.factor_variables[-1]
            for key, value in right_node_variables.items():
                if key not in node_variables:
                    node_variables[key] = 0
                node_variables[key] += nr_left_plans * value

        elif left_child_intermediate is None and right_child_intermediate is None:
            if "c" not in node_variables:
                node_variables["c"] = 0
            node_variables["c"] += (element.cost - k_tuple * element.plan_rows)
        if join_intermediate is not None:
            if join_intermediate not in node_variables:
                node_variables[join_intermediate] = 0
            node_variables[join_intermediate] += k_tuple
        for key, value in node_variables.items():
            if key not in self.join_keys:
                self.join_keys[key] = {"coefficient": 0, "variables": []}
            self.join_keys[key]["coefficient"] += value
        self.factor_variables = self.factor_variables[:-2] + [node_variables]

    def visit_hash_node(self, element: Node) -> None:
        intermediate_key = element.f_key
        node_variables = {}
        if intermediate_key is not None and element.d_std > 0:
            if intermediate_key not in node_variables:
                node_variables[intermediate_key] = 0
            node_variables[intermediate_key] += element.cost_per_card
        else:
            if "c" not in node_variables:
                node_variables["c"] = 0
            node_variables["c"] += element.cost
        for key, value in node_variables.items():
            if key not in self.join_keys:
                self.join_keys[key] = {"coefficient": 0, "variables": []}
            self.join_keys[key]["coefficient"] += value
        self.factor_variables = self.factor_variables[:-1] + [node_variables]

    def visit_aggregate_node(self, element: Node) -> None:
        intermediate_key = element.children[0].f_key
        node_variables = {}
        cost_per_card = element.cost / element.children[0].plan_rows
        if intermediate_key is not None and element.children[0].d_std > 0:
            if intermediate_key not in node_variables:
                node_variables[intermediate_key] = 0
            node_variables[intermediate_key] += cost_per_card
        else:
            if "c" not in node_variables:
                node_variables["c"] = 0
            node_variables["c"] += element.cost
        for key, value in node_variables.items():
            if key not in self.join_keys:
                self.join_keys[key] = {"coefficient": 0, "variables": []}
            self.join_keys[key]["coefficient"] += value
        self.factor_variables = self.factor_variables[:-1] + [node_variables]

    def visit_sort_node(self, element: Node) -> None:
        intermediate_key = element.f_key
        node_variables = {}
        if intermediate_key is not None and element.d_std > 0:
            if intermediate_key not in node_variables:
                node_variables[intermediate_key] = 0
            node_variables[intermediate_key] += element.cost_per_card
        else:
            if "c" not in node_variables:
                node_variables["c"] = 0
            node_variables["c"] += element.cost
        for key, value in node_variables.items():
            if key not in self.join_keys:
                self.join_keys[key] = {"coefficient": 0, "variables": []}
            self.join_keys[key]["coefficient"] += value
        self.factor_variables = self.factor_variables[:-1] + [node_variables]

    def visit_gather_node(self, element: Node) -> None:
        if "c" not in self.join_keys:
            self.join_keys["c"] = {"coefficient": 0, "variables": []}
        self.join_keys["c"]["coefficient"] += 1000

    def visit_materialize_node(self, element: Node) -> None:
        intermediate_key = element.f_key
        node_variables = {}
        if intermediate_key is not None and element.d_std > 0:
            if intermediate_key not in node_variables:
                node_variables[intermediate_key] = 0
            node_variables[intermediate_key] += element.cost_per_card
        else:
            if "c" not in node_variables:
                node_variables["c"] = 0
            node_variables["c"] += element.cost
        for key, value in node_variables.items():
            if key not in self.join_keys:
                self.join_keys[key] = {"coefficient": 0, "variables": []}
            self.join_keys[key]["coefficient"] += value
        self.factor_variables = self.factor_variables[:-1] + [node_variables]

    def visit_seq_scan_node(self, element: Node) -> None:
        # Seq Scan has no effect to join factors
        if "c" not in self.join_keys:
            self.join_keys["c"] = {"coefficient": 0, "variables": []}
        self.join_keys["c"]["coefficient"] += element.cost
        self.factor_variables.append({"c": element.cost})

    def visit_index_scan_node(self, element: Node) -> None:
        # Seq Scan has no effect to join factors
        if "c" not in self.join_keys:
            self.join_keys["c"] = {"coefficient": 0, "variables": []}
        self.join_keys["c"]["coefficient"] += element.cost
        self.factor_variables.append({"c": element.cost})

    def visit_bitmap_index_scan_node(self, element: Node) -> None:
        # Seq Scan has no effect to join factors
        if "c" not in self.join_keys:
            self.join_keys["c"] = {"coefficient": 0, "variables": []}
        self.join_keys["c"]["coefficient"] += element.cost
        self.factor_variables.append({"c": element.cost})

    def visit_bitmap_heap_scan_node(self, element: Node) -> None:
        # Seq Scan has no effect to join factors
        if "c" not in self.join_keys:
            self.join_keys["c"] = {"coefficient": 0, "variables": []}
        self.join_keys["c"]["coefficient"] += element.cost
        self.factor_variables.append({"c": element.cost})

    def visit_index_only_scan_node(self, element: Node) -> None:
        # Seq Scan has no effect to join factors
        if "c" not in self.join_keys:
            self.join_keys["c"] = {"coefficient": 0, "variables": []}
        self.join_keys["c"]["coefficient"] += element.cost
        self.factor_variables.append({"c": element.cost})
