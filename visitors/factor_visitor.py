from db.postgres import Postgres
from operators.node import Node
from visitors.visitor import Visitor


class FactorVisitor(Visitor):
    def __init__(self, postgres: Postgres):
        self.postgres = postgres
        self.join_keys = {}
        self.factor_variables = []

    def visit_hash_join_node(self, element: Node) -> None:
        outer_factor_product_dict = self.factor_variables[-2]
        inner_factor_product_dict = self.factor_variables[-1]
        if len(outer_factor_product_dict) > 0 and "c" not in outer_factor_product_dict:
            outer_factor_product = [k.f_key for k in element.children[0].f_nodes]
            product_key = "*".join(sorted(outer_factor_product))
            if product_key not in self.join_keys:
                self.join_keys[product_key] = {"coefficient": 0, "variables": []}
            if product_key not in outer_factor_product_dict:
                outer_factor_product_dict[product_key] = 0

            self.join_keys[product_key]["coefficient"] += \
                (self.postgres.parameters["cpu_operator_cost"] * element.children[0].card_product)
            outer_factor_product_dict[product_key] += \
                (self.postgres.parameters["cpu_operator_cost"] * element.children[0].card_product)
        else:
            if "c" not in self.join_keys:
                self.join_keys["c"] = {"coefficient": 0, "variables": []}
            if "c" not in outer_factor_product_dict:
                outer_factor_product_dict["c"] = 0
            self.join_keys["c"]["coefficient"] += (self.postgres.parameters["cpu_operator_cost"] * element.children[0].card_product)
            outer_factor_product_dict["c"] += \
                (self.postgres.parameters["cpu_operator_cost"] * element.children[0].card_product)

        for product_key in outer_factor_product_dict:
            if product_key not in inner_factor_product_dict:
                inner_factor_product_dict[product_key] = 0
            inner_factor_product_dict[product_key] += outer_factor_product_dict[product_key]

        result_factor_product = [k.f_key for k in element.f_nodes]
        product_key = "*".join(sorted(result_factor_product))
        if product_key not in self.join_keys:
            self.join_keys[product_key] = {"coefficient": 0, "variables": []}
        if product_key not in inner_factor_product_dict:
            inner_factor_product_dict[product_key] = 0

        self.join_keys[product_key]["coefficient"] += (
                    self.postgres.parameters["cpu_tuple_cost"] * element.card_product)
        inner_factor_product_dict[product_key] += (
                    self.postgres.parameters["cpu_tuple_cost"] * element.card_product)
        self.factor_variables = self.factor_variables[:-2] + [inner_factor_product_dict]

    # TODO: Fix the cost function later
    def visit_merge_join_node(self, element: Node) -> None:
        outer_factor_product_dict = self.factor_variables[-2]
        inner_factor_product_dict = self.factor_variables[-1]
        if len(outer_factor_product_dict) > 0 and "c" not in outer_factor_product_dict:
            outer_factor_product = [k.f_key for k in element.children[0].f_nodes]
            product_key = "*".join(sorted(outer_factor_product))
            if product_key not in self.join_keys:
                self.join_keys[product_key] = {"coefficient": 0, "variables": []}
            if product_key not in outer_factor_product_dict:
                outer_factor_product_dict[product_key] = 0

            self.join_keys[product_key]["coefficient"] += \
                (self.postgres.parameters["cpu_operator_cost"] * element.children[0].card_product)
            outer_factor_product_dict[product_key] += \
                (self.postgres.parameters["cpu_operator_cost"] * element.children[0].card_product)
        else:
            if "c" not in self.join_keys:
                self.join_keys["c"] = {"coefficient": 0, "variables": []}
            if "c" not in outer_factor_product_dict:
                outer_factor_product_dict["c"] = 0
            self.join_keys["c"]["coefficient"] += (self.postgres.parameters["cpu_operator_cost"] * element.children[0].card_product)
            outer_factor_product_dict["c"] += \
                (self.postgres.parameters["cpu_operator_cost"] * element.children[0].card_product)

        for product_key in outer_factor_product_dict:
            if product_key not in inner_factor_product_dict:
                inner_factor_product_dict[product_key] = 0
            inner_factor_product_dict[product_key] += outer_factor_product_dict[product_key]

        result_factor_product = [k.f_key for k in element.f_nodes]
        product_key = "*".join(sorted(result_factor_product))
        if product_key not in self.join_keys:
            self.join_keys[product_key] = {"coefficient": 0, "variables": []}
        if product_key not in inner_factor_product_dict:
            inner_factor_product_dict[product_key] = 0

        self.join_keys[product_key]["coefficient"] += (
                    self.postgres.parameters["cpu_tuple_cost"] * element.card_product)
        inner_factor_product_dict[product_key] += (
                    self.postgres.parameters["cpu_tuple_cost"] * element.card_product)
        self.factor_variables = self.factor_variables[:-2] + [inner_factor_product_dict]

    def visit_nested_loop_node(self, element: Node) -> None:
        outer_factor_product_dict = self.factor_variables[-2]
        inner_factor_product_dict = self.factor_variables[-1]

        # Outer cost
        if len(outer_factor_product_dict) > 0 and "c" not in outer_factor_product_dict:
            # TODO: product of two random variables
            if len(inner_factor_product_dict) > 0 and "c" not in inner_factor_product_dict:
                outer_factor_product = [k.f_key for k in element.children[0].f_nodes]
                product_key = "*".join(sorted(outer_factor_product))
                if product_key not in self.join_keys:
                    self.join_keys[product_key] = {"coefficient": 0, "variables": []}
                if product_key not in outer_factor_product_dict:
                    outer_factor_product_dict[product_key] = 0
                self.join_keys[product_key]["coefficient"] += (
                        element.children[0].card_product * element.children[1].cost)
                outer_factor_product_dict[product_key] += (
                        element.children[0].card_product * element.children[1].cost)
            else:
                outer_factor_product = [k.f_key for k in element.children[0].f_nodes]
                product_key = "*".join(sorted(outer_factor_product))
                if product_key not in self.join_keys:
                    self.join_keys[product_key] = {"coefficient": 0, "variables": []}
                if product_key not in outer_factor_product_dict:
                    outer_factor_product_dict[product_key] = 0
                self.join_keys[product_key]["coefficient"] += (
                            element.children[0].card_product * element.children[1].cost)
                outer_factor_product_dict[product_key] += (
                            element.children[0].card_product * element.children[1].cost)
        else:
            if len(inner_factor_product_dict) > 0 and "c" not in outer_factor_product_dict:
                for product_key in inner_factor_product_dict:
                    self.join_keys[product_key]["coefficient"] += (
                            element.children[0].plan_rows * element.children[1].cost)
                    outer_factor_product_dict[product_key] += (
                            element.children[0].plan_rows * element.children[1].cost)
            else:
                if "c" not in self.join_keys:
                    self.join_keys["c"] = {"coefficient": 0, "variables": []}
                if "c" not in outer_factor_product_dict:
                    outer_factor_product_dict["c"] = 0
                self.join_keys["c"]["coefficient"] += (element.children[0].card_product * element.children[1].cost)
                outer_factor_product_dict["c"] += (element.children[0].card_product * element.children[1].cost)

        # Output cost
        for product_key in outer_factor_product_dict:
            if product_key not in inner_factor_product_dict:
                inner_factor_product_dict[product_key] = 0
            inner_factor_product_dict[product_key] += outer_factor_product_dict[product_key]
        result_factor_product = [k.f_key for k in element.f_nodes]
        product_key = "*".join(sorted(result_factor_product))
        if product_key not in self.join_keys:
            self.join_keys[product_key] = {"coefficient": 0, "variables": []}
        if product_key not in inner_factor_product_dict:
            inner_factor_product_dict[product_key] = 0

        self.join_keys[product_key]["coefficient"] += (
                self.postgres.parameters["cpu_tuple_cost"] * element.card_product)
        inner_factor_product_dict[product_key] += (
                self.postgres.parameters["cpu_tuple_cost"] * element.card_product)

        self.factor_variables = self.factor_variables[:-2] + [inner_factor_product_dict]

    def visit_hash_node(self, element: Node) -> None:
        last_factor_product = self.factor_variables[-1]
        if len(last_factor_product) > 0 and "c" not in last_factor_product:
            result_factor_product = [k.f_key for k in element.f_nodes]
            product_key = "*".join(sorted(result_factor_product))
            if product_key not in self.join_keys:
                self.join_keys[product_key] = {"coefficient": 0, "variables": []}
            if product_key not in last_factor_product:
                last_factor_product[product_key] = 0
            self.join_keys[product_key]["coefficient"] += (element.cost_per_card * element.card_product)
            last_factor_product[product_key] += (element.cost_per_card * element.card_product)
        else:
            if "c" not in self.join_keys:
                self.join_keys["c"] = {"coefficient": 0, "variables": []}
            if "c" not in last_factor_product:
                last_factor_product["c"] = 0
            self.join_keys["c"]["coefficient"] += element.cost
            last_factor_product["c"] += element.cost

    def visit_aggregate_node(self, element: Node) -> None:
        last_factor_product = self.factor_variables[-1]
        if len(last_factor_product) > 0 and "c" not in last_factor_product:
            result_factor_product = [k.f_key for k in element.f_nodes]
            product_key = "*".join(sorted(result_factor_product))
            if product_key not in self.join_keys:
                self.join_keys[product_key] = {"coefficient": 0, "variables": []}
            if product_key not in last_factor_product:
                last_factor_product[product_key] = 0
            self.join_keys[product_key]["coefficient"] += (element.cost_per_card * element.card_product)
            last_factor_product[product_key] += (element.cost_per_card * element.card_product)
        else:
            if "c" not in self.join_keys:
                self.join_keys["c"] = {"coefficient": 0, "variables": []}
            if "c" not in last_factor_product:
                last_factor_product["c"] = 0
            self.join_keys["c"]["coefficient"] += element.cost
            last_factor_product["c"] += element.cost

    def visit_sort_node(self, element: Node) -> None:
        last_factor_product = self.factor_variables[-1]
        if len(last_factor_product) > 0 and "c" not in last_factor_product:
            result_factor_product = [k.f_key for k in element.f_nodes]
            product_key = "*".join(sorted(result_factor_product))
            if product_key not in self.join_keys:
                self.join_keys[product_key] = {"coefficient": 0, "variables": []}
            if product_key not in last_factor_product:
                last_factor_product[product_key] = 0
            self.join_keys[product_key]["coefficient"] += (element.cost_per_card * element.card_product)
            last_factor_product[product_key] += (element.cost_per_card * element.card_product)
        else:
            if "c" not in self.join_keys:
                self.join_keys["c"] = {"coefficient": 0, "variables": []}
            if "c" not in last_factor_product:
                last_factor_product["c"] = 0
            self.join_keys["c"]["coefficient"] += element.cost
            last_factor_product["c"] += element.cost

    def visit_gather_node(self, element: Node) -> None:
        last_factor_product = self.factor_variables[-1]
        if "c" not in self.join_keys:
            self.join_keys["c"] = {"coefficient": 0, "variables": []}
        if "c" not in last_factor_product:
            last_factor_product["c"] = 0
        self.join_keys["c"]["coefficient"] += 1000
        last_factor_product["c"] += 1000

    def visit_materialize_node(self, element: Node) -> None:
        last_factor_product = self.factor_variables[-1]
        if len(last_factor_product) > 0 and "c" not in last_factor_product:
            result_factor_product = [k.f_key for k in element.f_nodes]
            product_key = "*".join(sorted(result_factor_product))
            if product_key not in self.join_keys:
                self.join_keys[product_key] = {"coefficient": 0, "variables": []}
            if product_key not in last_factor_product:
                last_factor_product[product_key] = 0
            self.join_keys[product_key]["coefficient"] += (element.cost_per_card * element.card_product)
            last_factor_product[product_key] += (element.cost_per_card * element.card_product)
        else:
            if "c" not in self.join_keys:
                self.join_keys["c"] = {"coefficient": 0, "variables": []}
            if "c" not in last_factor_product:
                last_factor_product["c"] = 0
            self.join_keys["c"]["coefficient"] += element.cost
            last_factor_product["c"] += element.cost

    def visit_seq_scan_node(self, element: Node) -> None:
        # Seq Scan has no effect to join factors
        self.factor_variables.append({"c": element.cost})
        self.join_keys["c"] = {"coefficient": element.cost, "variables": []}

    def visit_index_scan_node(self, element: Node) -> None:
        # Seq Scan has no effect to join factors
        self.factor_variables.append({"c": element.cost})
        self.join_keys["c"] = {"coefficient": element.cost, "variables": []}

    def visit_bitmap_index_scan_node(self, element: Node) -> None:
        # Seq Scan has no effect to join factors
        self.factor_variables.append({"c": element.cost})
        self.join_keys["c"] = {"coefficient": element.cost, "variables": []}

    def visit_bitmap_heap_scan_node(self, element: Node) -> None:
        # Seq Scan has no effect to join factors
        self.factor_variables.append({"c": element.cost})
        self.join_keys["c"] = {"coefficient": element.cost, "variables": []}

    def visit_index_only_scan_node(self, element: Node) -> None:
        # Seq Scan has no effect to join factors
        self.factor_variables.append({"c": element.cost})
        self.join_keys["c"] = {"coefficient": element.cost, "variables": []}
