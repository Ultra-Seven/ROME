from abc import ABC, abstractmethod

from visitors.visitor import Visitor


class Node(ABC):
    def __init__(self, plan, postgres, children=None):
        self.UNCERTAINTY = postgres.uncertainty
        self.plan_rows = plan["Plan Rows"]
        self.total_cost = plan["Total Cost"]
        self.startup_cost = plan["Startup Cost"]
        self.node_type = plan["Node Type"]
        for key in plan:
            if key != "Plans":
                setattr(self, "_".join(key.lower().split()), plan[key])

        self.cost = self.total_cost
        self.children = []
        self.parent = None
        self.f_mean = 1
        self.f_std = -1

        self.f_nodes = set()
        self.u_nodes = set()
        self.buckets = {}
        self.d_mean = self.plan_rows
        self.d_std = 0
        self.card_product = 1
        self.filter_product = 1
        self.join_product = 1
        # Propagate the cardinality product
        for child_node in children:
            child_node.add_parent(self)
            self.card_product = self.card_product * child_node.card_product
            self.filter_product = self.filter_product * child_node.filter_product
            self.join_product = self.join_product * child_node.join_product
            self.d_std = max(self.d_std, child_node.d_std)
            for f_node in child_node.f_nodes:
                self.f_nodes.add(f_node)
            for u_node in child_node.u_nodes:
                self.u_nodes.add(u_node)
        self.cost_per_card = self.cost / self.plan_rows
        # Propagate the buckets
        self.generate_buckets()
        # Linear cost
        self.c_mean = self.cost
        self.c_std = self.cost_per_card * self.d_std

        # The node is a scan node
        if "Alias" in plan:
            self.card_product = postgres.table_records[plan["Relation Name"]]
        self.f_key = None if len(self.children) == 0 else self.children[0].f_key

    def add_parent(self, parent):
        self.parent = parent
        parent.children.append(self)
        parent.cost = parent.cost - self.total_cost

    def is_join_node(self):
        return False

    def generate_buckets(self):
        for child_node in self.children:
            self.buckets.update(child_node.buckets)

    @abstractmethod
    def accept(self, visitor: Visitor) -> None:
        pass

    def __str__(self):
        return f"[{self.node_type}]:\nM:{self.d_mean}\nV:{round(self.d_std, 1)}\nC:{round(self.total_cost, 1)}"
