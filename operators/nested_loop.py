from operators.join_node import JoinNode
from visitors.visitor import Visitor


# Cost function for Nested Loop node:
# Cost = (|Outer| - 1) * Cost of inner + cpu_tuple_cost * |Result|
class NestedLoop(JoinNode):
    def accept(self, visitor: Visitor) -> None:
        visitor.visit_nested_loop_node(self)

    def __init__(self, plan, postgres, children=None):
        super().__init__(plan, postgres, children)
        self.f_mean = self.plan_rows / self.children[0].plan_rows / self.children[1].plan_rows
        self.f_std = self.UNCERTAINTY * self.f_mean
        k_tuple = postgres.parameters["cpu_tuple_cost"]
        # The inner node is materialized
        if self.children[1].node_type == "Materialize":
            self.c_std = ((k_tuple * self.d_std) ** 2 + (self.children[0].d_std * 0.0125) ** 2) ** 0.5
        # The inner node has no uncertainty
        elif self.children[1].c_std == 0:
            self.c_std = ((k_tuple * self.d_std) ** 2 + (self.children[0].d_std * self.children[1].cost) ** 2) ** 0.5
        # The inner node has uncertainty
        else:
            product_var = (self.children[0].d_std ** 2) * self.children[1].c_mean ** 2 + \
                          (self.children[1].c_std ** 2) * self.children[0].d_mean ** 2
            self.c_std = ((k_tuple * self.d_std) ** 2 + product_var) ** 0.5
        # if hasattr(self.children[1], "index_cond"):
        #     print("Index Cond: ", self.children[0].tables, self.children[1].tables, self.children[1].index_cond)
        # elif hasattr(self.children[1], "recheck_cond"):
        #     print("Recheck Cond: ", self.children[0].tables, self.children[1].tables, self.children[1].recheck_cond)
        # else:
        #     print("Nested Loop Cond: ", self.children[0].tables, self.children[1].tables,
        #           self.join_filter + " " + self.children[1].node_type)
