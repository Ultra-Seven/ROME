from operators.join_node import JoinNode
from visitors.visitor import Visitor


# Cost function for Hash Join node:
# Cost = cpu_tuple_cost * (|Outer| - 1) + cpu_tuple_cost * |Result|
class MergeJoin(JoinNode):
    def accept(self, visitor: Visitor) -> None:
        visitor.visit_merge_join_node(self)

    def __init__(self, plan, postgres, children=None):
        super().__init__(plan, postgres, children)
        self.f_mean = self.plan_rows / self.children[0].plan_rows / self.children[1].plan_rows
        self.f_std = self.UNCERTAINTY * self.f_mean
        k_tuple = postgres.parameters["cpu_tuple_cost"]
        k_op = postgres.parameters["cpu_operator_cost"]
        self.c_std = ((k_tuple + k_op) ** 2 * self.children[0].d_std ** 2 +
                      k_op ** 2 * self.children[1].d_std ** 2) ** 0.5
