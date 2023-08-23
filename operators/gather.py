from operators.node import Node
from visitors.visitor import Visitor


# Cost function for Gather node:
# parallel_setup_cost(1000) + parallel_tuple_cost(0.1) * total_tuples
class Gather(Node):
    def accept(self, visitor: Visitor) -> None:
        visitor.visit_gather_node(self)

    def __init__(self, plan, postgres, children=None):
        super().__init__(plan, postgres, children)
        self.f_mean = -1
        self.f_std = -1
        self.c_std = 0
