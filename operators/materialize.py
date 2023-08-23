from operators.node import Node
from visitors.visitor import Visitor


# Cost function for Materialize node:
# TODO: Cost = 0.005 * |Result|?
class Materialize(Node):
    def accept(self, visitor: Visitor) -> None:
        visitor.visit_materialize_node(self)

    def __init__(self, plan, postgres, children=None):
        super().__init__(plan, postgres, children)
        self.cost_per_card = self.cost / self.plan_rows
        self.f_mean = -1
        self.f_std = -1
