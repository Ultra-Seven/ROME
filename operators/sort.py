from operators.node import Node
from visitors.visitor import Visitor


# Cost function for hash node:
# cost = f * plan_rows
class Sort(Node):
    def accept(self, visitor: Visitor) -> None:
        visitor.visit_sort_node(self)

    def __init__(self, plan, postgres, children=None):
        super().__init__(plan, postgres, children)
        self.cost_per_card = self.cost / self.plan_rows
