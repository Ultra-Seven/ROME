from operators.node import Node
from visitors.visitor import Visitor


# Cost function for hash node:
# cost = f * plan_rows
class Hash(Node):
    def accept(self, visitor: Visitor) -> None:
        visitor.visit_hash_node(self)

    def __init__(self, plan, postgres, children=None):
        super().__init__(plan, postgres, children)
