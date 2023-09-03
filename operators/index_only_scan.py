from operators.node import Node
from visitors.visitor import Visitor


# Cost function for Index Only Seq Scan:
# Cost = (disk pages read * seq_page_cost) + (rows scanned * cpu_tuple_cost)
class IndexOnlyScan(Node):
    def accept(self, visitor: Visitor) -> None:
        visitor.visit_index_only_scan_node(self)

    def __init__(self, plan, postgres, children=None):
        super().__init__(plan, postgres, children)
        self.card_product = postgres.table_records[plan["Relation Name"]]
        self.filter_product = 0 if self.card_product == 0 else self.plan_rows / self.card_product
        self.f_mean = 0 if self.card_product == 0 else self.plan_rows / self.card_product

    def __str__(self):
        return f"[{self.node_type}]:\n{self.relation_name}"
