from operators.node import Node
from visitors.visitor import Visitor


# Cost function for Seq Scan:
# Cost = (disk pages read * seq_page_cost) + (rows scanned * cpu_tuple_cost)
class BitmapIndexScan(Node):
    def accept(self, visitor: Visitor) -> None:
        visitor.visit_bitmap_index_scan_node(self)

    def __init__(self, plan, postgres, children=None):
        super().__init__(plan, postgres, children)
        if "Relation Name" in plan:
            self.card_product = postgres.table_records[plan["Relation Name"]]
            self.filter_product = 0 if self.card_product == 0 else self.plan_rows / self.card_product
            self.f_mean = 0 if self.card_product == 0 else self.plan_rows / self.card_product
            # if "Filter" in plan and "%" in plan["Filter"]:
            #     self.buckets = {self.plan_rows: 1 - self.UNCERTAINTY, 10 * self.plan_rows: self.UNCERTAINTY}
            # else:
            #     self.buckets = {self.plan_rows: 1}
            self.buckets = {self.plan_rows: 1}

    def __str__(self):
        return f"[{self.node_type}]:\n{self.index_name}"
