from operators.join_node import JoinNode
from visitors.visitor import Visitor


# Cost function for Hash Join node:
# Cost = K (M + N)
class HashJoin(JoinNode):
    def accept(self, visitor: Visitor) -> None:
        visitor.visit_hash_join_node(self)

    def __init__(self, plan, postgres, children=None):
        super().__init__(plan, postgres, children)
        left_node = self.children[0]
        right_node = self.children[1] if self.children[1].node_type != "Hash" \
            else self.children[1].children[0]
        # self.f_mean = self.plan_rows / left_node.plan_rows / right_node.plan_rows
        # self.f_nodes.add(self)
        # self.f_std = self.UNCERTAINTY * self.f_mean
        # prod_std_mean_square = 1
        # prod_mean_square = 1
        # # Derive the standard deviation of the join cardinality
        # for f_node in self.f_nodes.union(self.u_nodes):
        #     prod_std_mean_square = prod_std_mean_square * (f_node.f_std ** 2 + f_node.f_mean ** 2)
        #     prod_mean_square = prod_mean_square * f_node.f_mean ** 2
        # delta_variance = (prod_std_mean_square - prod_mean_square) ** 0.5
        # self.d_std = self.card_product * self.filter_product * delta_variance
        # self.d_mean = self.plan_rows
        self.c_std = self.cost / (right_node.plan_rows + right_node.plan_rows) * \
                     ((right_node.d_std ** 2 + right_node.d_std ** 2) ** 0.5)
        # print("Hash Cond: ", self.children[0].tables, self.children[1].tables, self.hash_cond)
        # if self.f_key == "ci:it-n-pi":
        #     print("Hash Cond: ", self.children[0].tables, self.children[1].tables, self.hash_cond)