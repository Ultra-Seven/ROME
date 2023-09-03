import json
import os
import re
import sys

from operators.aggregate import Aggregate
from operators.bitmap_heap_scan import BitmapHeapScan
from operators.bitmap_index_scan import BitmapIndexScan
from operators.gather import Gather
from operators.hash import Hash
from operators.hash_join import HashJoin
from operators.index_only_scan import IndexOnlyScan
from operators.index_scan import IndexScan
from operators.limit import Limit
from operators.materialize import Materialize
from operators.merge_join import MergeJoin
from operators.nested_loop import NestedLoop
from operators.seq_scan import SeqScan
import igraph as ig
from igraph import Graph, EdgeSeq, summary, drawing
# import plotly.graph_objects as go
# import matplotlib.pyplot as plt

from operators.sort import Sort
from visitors.factor_visitor import FactorVisitor
from visitors.intermediate_visitor import IntermediateVisitor
from visitors.predicate_visitor import PredicateVisitor


class PlanTree:
    def __init__(self, sql, result, postgres,
                 visualization=True, pid=-1, query_name="default"):
        self.decoded_plan = result.stdout.decode('utf-8')
        # print(self.decoded_plan)
        query_plan = json.loads(self.decoded_plan)
        self.plan = query_plan[0]
        self.postgres = postgres
        # Generate intermediate tables
        self.nr_vertices = 0
        self.join_keys = {}
        self.f_visitor = FactorVisitor(postgres)
        self.d_visitor = IntermediateVisitor(postgres)
        self.p_visitor = PredicateVisitor(postgres, sql)
        self.root = self.generate_intermediate_tables(query_plan[0]["Plan"])
        self.pid = pid
        self.query_name = query_name
        # Visualize the plan tree
        if visualization:
            self.visualize()

    def generate_intermediate_tables(self, node):
        node_alias = set()
        if "Alias" in node:
            node_alias.add(node["Alias"])
        child_nodes = []
        if "Plans" in node:
            for plan_node in node["Plans"]:
                child_node = self.generate_intermediate_tables(plan_node)
                for child_alias in plan_node["tables"]:
                    node_alias.add(child_alias)
                child_nodes.append(child_node)

        node["tables"] = node_alias
        plan_node = self.construct_node(node, child_nodes)
        if plan_node is None:
            return child_nodes[0]
        # print(plan_node, plan_node.f_key, plan_node.cost)
        # Generate coefficients
        # plan_node.accept(self.f_visitor)
        # plan_node.cost_variables = {k: "{:.2e}".format(v)
        #                             for k, v in self.f_visitor.factor_variables[-1].copy().items()}
        plan_node.accept(self.d_visitor)
        plan_node.accept(self.p_visitor)
        self.nr_vertices += 1
        return plan_node

    def construct_node(self, node, child_nodes):
        node_type = node["Node Type"]
        plan_node = None
        if node_type == "Seq Scan":
            plan_node = SeqScan(node, self.postgres, child_nodes)
        elif node_type == "Index Scan":
            plan_node = IndexScan(node, self.postgres, child_nodes)
        elif node_type == "Bitmap Heap Scan":
            plan_node = BitmapHeapScan(node, self.postgres, child_nodes)
        elif node_type == "Bitmap Index Scan":
            plan_node = BitmapIndexScan(node, self.postgres, child_nodes)
        elif node_type == "Nested Loop":
            plan_node = NestedLoop(node, self.postgres, child_nodes)
        elif node_type == "Merge Join":
            plan_node = MergeJoin(node, self.postgres, child_nodes)
        elif node_type == "Hash Join":
            plan_node = HashJoin(node, self.postgres, child_nodes)
        elif node_type == "Aggregate":
            plan_node = Aggregate(node, self.postgres, child_nodes)
        elif node_type == "Sort":
            plan_node = Sort(node, self.postgres, child_nodes)
        elif node_type == "Limit":
            plan_node = Limit(node, self.postgres, child_nodes)
        elif node_type == "Result":
            raise Exception("Unknown node type: " + node_type)
        elif node_type == "Append":
            raise Exception("Unknown node type: " + node_type)
        elif node_type == "Materialize":
            plan_node = Materialize(node, self.postgres, child_nodes)
        elif node_type == "Unique":
            # raise Exception("Unknown node type: " + node_type)
            return None
        elif node_type == "Gather" or node_type == "Gather Merge":
            plan_node = Gather(node, self.postgres, child_nodes)
        # elif node_type == "Gather Merge":
        #     raise Exception("Unknown node type: " + node_type)
        elif node_type == "Hash":
            plan_node = Hash(node, self.postgres, child_nodes)
        elif node_type == "Index Only Scan":
            plan_node = IndexOnlyScan(node, self.postgres, child_nodes)
        else:
            if float(node["Total Cost"]) < 10:
                return None
            else:
                raise Exception("Unknown node type: " + node_type)
        return plan_node

    def visualize(self):
        # g = ig.Graph(n=self.nr_vertices, directed=True)
        g = Graph.Tree(self.nr_vertices, 2)
        g.delete_edges(g.es)
        self.add_edges(self.root, g, 0, [self.root])
        layout = g.layout_reingold_tilford(mode="in", root=[0])
        visual_style = {}
        # Set bbox and margin
        visual_style["bbox"] = (1000, 900)
        visual_style["margin"] = 60
        # Set vertex colours
        visual_style["vertex_color"] = 'white'
        # Set vertex size
        # visual_style["vertex_size"] = 80
        # Set vertex lable size
        visual_style["vertex_label_size"] = 5
        # Don't curve the edges
        visual_style["edge_curved"] = False
        # Set the layout
        visual_style["layout"] = layout
        if not os.path.exists(f'./figs/{self.query_name}'):
            os.mkdir(f'./figs/{self.query_name}')
        ig.plot(g, target=f'./figs/{self.query_name}/{self.pid}.pdf', **visual_style)

    def add_edges(self, node, g, nid, added_vertices):
        c = "{:.2e}".format(node.cost)
        g.vs[nid]["label"] = f"{node}"
        g.vs[nid]["shape"] = "rectangle"
        g.vs[nid]["height"] = 40
        nr_l = 15
        g.vs[nid]["width"] = nr_l * 3

        # g.vs[nid]["bbox"] = (100, 60)
        if len(node.children) > 0:
            for idx, child in enumerate(node.children):
                child_next_id = len(added_vertices)
                added_vertices.append(child)
                g.add_edges([(nid, child_next_id)])
                self.add_edges(child, g, child_next_id, added_vertices)
