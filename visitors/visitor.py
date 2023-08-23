from abc import ABC, abstractmethod

class Visitor(ABC):

    @abstractmethod
    def visit_hash_join_node(self, element) -> None:
        pass

    @abstractmethod
    def visit_merge_join_node(self, element) -> None:
        pass

    @abstractmethod
    def visit_nested_loop_node(self, element) -> None:
        pass

    @abstractmethod
    def visit_hash_node(self, element) -> None:
        pass

    @abstractmethod
    def visit_aggregate_node(self, element) -> None:
        pass

    @abstractmethod
    def visit_sort_node(self, element) -> None:
        pass

    @abstractmethod
    def visit_gather_node(self, element) -> None:
        pass

    @abstractmethod
    def visit_materialize_node(self, element) -> None:
        pass

    @abstractmethod
    def visit_seq_scan_node(self, element) -> None:
        pass

    @abstractmethod
    def visit_index_scan_node(self, element) -> None:
        pass

    @abstractmethod
    def visit_bitmap_index_scan_node(self, element) -> None:
        pass

    @abstractmethod
    def visit_bitmap_heap_scan_node(self, element) -> None:
        pass

    @abstractmethod
    def visit_index_only_scan_node(self, element) -> None:
        pass
