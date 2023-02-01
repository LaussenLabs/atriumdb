from abc import ABC, abstractmethod


class SQLHandler(ABC):
    @abstractmethod
    def create_schema(self):
        pass

    @abstractmethod
    def insert_measure(self, measure_tag: str, freq_nhz: int, units: str = None, measure_name: str = None):
        pass

    @abstractmethod
    def select_measure(self, measure_id: int = None, measure_tag: str = None, freq_nhz: int = None, units: str = None):
        pass

    # @abstractmethod
    # def insert_device(self, device_tag: str, device_name: str = None):
    #     pass
