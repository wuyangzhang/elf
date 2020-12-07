from abc import ABC, abstractmethod


class ResourceMonitorBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def estimate_resource(self):
        pass


class ResourceMonitorLatency(ResourceMonitorBase):
    def __init__(self):
        super().__init__()

    def estimate_resource(self):
        """ Resource estimation.
        this function takes as the input the historical e2e latency in order to
        evaluate the computation capability of all the involved nodes.

        :return estimated computing capability of remote servers.
        """
        pass
