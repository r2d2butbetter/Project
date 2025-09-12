# Temporal graph data structure
from dataclasses import dataclass

@dataclass(frozen=True) #frozen makes it immutable, and so hashable
class Temporal_edge:
    """
    Each edge has 4 attributes:
        (u, v, t, lambda) are src, dst, departure time, duration of journey
        id is for convinence
    """

    id: int
    u: int
    v: int
    departure: int
    duration: int

    @property
    def get_arrival_time(self):
        """Basically this is t+lambda because thats when u reach"""
        return self.departure + self.duration
    
    def __repr__(self):
        return f"e{self.id}({self.u} -> {self.v} @ {self.departure}, arrival: {self.get_arrival_time})"

