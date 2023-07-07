import typing as t
import numpy.typing as nt
from dataclasses import dataclass


Rectangle = t.NamedTuple('Rectangle', [('width', float), ('height', float)])
Stage = t.NamedTuple('Stage', [('width', float), ('height', float), ('bottom_x', float), ('bottom_y', float)])
Room = t.NamedTuple('Room', [('width', float), ('height', float)])
Attendee = t.NamedTuple('Attendee', [('x', float), ('y', float), ('tastes', t.Iterable[float])])
Placement = t.NamedTuple('Placement', [('x', float), ('y', float)])


@dataclass
class ProblemInfo:
    room: Room
    stage: Stage
    musicians: t.Iterable[int]
    attendees: t.Iterable[Attendee]


@dataclass
class ProblemSolution:
    placements: t.List[Placement]
