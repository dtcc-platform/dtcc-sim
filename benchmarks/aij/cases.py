"""AIJ benchmark cases A and B: isolated buildings in a wind-tunnel boundary layer.

Coordinates have the origin at the centre of the building footprint, x along
the wind and z up. Both cases are symmetric about y = 0, where all
measurements lie, so the half domain y <= 0 is solved with a slip plane at
y = 0. Domain extents follow the AIJ guidelines (Tominaga et al., 2008):
5H upstream, laterally and above the building, 10H downstream.
"""

from dataclasses import dataclass

from .data import DATA_DIR
from .geometry import Box


@dataclass(frozen=True)
class Case:
    name: str
    title: str
    workbook: str
    length: float
    """Reference size b of the workbook coordinates [m]."""
    width: float
    depth: float
    height: float
    reference: str

    @property
    def path(self):
        return DATA_DIR / self.workbook

    def building(self, offset=0.):
        """Half building for the symmetric domain, grown by ``offset`` on its walls."""
        return Box(-.5 * self.depth - offset, .5 * self.depth + offset, -.5 * self.width - offset, 0., self.height)

    def domain(self, upstream=5., downstream=10., lateral=5., above=5.):
        """Half domain ((x0, x1), (y0, 0), top) with distances in building heights."""
        H = self.height
        return ((-.5 * self.depth - upstream * H, .5 * self.depth + downstream * H),
                (-.5 * self.width - lateral * H, 0.), H + above * H)


CASES = {
    "A": Case(
        "A", "1:1:2 square-section high-rise building", "CaseA(1_1_2).xls",
        length=.08, width=.08, depth=.08, height=.16,
        reference="Meng and Hibi (1998), Turbulent measurements of the flow field around a high-rise building",
    ),
    "B": Case(
        "B", "4:4:1 slab normal to the wind", "CaseB(4_4_1).xls",
        length=.05, width=.2, depth=.05, height=.2,
        reference="Miyazaki and Tominaga (2003), wind tunnel experiment cited by the AIJ benchmark",
    ),
}
