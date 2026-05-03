"""
Static traffic assignment on DTCC road networks.

The simulator builds a synthetic DeSO-based OD matrix and solves a standard
Wardrop user-equilibrium assignment with BPR link costs using Frank-Wolfe.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import heapq
import math
import re
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, Field

from dtcc_core.model import RoadNetwork


DRIVABLE_HIGHWAY_TYPES = (
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
)


DEFAULT_SPEED_KMH_BY_HIGHWAY = {
    "motorway": 90.0,
    "motorway_link": 50.0,
    "trunk": 80.0,
    "trunk_link": 50.0,
    "primary": 60.0,
    "primary_link": 45.0,
    "secondary": 50.0,
    "secondary_link": 40.0,
    "tertiary": 45.0,
    "tertiary_link": 35.0,
    "unclassified": 40.0,
    "residential": 30.0,
    "living_street": 10.0,
    "service": 20.0,
}


DEFAULT_LANES_BY_HIGHWAY = {
    "motorway": 2.0,
    "motorway_link": 1.0,
    "trunk": 2.0,
    "trunk_link": 1.0,
    "primary": 2.0,
    "primary_link": 1.0,
    "secondary": 2.0,
    "secondary_link": 1.0,
    "tertiary": 2.0,
    "tertiary_link": 1.0,
    "unclassified": 2.0,
    "residential": 2.0,
    "living_street": 1.0,
    "service": 1.0,
}


CAPACITY_MULTIPLIER_BY_HIGHWAY = {
    "motorway": 1.2,
    "motorway_link": 0.8,
    "trunk": 1.1,
    "trunk_link": 0.8,
    "primary": 1.0,
    "primary_link": 0.75,
    "secondary": 0.9,
    "secondary_link": 0.7,
    "tertiary": 0.8,
    "tertiary_link": 0.65,
    "unclassified": 0.7,
    "residential": 0.55,
    "living_street": 0.2,
    "service": 0.35,
}


class TrafficAssignmentParameters(BaseModel):
    """Parameters for synthetic demand and static traffic assignment."""

    trips_per_employed: float = Field(
        2.0,
        ge=0.0,
        description="Daily vehicle-trip production per employed resident.",
    )
    peak_hour_factor: float = Field(
        0.10,
        ge=0.0,
        le=1.0,
        description="Fraction of daily trips assigned to the simulated peak hour.",
    )
    population_attraction_weight: float = Field(
        0.25,
        ge=0.0,
        description="Attraction weight for DeSO population.",
    )
    employment_attraction_weight: float = Field(
        1.0,
        ge=0.0,
        description="Attraction weight for employed residents.",
    )
    gravity_gamma: float = Field(
        0.08,
        ge=0.0,
        description="Exponential gravity decay per minute of free-flow travel time.",
    )

    alpha: float = Field(0.15, ge=0.0, description="BPR alpha parameter.")
    beta: float = Field(4.0, gt=0.0, description="BPR beta parameter.")
    capacity_per_lane: float = Field(
        1800.0,
        gt=0.0,
        description="Nominal hourly capacity per lane.",
    )
    background_flow_fraction: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Exogenous background traffic as a fraction of each directed arc capacity."
        ),
    )
    default_speed_kmh: float = Field(
        40.0,
        gt=0.0,
        description="Fallback road speed when OSM maxspeed/highway is unavailable.",
    )
    default_lanes: float = Field(
        2.0,
        gt=0.0,
        description="Fallback physical lanes when OSM lanes/highway is unavailable.",
    )

    bidirectional: bool = Field(
        True,
        description="Add reverse directed arcs for non-oneway physical roads.",
    )
    exclude_self_trips: bool = Field(
        True,
        description="Do not create OD demand from a zone to itself.",
    )
    max_iterations: int = Field(
        30,
        ge=1,
        description="Maximum Frank-Wolfe iterations.",
    )
    relative_gap_tolerance: float = Field(
        1.0e-4,
        ge=0.0,
        description="Convergence tolerance for the relative assignment gap.",
    )
    line_search_iterations: int = Field(
        32,
        ge=1,
        description="Bisection iterations for the Frank-Wolfe line search.",
    )

    drivable_highway_types: tuple[str, ...] = Field(
        DRIVABLE_HIGHWAY_TYPES,
        description="OSM highway classes included in the car traffic graph.",
    )


@dataclass
class _DirectedGraph:
    edge_index: np.ndarray
    direction: np.ndarray
    tail: np.ndarray
    head: np.ndarray
    free_flow_time: np.ndarray
    capacity: np.ndarray
    adjacency: list[list[tuple[int, int]]]
    active_vertices: np.ndarray


class TrafficAssignmentSimulator:
    """Synthetic DeSO demand and static traffic assignment for road networks."""

    def __init__(
        self,
        roads: RoadNetwork,
        zones: Any | None = None,
        params: TrafficAssignmentParameters | None = None,
    ):
        self.roads = roads
        self.zones = zones
        self.params = params or TrafficAssignmentParameters()
        self.diagnostics: dict[str, Any] = {}
        self.od_matrix: np.ndarray | None = None
        self.zone_vertices: np.ndarray | None = None
        self.zone_codes: np.ndarray | None = None

    def simulate(self) -> RoadNetwork:
        """Run the traffic assignment and return a RoadNetwork with edge results."""
        graph = self._build_directed_graph(self.roads)
        zone_data = self._zone_data(self.zones)
        zone_vertices = self._connect_zones(zone_data["centroids"], graph)
        demand = self._build_demand_matrix(zone_data, zone_vertices, graph)
        background_arc_flow = self._background_arc_flow(graph)

        self.od_matrix = demand
        self.zone_vertices = zone_vertices
        self.zone_codes = zone_data["codes"]

        if demand.sum() <= 0.0:
            assigned_arc_flow = np.zeros(len(graph.tail), dtype=float)
            iterations = 0
            relative_gap = 0.0
            assigned_demand = 0.0
        else:
            initial_costs = self._bpr_costs(background_arc_flow, graph)
            assigned_arc_flow, assigned_demand = self._all_or_nothing(
                demand,
                zone_vertices,
                initial_costs,
                graph,
            )
            relative_gap = math.inf
            iterations = 0

            for iteration in range(1, self.params.max_iterations + 1):
                costs = self._bpr_costs(assigned_arc_flow + background_arc_flow, graph)
                auxiliary_flow, assigned_demand = self._all_or_nothing(
                    demand,
                    zone_vertices,
                    costs,
                    graph,
                )
                relative_gap = self._relative_gap(
                    assigned_arc_flow,
                    auxiliary_flow,
                    costs,
                )
                iterations = iteration
                if relative_gap <= self.params.relative_gap_tolerance:
                    break

                step = self._line_search(
                    assigned_arc_flow,
                    auxiliary_flow,
                    graph,
                    background_arc_flow=background_arc_flow,
                )
                assigned_arc_flow = assigned_arc_flow + step * (
                    auxiliary_flow - assigned_arc_flow
                )

        arc_flow = assigned_arc_flow + background_arc_flow
        costs = self._bpr_costs(arc_flow, graph)

        result = self._result_roadnetwork(
            arc_flow=arc_flow,
            arc_costs=costs,
            graph=graph,
            assigned_arc_flow=assigned_arc_flow,
            background_arc_flow=background_arc_flow,
        )
        self.diagnostics = {
            "iterations": iterations,
            "relative_gap": float(relative_gap),
            "total_demand": float(demand.sum()),
            "assigned_demand": float(assigned_demand),
            "unassigned_demand": float(max(demand.sum() - assigned_demand, 0.0)),
            "background_link_flow": float(background_arc_flow.sum()),
            "zone_count": int(len(zone_vertices)),
        }
        return result

    def _build_directed_graph(self, roads: RoadNetwork) -> _DirectedGraph:
        vertices = np.asarray(roads.vertices, dtype=float)
        edges = np.asarray(roads.edges, dtype=np.int64).reshape((-1, 2))
        lengths = np.asarray(roads.length, dtype=float).reshape((-1,))
        if len(vertices) == 0 or len(edges) == 0:
            raise ValueError("Traffic assignment requires a non-empty RoadNetwork.")

        tails = []
        heads = []
        edge_index = []
        direction = []
        free_flow_time = []
        capacity = []

        for index, (start, end) in enumerate(edges):
            highway = self._highway(index)
            if not self._is_drivable(highway):
                continue

            length = max(float(lengths[index]), 1.0e-6)
            speed_kmh = self._speed_kmh(index, highway)
            one_way = self._oneway(index)
            lanes = self._lanes(index, highway)
            capacity_multiplier = CAPACITY_MULTIPLIER_BY_HIGHWAY.get(
                highway,
                0.6,
            )

            if one_way or not self.params.bidirectional:
                directional_lanes = max(lanes, 0.25)
            else:
                directional_lanes = max(lanes / 2.0, 0.25)

            arc_capacity = (
                directional_lanes
                * self.params.capacity_per_lane
                * capacity_multiplier
            )
            t0 = length / (speed_kmh / 3.6)

            tails.append(int(start))
            heads.append(int(end))
            edge_index.append(index)
            direction.append(1)
            free_flow_time.append(t0)
            capacity.append(arc_capacity)

            if self.params.bidirectional and not one_way and start != end:
                tails.append(int(end))
                heads.append(int(start))
                edge_index.append(index)
                direction.append(-1)
                free_flow_time.append(t0)
                capacity.append(arc_capacity)

        if len(tails) == 0:
            raise ValueError("Traffic assignment found no drivable road arcs.")

        tail_array = np.asarray(tails, dtype=np.int64)
        head_array = np.asarray(heads, dtype=np.int64)
        adjacency = [[] for _ in range(len(vertices))]
        for arc_index, (tail, head) in enumerate(zip(tail_array, head_array)):
            adjacency[int(tail)].append((int(head), int(arc_index)))

        active_vertices = np.unique(np.concatenate((tail_array, head_array)))
        return _DirectedGraph(
            edge_index=np.asarray(edge_index, dtype=np.int64),
            direction=np.asarray(direction, dtype=np.int8),
            tail=tail_array,
            head=head_array,
            free_flow_time=np.asarray(free_flow_time, dtype=float),
            capacity=np.asarray(capacity, dtype=float),
            adjacency=adjacency,
            active_vertices=active_vertices,
        )

    def _zone_data(self, zones: Any | None) -> dict[str, np.ndarray]:
        if zones is None:
            raise ValueError("Traffic assignment requires DeSO zones with statistics.")

        arrays = zones.to_arrays()
        centroids = np.asarray(arrays["centroids"], dtype=float)
        if centroids.ndim != 2 or len(centroids) == 0:
            raise ValueError("Traffic assignment requires zones with centroids.")
        centroids = centroids[:, :2]

        codes = np.asarray(
            arrays.get("codes", np.arange(len(centroids), dtype=int)),
            dtype=object,
        )
        population = self._zone_values(
            zones,
            len(centroids),
            ("population_total", "population"),
            default=1.0,
        )
        employed = self._zone_values(
            zones,
            len(centroids),
            ("employed_residents_total", "employment_total", "employment"),
            default=None,
        )
        if employed is None:
            employed = population.copy()

        attractions = (
            self.params.population_attraction_weight * population
            + self.params.employment_attraction_weight * employed
        )
        productions = (
            employed
            * self.params.trips_per_employed
            * self.params.peak_hour_factor
        )

        return {
            "centroids": centroids,
            "codes": codes,
            "population": np.nan_to_num(population, nan=0.0, posinf=0.0, neginf=0.0),
            "employed": np.nan_to_num(employed, nan=0.0, posinf=0.0, neginf=0.0),
            "attractions": np.maximum(
                np.nan_to_num(attractions, nan=0.0, posinf=0.0, neginf=0.0),
                0.0,
            ),
            "productions": np.maximum(
                np.nan_to_num(productions, nan=0.0, posinf=0.0, neginf=0.0),
                0.0,
            ),
        }

    def _connect_zones(self, centroids: np.ndarray, graph: _DirectedGraph) -> np.ndarray:
        vertices = np.asarray(self.roads.vertices, dtype=float)[:, :2]
        candidates = graph.active_vertices
        candidate_vertices = vertices[candidates]
        zone_vertices = []
        for centroid in centroids:
            distances = np.sum((candidate_vertices - centroid[:2]) ** 2, axis=1)
            zone_vertices.append(int(candidates[int(np.argmin(distances))]))
        return np.asarray(zone_vertices, dtype=np.int64)

    def _build_demand_matrix(
        self,
        zone_data: dict[str, np.ndarray],
        zone_vertices: np.ndarray,
        graph: _DirectedGraph,
    ) -> np.ndarray:
        n_zones = len(zone_vertices)
        demand = np.zeros((n_zones, n_zones), dtype=float)
        attractions = zone_data["attractions"]
        productions = zone_data["productions"]

        for origin_index, origin_vertex in enumerate(zone_vertices):
            production = productions[origin_index]
            if production <= 0.0:
                continue
            distances, _ = self._shortest_paths(origin_vertex, graph.free_flow_time, graph)
            weights = np.zeros(n_zones, dtype=float)
            for destination_index, destination_vertex in enumerate(zone_vertices):
                if self.params.exclude_self_trips and destination_index == origin_index:
                    continue
                if destination_vertex == origin_vertex:
                    continue
                travel_time = distances[destination_vertex]
                if not np.isfinite(travel_time):
                    continue
                minutes = travel_time / 60.0
                weights[destination_index] = (
                    attractions[destination_index]
                    * math.exp(-self.params.gravity_gamma * minutes)
                )

            total_weight = weights.sum()
            if total_weight > 0.0:
                demand[origin_index, :] = production * weights / total_weight

        return demand

    def _all_or_nothing(
        self,
        demand: np.ndarray,
        zone_vertices: np.ndarray,
        costs: np.ndarray,
        graph: _DirectedGraph,
    ) -> tuple[np.ndarray, float]:
        arc_flow = np.zeros(len(graph.tail), dtype=float)
        assigned_demand = 0.0

        for origin_index, origin_vertex in enumerate(zone_vertices):
            destinations = np.flatnonzero(demand[origin_index] > 0.0)
            if len(destinations) == 0:
                continue

            _, predecessor_arc = self._shortest_paths(origin_vertex, costs, graph)
            for destination_index in destinations:
                destination_vertex = int(zone_vertices[destination_index])
                flow = float(demand[origin_index, destination_index])
                if destination_vertex == origin_vertex:
                    continue

                path_arcs = []
                vertex = destination_vertex
                while vertex != origin_vertex:
                    arc = int(predecessor_arc[vertex])
                    if arc < 0:
                        path_arcs = []
                        break
                    path_arcs.append(arc)
                    vertex = int(graph.tail[arc])

                if not path_arcs:
                    continue

                for arc in path_arcs:
                    arc_flow[arc] += flow
                assigned_demand += flow

        return arc_flow, assigned_demand

    def _shortest_paths(
        self,
        origin: int,
        costs: np.ndarray,
        graph: _DirectedGraph,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_vertices = len(graph.adjacency)
        distances = np.full(n_vertices, np.inf, dtype=float)
        predecessor_arc = np.full(n_vertices, -1, dtype=np.int64)
        distances[int(origin)] = 0.0
        queue = [(0.0, int(origin))]

        while queue:
            distance, vertex = heapq.heappop(queue)
            if distance > distances[vertex]:
                continue
            for neighbor, arc_index in graph.adjacency[vertex]:
                new_distance = distance + float(costs[arc_index])
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessor_arc[neighbor] = arc_index
                    heapq.heappush(queue, (new_distance, neighbor))

        return distances, predecessor_arc

    def _bpr_costs(self, arc_flow: np.ndarray, graph: _DirectedGraph) -> np.ndarray:
        ratio = np.divide(
            arc_flow,
            graph.capacity,
            out=np.zeros_like(arc_flow, dtype=float),
            where=graph.capacity > 0.0,
        )
        return graph.free_flow_time * (
            1.0 + self.params.alpha * np.power(ratio, self.params.beta)
        )

    def _background_arc_flow(self, graph: _DirectedGraph) -> np.ndarray:
        if self.params.background_flow_fraction <= 0.0:
            return np.zeros(len(graph.tail), dtype=float)
        return graph.capacity * self.params.background_flow_fraction

    def _relative_gap(
        self,
        flow: np.ndarray,
        auxiliary_flow: np.ndarray,
        costs: np.ndarray,
    ) -> float:
        denominator = float(np.dot(costs, flow))
        if denominator <= 1.0e-12:
            return math.inf
        gap = float(np.dot(costs, flow - auxiliary_flow))
        return max(gap, 0.0) / denominator

    def _line_search(
        self,
        flow: np.ndarray,
        auxiliary_flow: np.ndarray,
        graph: _DirectedGraph,
        background_arc_flow: np.ndarray | None = None,
    ) -> float:
        direction = auxiliary_flow - flow
        if np.allclose(direction, 0.0):
            return 0.0

        background = (
            background_arc_flow
            if background_arc_flow is not None
            else np.zeros_like(flow, dtype=float)
        )

        def derivative(step: float) -> float:
            candidate = flow + step * direction + background
            return float(np.dot(self._bpr_costs(candidate, graph), direction))

        left = 0.0
        right = 1.0
        if derivative(left) >= 0.0:
            return 0.0
        if derivative(right) <= 0.0:
            return 1.0

        for _ in range(self.params.line_search_iterations):
            midpoint = 0.5 * (left + right)
            if derivative(midpoint) <= 0.0:
                left = midpoint
            else:
                right = midpoint
        return 0.5 * (left + right)

    def _result_roadnetwork(
        self,
        arc_flow: np.ndarray,
        arc_costs: np.ndarray,
        graph: _DirectedGraph,
        assigned_arc_flow: np.ndarray | None = None,
        background_arc_flow: np.ndarray | None = None,
    ) -> RoadNetwork:
        result = deepcopy(self.roads)
        n_edges = len(np.asarray(self.roads.edges).reshape((-1, 2)))

        flow_forward = np.zeros(n_edges, dtype=float)
        flow_reverse = np.zeros(n_edges, dtype=float)
        assigned_flow = np.zeros(n_edges, dtype=float)
        background_flow = np.zeros(n_edges, dtype=float)
        capacity_forward = np.zeros(n_edges, dtype=float)
        capacity_reverse = np.zeros(n_edges, dtype=float)
        free_flow_time = np.zeros(n_edges, dtype=float)
        travel_time = np.zeros(n_edges, dtype=float)
        volume_capacity_ratio = np.zeros(n_edges, dtype=float)
        traffic_excluded = np.ones(n_edges, dtype=bool)

        for arc_index, edge_index in enumerate(graph.edge_index):
            edge_index = int(edge_index)
            traffic_excluded[edge_index] = False
            free_flow_time[edge_index] = max(
                free_flow_time[edge_index],
                float(graph.free_flow_time[arc_index]),
            )
            travel_time[edge_index] = max(
                travel_time[edge_index],
                float(arc_costs[arc_index]),
            )
            ratio = (
                float(arc_flow[arc_index]) / float(graph.capacity[arc_index])
                if graph.capacity[arc_index] > 0.0
                else 0.0
            )
            volume_capacity_ratio[edge_index] = max(
                volume_capacity_ratio[edge_index],
                ratio,
            )
            if graph.direction[arc_index] > 0:
                flow_forward[edge_index] += arc_flow[arc_index]
                capacity_forward[edge_index] += graph.capacity[arc_index]
            else:
                flow_reverse[edge_index] += arc_flow[arc_index]
                capacity_reverse[edge_index] += graph.capacity[arc_index]

            if assigned_arc_flow is not None:
                assigned_flow[edge_index] += assigned_arc_flow[arc_index]
            if background_arc_flow is not None:
                background_flow[edge_index] += background_arc_flow[arc_index]

        lengths = np.asarray(self.roads.length, dtype=float).reshape((-1,))
        missing_time = travel_time <= 0.0
        if np.any(missing_time):
            fallback_speed = self.params.default_speed_kmh / 3.6
            travel_time[missing_time] = lengths[missing_time] / fallback_speed
            free_flow_time[missing_time] = travel_time[missing_time]

        speed = np.divide(
            lengths,
            travel_time,
            out=np.zeros_like(lengths, dtype=float),
            where=travel_time > 0.0,
        ) * 3.6

        result.attributes["flow"] = (flow_forward + flow_reverse).tolist()
        result.attributes["flow_forward"] = flow_forward.tolist()
        result.attributes["flow_reverse"] = flow_reverse.tolist()
        result.attributes["flow_assigned"] = assigned_flow.tolist()
        result.attributes["flow_background"] = background_flow.tolist()
        result.attributes["capacity"] = (capacity_forward + capacity_reverse).tolist()
        result.attributes["capacity_forward"] = capacity_forward.tolist()
        result.attributes["capacity_reverse"] = capacity_reverse.tolist()
        result.attributes["free_flow_time"] = free_flow_time.tolist()
        result.attributes["travel_time"] = travel_time.tolist()
        result.attributes["speed"] = speed.tolist()
        result.attributes["volume_capacity_ratio"] = volume_capacity_ratio.tolist()
        result.attributes["traffic_excluded"] = traffic_excluded.tolist()
        return result

    def _zone_values(
        self,
        zones: Any,
        size: int,
        names: Sequence[str],
        default: float | None,
    ) -> np.ndarray | None:
        fields = getattr(zones, "fields", {}) or {}
        for name in names:
            field = fields.get(name)
            if field is None:
                continue
            values = np.asarray(field.values, dtype=float).reshape((-1,))
            if len(values) == size:
                return values

        arrays = zones.to_arrays()
        array_fields = arrays.get("fields", {})
        for name in names:
            if name in array_fields:
                values = np.asarray(array_fields[name], dtype=float).reshape((-1,))
                if len(values) == size:
                    return values

        if default is None:
            return None
        return np.full(size, float(default), dtype=float)

    def _highway(self, edge_index: int) -> str:
        value = self._edge_attribute("highway", edge_index, "")
        if isinstance(value, (list, tuple, np.ndarray)):
            value = value[0] if len(value) > 0 else ""
        return str(value).strip().lower()

    def _is_drivable(self, highway: str) -> bool:
        if not highway:
            return True
        return highway in set(self.params.drivable_highway_types)

    def _oneway(self, edge_index: int) -> bool:
        value = self._edge_attribute("oneway", edge_index, False)
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        normalized = str(value).strip().lower()
        return normalized in {"true", "yes", "1", "-1"}

    def _speed_kmh(self, edge_index: int, highway: str) -> float:
        speed = self._parse_number(self._edge_attribute("maxspeed_kmh", edge_index))
        if speed is None:
            speed = self._parse_number(self._edge_attribute("maxspeed", edge_index))
        if speed is None:
            speed = DEFAULT_SPEED_KMH_BY_HIGHWAY.get(
                highway,
                self.params.default_speed_kmh,
            )
        return max(float(speed), 1.0)

    def _lanes(self, edge_index: int, highway: str) -> float:
        lanes = self._parse_number(self._edge_attribute("lanes", edge_index))
        if lanes is None:
            lanes = DEFAULT_LANES_BY_HIGHWAY.get(highway, self.params.default_lanes)
        return max(float(lanes), 0.25)

    def _edge_attribute(self, name: str, edge_index: int, default=None):
        values = self.roads.attributes.get(name)
        if values is None:
            return default
        if isinstance(values, (str, bytes)):
            return values
        try:
            if len(values) == len(self.roads.edges):
                return values[edge_index]
        except TypeError:
            return values
        return default

    @staticmethod
    def _parse_number(value) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float, np.number)):
            if np.isfinite(value):
                return float(value)
            return None
        text = str(value).strip().replace(",", ".")
        if not text:
            return None
        match = re.search(r"[-+]?\d*\.?\d+", text)
        if match is None:
            return None
        try:
            return float(match.group())
        except ValueError:
            return None


__all__ = ["TrafficAssignmentParameters", "TrafficAssignmentSimulator"]
