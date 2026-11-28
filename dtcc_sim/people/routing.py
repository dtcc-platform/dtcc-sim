"""Routing and time assignment on Core road networks."""

from dataclasses import dataclass
from heapq import heappop, heappush
from math import hypot

import numpy as np
from dtcc_core.builder.roadnetwork import road_network_mode_mask
from dtcc_core.model.object.people import TimedPath
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Route:
    """Connected metric coordinates and aligned edge IDs."""

    xyz: np.ndarray
    edges: tuple[str, ...]
    signals: frozenset[int]


class Router:
    """Shortest paths with optional costs on mapped pedestrian links."""

    def __init__(self, network, mode="walk", closed_edge=None, distance_only=False):
        self.xyz = np.zeros((len(network.vertices), 3))
        self.xyz[:, :2] = network.vertices[:, :2]
        self.xy = list(map(tuple, self.xyz[:, :2].tolist()))
        self.graph = [[] for _ in self.xyz]
        self.signals = set()
        self.edges = {}
        n = len(network.edges)
        tags = network.attributes
        way = tags.get("osm_way_id", [None] * n)
        segment = tags.get("segment_index", range(n))
        highway = tags.get("highway", [None] * n)
        sidewalk = tags.get("sidewalk", [None] * n)
        direction = tags.get("oneway", [False] * n)
        start_kind = tags.get("start_node_highway", [None] * n)
        end_kind = tags.get("end_node_highway", [None] * n)
        closed_edge_found = closed_edge is None
        for i in np.flatnonzero(road_network_mode_mask(network, mode)):
            a, b = map(int, network.edges[i])
            edge = str(i) if way[i] is None else f"{way[i]}:{segment[i]}"
            distance = float(np.linalg.norm(self.xyz[a] - self.xyz[b]))
            if not distance:
                continue
            if edge == closed_edge:
                closed_edge_found = True
                continue
            penalty = 1.0
            if mode == "walk" and not distance_only:
                if highway[i] in {"primary", "secondary", "tertiary"} and sidewalk[
                    i
                ] in {None, "no", "separate"}:
                    penalty = 1.8
                elif highway[i] == "steps":
                    penalty = 1.5
            self.graph[a].append((b, distance * penalty, edge))
            if mode != "drive" or not direction[i]:
                self.graph[b].append((a, distance * penalty, edge))
            self.edges[edge] = (a, b)
            if start_kind[i] == "traffic_signals":
                self.signals.add(a)
            if end_kind[i] == "traffic_signals":
                self.signals.add(b)
        if not closed_edge_found:
            raise ValueError(f"Unknown walk link {closed_edge!r}")
        self.anchors = np.array(
            [i for i, edges in enumerate(self.graph) if edges], dtype=int
        )
        self.tree = cKDTree(self.xyz[self.anchors, :2])
        self._cache = {}

    def snap(self, xy, limit=90):
        """Return the nearest accessible node within the connector limit."""
        distance, index = self.tree.query(xy)
        return int(self.anchors[index]) if distance <= limit else None

    def between(self, start, end):
        """Return an oriented route including inferred access connectors."""
        begin, finish = self.snap(start[:2]), self.snap(end[:2])
        if begin is None or finish is None:
            return None
        key = (begin, finish)
        if key not in self._cache:
            if len(self._cache) >= 4096:
                self._cache.clear()
            self._cache[key] = self._path(*key)
        route = self._cache[key]
        if route is None:
            return None
        nodes, ids = route
        points = self.xyz[list(nodes)]
        signals = {i for i, node in enumerate(nodes) if node in self.signals}
        if np.any(start != points[0]):
            points = np.vstack((start, points))
            ids = ("", *ids)
            signals = {i + 1 for i in signals}
        if np.any(end != points[-1]):
            points = np.vstack((points, end))
            ids = (*ids, "")
        return Route(points, ids, frozenset(signals)) if ids else None

    def _path(self, start, end):
        target_x, target_y = self.xy[end]
        cost = {start: 0.0}
        prior = {}
        queue = [(0.0, 0.0, start)]
        while queue:
            _, travel, node = heappop(queue)
            if node == end:
                nodes, edges = [end], []
                while node != start:
                    node, edge = prior[node]
                    nodes.append(node)
                    edges.append(edge)
                return tuple(reversed(nodes)), tuple(reversed(edges))
            if travel > cost[node]:
                continue
            for neighbor, metres, edge in self.graph[node]:
                next_cost = travel + metres
                if next_cost < cost.get(neighbor, float("inf")):
                    cost[neighbor] = next_cost
                    prior[neighbor] = (node, edge)
                    x, y = self.xy[neighbor]
                    heappush(
                        queue,
                        (
                            next_cost + hypot(x - target_x, y - target_y),
                            next_cost,
                            neighbor,
                        ),
                    )
        return None


def stationary(state, purpose, place_id, xyz, start, end, journey_id=""):
    """Create an activity or waiting interval."""
    return TimedPath(state, purpose, [[start, *xyz], [end, *xyz]], place_id, journey_id)


def timed_route(route, depart, speed, purpose, journey_id, mode="walk", signal_wait=0):
    """Time a route over its vertices, keeping crossing waits separate."""
    stages = []
    start = 0
    elapsed = float(depart)
    samples = [[elapsed, *route.xyz[0]]]
    for i, (a, b) in enumerate(zip(route.xyz, route.xyz[1:]), 1):
        next_time = elapsed + float(np.linalg.norm(b - a)) / speed
        elapsed = max(next_time, float(np.nextafter(elapsed, np.inf)))
        if elapsed > 86400:
            return []
        samples.append([elapsed, *b])
        wait = signal_wait and i in route.signals and i < len(route.edges)
        if wait or i == len(route.edges):
            stages.append(
                TimedPath(
                    mode,
                    purpose,
                    samples,
                    journey_id=journey_id,
                    edge_ids=route.edges[start:i],
                )
            )
        if wait:
            if elapsed + signal_wait > 86400:
                return []
            stages.append(
                stationary(
                    "wait", purpose, "", b, elapsed, elapsed + signal_wait, journey_id
                )
            )
            elapsed += signal_wait
            start = i
            samples = [[elapsed, *b]]
    return stages
