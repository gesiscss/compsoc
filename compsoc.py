"""
compsoc — Computational Social Methods in Python
"""
from __future__ import annotations

import graph_tool.all as gt
import json
import numpy as np
import pandas as pd
import powerlaw as pl
import subprocess
import tempfile
import warnings

from collections import Counter
from collections.abc import Sequence
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.preprocessing import normalize
from typing import Literal, NamedTuple


# Allowed normaizations in bipartite matrix projection
_VALID_NORMS = (None, "partial", "full")

def co_occurrence(
    node_list_u: pd.DataFrame,
    node_list_v: pd.DataFrame,
    edge_list: pd.DataFrame,
    node_u_identifier: str,
    node_v_identifier: str,
    weight: str | None,
    norm: Literal[None, "partial", "full"],
    category_identifier: str | None = None,
    directed: bool = False,
    remove_self_loops: bool = True,
    enrich_node_list_v: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Project a bipartite occurrence matrix to a unipartite co-occurrence matrix.

    Parameters
    ----------
    node_list_u : pd.DataFrame
        Node list for set U.  Must contain a contiguous integer identifier
        column running from 0 to N_u − 1.
    node_list_v : pd.DataFrame
        Node list for set V.  Must contain a contiguous integer identifier
        column running from 0 to N_v − 1.
    edge_list : pd.DataFrame
        Bipartite occurrence (edge) list with one (u, v) pair per row.
    node_u_identifier : str
        Column in *node_list_u* (and *edge_list*) that holds the U node id.
    node_v_identifier : str
        Column in *node_list_v* (and *edge_list*) that holds the V node id.
    weight : str or None
        Column in *edge_list* that holds edge weights.  ``None`` treats all
        weights as 1.
    norm : {None, 'partial', 'full'}
        Row-normalisation strategy applied to the bipartite matrix B before
        projection:

        * ``None``      — G = B^T · B  (raw co-occurrence counts)
        * ``'partial'`` — G = B^T · B^N  (partially normalised; Leydesdorff
          & Opthof 2010; Batagelj & Cerinšek 2013)
        * ``'full'``    — G = (B^N)^T · B^N  (fully normalised / Salton
          cosine similarity)

    category_identifier : str or None, optional
        Column in *edge_list* that partitions occurrences into disjoint
        categories.  When given, co-occurrences are computed per category
        and stacked into a single edge list.  ``enrich_node_list_v`` is
        ignored in this case.
    directed : bool, optional
        Whether the co-occurrence network is directed.  Default ``False``.
    remove_self_loops : bool, optional
        Drop diagonal entries (self-co-occurrences).  Default ``True``.
    enrich_node_list_v : bool, optional
        When ``True``, return a copy of *node_list_v* enriched with
        projection-derived node attributes (see *Returns*).  Default
        ``False``.

    Returns
    -------
    edge_list_v : pd.DataFrame
        Co-occurrence edge list with columns
        ``['{node_v_identifier}_i', '{node_v_identifier}_j', 'weight',
        'cumfrac']`` — plus *category_identifier* when supplied.
        ``cumfrac`` is the cumulative fraction of total matrix weight
        accounted for by edges at least as strong as each row's weight,
        sorted from strongest to weakest.
    node_list_v_enriched : pd.DataFrame or None
        A copy of *node_list_v* with added attribute columns, or ``None``
        when ``enrich_node_list_v=False`` or *category_identifier* is set.

        Columns added per *norm*:

        * ``None``      — ``occurrence``
        * ``'partial'`` — ``occurrence``, ``self_occurrence``,
          ``self_sufficiency``, ``embeddedness``
        * ``'full'``    — ``self_occurrence``

    Raises
    ------
    ValueError
        If *norm* is not one of ``(None, 'partial', 'full')``.
    ValueError
        If the identifier column of either node list is not a contiguous
        0 … N−1 integer range after sorting.

    Notes
    -----
    Follows the matrix formalism of Batagelj & Cerinšek (2013).
    """
    if norm not in _VALID_NORMS:
        raise ValueError(f"norm must be one of {_VALID_NORMS}; got {norm!r}")

    # ------------------------------------------------------------------ #
    # Category path: recurse once per category, then stack results
    # ------------------------------------------------------------------ #
    if category_identifier is not None:
        if enrich_node_list_v:
            warnings.warn(
                "enrich_node_list_v is ignored when category_identifier is set.",
                UserWarning,
                stacklevel=2,
            )
        categories = (
            edge_list[category_identifier].drop_duplicates().sort_values().tolist()
        )
        parts: list[pd.DataFrame] = []
        for category in categories:
            cat_edges, _ = co_occurrence(
                node_list_u=node_list_u,
                node_list_v=node_list_v,
                edge_list=edge_list[edge_list[category_identifier] == category],
                node_u_identifier=node_u_identifier,
                node_v_identifier=node_v_identifier,
                weight=weight,
                norm=norm,
                category_identifier=None,
                directed=directed,
                remove_self_loops=remove_self_loops,
                enrich_node_list_v=False,
            )
            if cat_edges.empty:
                continue
            cat_edges = cat_edges.copy()
            cat_edges[category_identifier] = category
            parts.append(
                cat_edges[
                    [
                        node_v_identifier + "_i",
                        node_v_identifier + "_j",
                        category_identifier,
                        "weight",
                        "cumfrac",
                    ]
                ]
            )
        if parts:
            edge_list_v = pd.concat(parts, ignore_index=True)
        else:
            edge_list_v = pd.DataFrame(
                columns=[
                    node_v_identifier + "_i",
                    node_v_identifier + "_j",
                    category_identifier,
                    "weight",
                    "cumfrac",
                ]
            )
        return edge_list_v, None

    # ------------------------------------------------------------------ #
    # Validate node lists
    # ------------------------------------------------------------------ #
    node_list_u = (
        node_list_u.sort_values(node_u_identifier).reset_index(drop=True)
    )
    node_list_v = (
        node_list_v.sort_values(node_v_identifier).reset_index(drop=True)
    )
    if not (node_list_u[node_u_identifier] == node_list_u.index).all():
        raise ValueError(
            f"node_list_u['{node_u_identifier}'] must be a contiguous "
            f"0…{len(node_list_u) - 1} integer range."
        )
    if not (node_list_v[node_v_identifier] == node_list_v.index).all():
        raise ValueError(
            f"node_list_v['{node_v_identifier}'] must be a contiguous "
            f"0…{len(node_list_v) - 1} integer range."
        )

    # ------------------------------------------------------------------ #
    # Build sparse bipartite matrix B  (m × n)
    # ------------------------------------------------------------------ #
    rows = edge_list[node_u_identifier].to_numpy()
    cols = edge_list[node_v_identifier].to_numpy()
    w = (
        edge_list[weight].to_numpy()
        if weight is not None
        else np.ones(len(edge_list))
    )
    B = coo_matrix(
        (w, (rows, cols)), shape=(len(node_list_u), len(node_list_v))
    ).tocsr()

    # ------------------------------------------------------------------ #
    # Project to co-occurrence matrix G and collect enrichment vectors
    # ------------------------------------------------------------------ #
    enrichment: dict[str, np.ndarray] = {}

    if norm is None:
        G = (B.T @ B).tocsr()
        if enrich_node_list_v:
            enrichment["occurrence"] = G.diagonal()

    elif norm == "partial":
        BN = normalize(B, norm="l1", axis=1)
        G = (B.T @ BN).tocsr()
        if enrich_node_list_v:
            occurrence = np.asarray(G.sum(axis=0)).ravel()
            self_occurrence = G.diagonal()
            self_sufficiency = self_occurrence / occurrence
            enrichment = {
                "occurrence": occurrence,
                "self_occurrence": self_occurrence,
                "self_sufficiency": self_sufficiency,
                "embeddedness": 1.0 - self_sufficiency,
            }

    else:  # norm == "full"
        BN = normalize(B, norm="l1", axis=1)
        G = (BN.T @ BN).tocsr()
        if enrich_node_list_v:
            enrichment["self_occurrence"] = np.asarray(G.sum(axis=0)).ravel()

    # ------------------------------------------------------------------ #
    # Build co-occurrence edge list
    # ------------------------------------------------------------------ #
    G_coo = G.tocoo()
    col_i = node_v_identifier + "_i"
    col_j = node_v_identifier + "_j"
    edge_list_v = pd.DataFrame(
        {col_i: G_coo.row, col_j: G_coo.col, "weight": G_coo.data}
    )

    if not directed:
        edge_list_v = edge_list_v[edge_list_v[col_j] >= edge_list_v[col_i]]
    if remove_self_loops:
        edge_list_v = edge_list_v[edge_list_v[col_j] != edge_list_v[col_i]]

    edge_list_v = edge_list_v.reset_index(drop=True)

    # Cumulative fraction of total matrix weight, sorted strongest-first
    weight_totals = (
        edge_list_v.groupby("weight")["weight"].sum().sort_index(ascending=False)
    )
    cumfrac = (
        (weight_totals.cumsum() / weight_totals.sum()).round(6).rename("cumfrac")
    )
    edge_list_v = edge_list_v.merge(cumfrac, left_on="weight", right_index=True)

    # ------------------------------------------------------------------ #
    # Enriched node list — returned as a new DataFrame
    # ------------------------------------------------------------------ #
    node_list_v_enriched: pd.DataFrame | None = None
    if enrich_node_list_v and enrichment:
        node_list_v_enriched = node_list_v.copy()
        for col, values in enrichment.items():
            node_list_v_enriched[col] = values

    return edge_list_v, node_list_v_enriched


# graph-tool property-map types that expose a fast NumPy array interface (.a).
# All other types (string, vector, object) require a per-vertex Python loop.
_GT_SCALAR_TYPES: frozenset[str] = frozenset({
    "bool",
    "uint8_t", "int16_t", "short", "int32_t", "int",
    "int64_t", "long", "long long",
    "double", "float", "long double",
})


def construct_graph(
    node_list: pd.DataFrame,
    node_identifier: str,
    edge_list: pd.DataFrame,
    directed: bool = True,
    graph_name: str | None = None,
    node_properties: dict[str, str] | None = None,
    edge_properties: dict[str, str] | None = None,
) -> gt.Graph:
    """Construct a graph-tool graph from pandas node and edge lists.

    Parameters
    ----------
    node_list : pd.DataFrame
        Node list with identifiers and optional attributes.  The identifier
        column must contain a contiguous integer range 0 … N−1 after sorting.
    node_identifier : str
        Column in *node_list* that holds the node id.
    edge_list : pd.DataFrame
        Edge list whose first two columns are the source and target node ids.
        Additional columns are used as edge attributes when *edge_properties*
        is given.
    directed : bool, optional
        Whether the graph is directed.  Default ``True``.
    graph_name : str or None, optional
        Value stored in the internal graph property ``g.gp['name']``.
        Default ``None`` (property not created).
    node_properties : dict[str, str] or None, optional
        Mapping of ``{column_name: graph_tool_type}`` for vertex attributes
        to internalise.  Default ``None``.

        Scalar types (support fast NumPy write via ``.a``):
        ``'bool'``, ``'int'``, ``'int32_t'``, ``'int64_t'``, ``'long'``,
        ``'long long'``, ``'short'``, ``'int16_t'``, ``'uint8_t'``,
        ``'float'``, ``'double'``, ``'long double'``.

        Non-scalar types (written per-vertex): ``'string'``, ``'vector<...>'``,
        ``'python::object'``, etc.
    edge_properties : dict[str, str] or None, optional
        Mapping of ``{column_name: graph_tool_type}`` for edge attributes to
        internalise.  Default ``None``.

    Returns
    -------
    gt.Graph
        graph-tool Graph with all requested property maps populated.

    Raises
    ------
    ValueError
        If the node identifier column is not a contiguous 0 … N−1 integer
        range.
    ValueError
        If *node_properties* and *edge_properties* reference columns not
        present in the respective DataFrames.

    Notes
    -----
    For valid graph-tool type strings see
    https://graph-tool.skewed.de/static/doc/quickstart.html#property-maps
    """
    # ------------------------------------------------------------------ #
    # Validate node list
    # ------------------------------------------------------------------ #
    node_list = node_list.sort_values(node_identifier).reset_index(drop=True)
    if not (node_list[node_identifier] == node_list.index).all():
        raise ValueError(
            f"node_list['{node_identifier}'] must be a contiguous "
            f"0…{len(node_list) - 1} integer range."
        )

    # ------------------------------------------------------------------ #
    # Validate property dicts
    # ------------------------------------------------------------------ #
    if node_properties is not None:
        missing = set(node_properties) - set(node_list.columns)
        if missing:
            raise ValueError(
                f"node_properties references columns not in node_list: {missing}"
            )
    if edge_properties is not None:
        missing = set(edge_properties) - set(edge_list.columns)
        if missing:
            raise ValueError(
                f"edge_properties references columns not in edge_list: {missing}"
            )

    # ------------------------------------------------------------------ #
    # Create graph
    # ------------------------------------------------------------------ #
    g = gt.Graph(directed=directed)

    if graph_name is not None:
        g.gp["name"] = g.new_gp("string")
        g.gp["name"] = graph_name

    g.add_vertex(len(node_list))

    # ------------------------------------------------------------------ #
    # Vertex properties
    # ------------------------------------------------------------------ #
    if node_properties is not None:
        for prop, ptype in node_properties.items():
            g.vp[prop] = g.new_vp(ptype)

        scalar_props = [p for p, t in node_properties.items() if t in _GT_SCALAR_TYPES]
        nonscalar_props = [p for p in node_properties if p not in scalar_props]

        # Fast vectorised write for numeric types
        for prop in scalar_props:
            g.vp[prop].a = node_list[prop].to_numpy()

        # Per-vertex write for strings, vectors, and other non-scalar types
        for prop in nonscalar_props:
            values = node_list[prop].to_numpy()
            pm = g.vp[prop]
            for v, val in enumerate(values):
                pm[v] = val

    # ------------------------------------------------------------------ #
    # Edges (and edge properties)
    # ------------------------------------------------------------------ #
    edge_cols = list(edge_list.columns[:2])

    if edge_properties is not None:
        for prop, ptype in edge_properties.items():
            g.ep[prop] = g.new_ep(ptype)
        edge_cols.extend(edge_properties.keys())
        eprops = [g.ep[prop] for prop in edge_properties]
        g.add_edge_list(edge_list[edge_cols].to_numpy(), eprops=eprops)
    else:
        g.add_edge_list(edge_list[edge_cols].to_numpy())

    return g


class PercolationResult(NamedTuple):
    """Return value of :func:`percolation`."""

    S_1: int    # size of largest component (N when only one component)
    S_2: int    # size of second-largest component (0 when only one component)
    P: float    # percolation probability = S_1 / N
    chi: float  # susceptibility (nan when graph has a single component)


def percolation(g: gt.Graph) -> PercolationResult:
    """Compute percolation observables from a graph's component structure.

    Parameters
    ----------
    g : gt.Graph
        An undirected graph-tool graph.

    Returns
    -------
    PercolationResult
        Named tuple with four fields:

        ``S_1`` — size of the largest component.  Equals N when the graph
        consists of exactly one component.

        ``S_2`` — size of the second-largest component.  Returns ``0`` when
        the graph consists of exactly one component.

        ``P`` — percolation probability S_1 / N, where S_1 is the largest
        component size and N the total number of vertices.

        ``chi`` — susceptibility χ = (Σ_s s² · n_s − S_1²) / N, where n_s
        is the number of components of size s.  This is the size-weighted
        expected component size of a randomly chosen node, with the single
        largest component excluded.  Returns ``nan`` when the graph consists
        of exactly one component (susceptibility is undefined).

    Raises
    ------
    ValueError
        If *g* is directed.

    Notes
    -----
    The susceptibility formula subtracts S_1² from the full weighted sum
    rather than excluding the largest *size class*.  This correctly handles
    the rare case where two components share the maximum size: exactly one
    giant is excluded from χ while P still reflects only the single largest.
    """
    if g.is_directed():
        raise ValueError(
            "g must be undirected; pass gt.GraphView(g, directed=False) "
            "to analyse a directed graph as undirected."
        )

    n_nodes = g.num_vertices()
    _, hist = gt.label_components(g)
    comp_sizes = sorted(hist.tolist(), reverse=True)

    S_1 = comp_sizes[0]
    P = S_1 / n_nodes

    # Second-largest component size (0 when there is only one component)
    S_2 = comp_sizes[1] if len(comp_sizes) > 1 else 0

    # Susceptibility: χ = (Σ s² · n_s − S_1²) / N
    # Subtracting S_1² excludes exactly one giant even when size ties exist.
    size_counts = Counter(comp_sizes)
    total_weighted = sum(s * s * n for s, n in size_counts.items())
    chi_num = total_weighted - S_1 * S_1
    chi = chi_num / n_nodes if chi_num > 0 else np.nan

    return PercolationResult(S_1=S_1, S_2=S_2, P=P, chi=chi)


# Upper bound on the entries of one dense block of shortest-path distances
# (8 bytes each); caps the memory correlation_length needs per BFS call.
_CL_MAX_BLOCK_ENTRIES: int = 1 << 22
# Small components are packed into blocks of up to this many vertices so they
# share one shortest-path call (fastest on 10^6-vertex Erdős–Rényi tests).
_CL_BATCH_VERTICES: int = 256


def _cl_squared_distance_sums(
    g: gt.Graph,
    labels: np.ndarray,
    sizes: np.ndarray,
    comps: np.ndarray,
) -> np.ndarray:
    """Sum squared shortest-path distances over ordered vertex pairs per component.

    Returns ``Σ_{i,j∈C} d_ij²`` for each component label C in *comps*, in the
    same order.  *labels* holds the component label of every vertex, indexed
    by vertex index; *sizes* holds the component sizes, indexed by label.
    Each component gets a contiguous block of a sparse adjacency matrix, and
    whole components are packed into blocks of up to ``_CL_BATCH_VERTICES``
    vertices that share one breadth-first shortest-path call.  Distances
    between different components of a block are infinite and ignored.

    ``gt.distance_histogram`` is deliberately avoided: on filtered graph views
    its parallel implementation returns wrong, run-dependent counts
    (graph-tool 2.98).
    """
    out = np.zeros(len(comps))
    multi = sizes[comps] > 1                    # singletons contribute nothing
    if not multi.any():
        return out
    comps = comps[multi]

    # Order the vertices of the requested components by component
    rank = np.full(len(sizes), -1, dtype=np.int64)
    rank[comps] = np.arange(len(comps))
    vertices = g.get_vertices()
    vertex_rank = rank[labels[vertices]]
    keep = vertex_rank >= 0
    by_comp = np.argsort(vertex_rank[keep], kind="stable")
    members = vertices[keep][by_comp]
    comp_of_row = vertex_rank[keep][by_comp]

    # Sparse adjacency in that order; edges of other components are dropped
    n = len(members)
    pos = np.full(g.num_vertices(ignore_filter=True), -1, dtype=np.int64)
    pos[members] = np.arange(n)
    edges = pos[g.get_edges()]
    edges = edges[edges[:, 0] >= 0]
    adj = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))

    bounds = np.zeros(len(comps) + 1, dtype=np.int64)
    bounds[1:] = np.cumsum(sizes[comps])
    row_sums = np.zeros(n)
    start = 0
    while start < n:
        # Extend the block by whole components; a large one forms its own block
        last = np.searchsorted(bounds, start + _CL_BATCH_VERTICES, side="right") - 1
        end = bounds[last] if bounds[last] > start else bounds[last + 1]
        block = adj[start:end, start:end]
        # Process the sources in chunks to bound the dense distance block
        step = max(1, _CL_MAX_BLOCK_ENTRIES // (end - start))
        for i in range(start, end, step):
            stop = min(i + step, end)
            dist = shortest_path(
                block,
                method="D",
                unweighted=True,
                directed=False,
                indices=np.arange(i - start, stop - start),
            )
            dist[np.isinf(dist)] = 0.0          # pairs in different components
            row_sums[i:stop] = np.square(dist).sum(axis=1)
        start = end

    out[multi] = np.bincount(comp_of_row, weights=row_sums, minlength=len(comps))
    return out


class CorrelationLengthResult(NamedTuple):
    """Return value of :func:`correlation_length`."""

    xi: float        # correlation length of the finite components
    xi_no_S2: float  # correlation length without the second-largest component
    R_S2: float      # radius of gyration of the second-largest component


def correlation_length(g: gt.Graph) -> CorrelationLengthResult:
    """Compute the correlation length of the finite components of a graph.

    Parameters
    ----------
    g : gt.Graph
        An undirected graph-tool graph.  Filtered graph views are supported.

    Returns
    -------
    CorrelationLengthResult
        Named tuple with three fields:

        ``xi`` — correlation length ξ of the finite components, i.e. all
        components except the largest:
        ξ² = Σ_C Σ_{i,j∈C} d_ij² / Σ_C s_C² = Σ_C 2R_C² · s_C² / Σ_C s_C²,
        where d_ij is the shortest-path distance between vertices i and j,
        and s_C and R_C are the size and radius of gyration of component C.
        ξ is the root-mean-square distance between two vertices drawn from
        the same finite component (ordered pairs, i = j included).  Returns
        ``nan`` when the graph has fewer than two components.

        ``xi_no_S2`` — the same correlation length with the second-largest
        component excluded as well.  Returns ``nan`` when the graph has
        fewer than three components.

        ``R_S2`` — radius of gyration of the second-largest component,
        R_S2² = Σ_{i,j∈S_2} d_ij² / (2 S_2²), i.e. half the mean squared
        distance between two of its vertices (pairs counted as for ``xi``).
        Returns ``nan`` when the graph has fewer than two components.

    Raises
    ------
    ValueError
        If *g* is directed.

    Notes
    -----
    ξ follows Stauffer & Aharony (1994), with distances measured in hops and
    the largest component standing in for the infinite cluster, as for the
    susceptibility in :func:`percolation`.  The pair form of the radius of
    gyration equals the root-mean-square distance from the centre of mass in
    Euclidean space; networks have no centre of mass, so the pair form serves
    as the definition.

    Because ξ² weights each component by s_C², the largest finite component
    can dominate it near the percolation threshold.  The three outputs are
    tied by the exact identity

        ξ² = w · 2R_S2² + (1 − w) · ξ_no_S2²,   w = S_2² / (χ N),

    where S_2 and χ are as returned by :func:`percolation` and N is the number
    of vertices.  Comparing ``xi`` with ``xi_no_S2`` thus shows how much of ξ
    is due to S_2 alone.

    Ties in component size are broken deterministically.  Shortest paths are
    computed within each finite component only, so the run time grows with
    Σ_C s_C² rather than with N².
    """
    if g.is_directed():
        raise ValueError(
            "g must be undirected; pass gt.GraphView(g, directed=False) "
            "to analyse a directed graph as undirected."
        )

    labels, hist = gt.label_components(g)
    sizes = hist.astype(np.int64)
    if len(sizes) < 2:
        return CorrelationLengthResult(xi=np.nan, xi_no_S2=np.nan, R_S2=np.nan)

    # Finite components, largest first; the stable sort breaks ties by label
    finite = np.argsort(-sizes, kind="stable")[1:]
    d2 = _cl_squared_distance_sums(g, labels.a, sizes, finite)
    s2 = sizes[finite].astype(float) ** 2

    xi = np.sqrt(d2.sum() / s2.sum())
    xi_no_S2 = np.sqrt(d2[1:].sum() / s2[1:].sum()) if len(finite) > 1 else np.nan
    R_S2 = np.sqrt(d2[0] / (2.0 * s2[0]))

    return CorrelationLengthResult(
        xi=float(xi), xi_no_S2=float(xi_no_S2), R_S2=float(R_S2)
    )


# Graphs larger than this use sampled ASPL estimation instead of all-pairs BFS.
_ASPL_SAMPLE_THRESHOLD: int = 40_000
# Default number of source vertices for sampled ASPL estimation.
_ASPL_SAMPLE_SIZE: int = 500


class SmallWorldResult(NamedTuple):
    """Return value of :func:`small_world`."""

    L: float       # average shortest path length of the LCC
    L_norm: float  # L / L_random  (Erdős–Rényi baseline)
    C: float       # average local clustering coefficient of the LCC
    C_norm: float  # C / C_random  (Erdős–Rényi baseline)


def small_world(
    g: gt.Graph,
    sample_size: int = _ASPL_SAMPLE_SIZE,
    seed: int | None = None,
) -> SmallWorldResult:
    """Compute small-world properties of the largest connected component.

    Parameters
    ----------
    g : gt.Graph
        An undirected graph-tool graph.
    sample_size : int, optional
        Number of source vertices used to *estimate* the average shortest
        path length when the LCC exceeds ``40 000`` nodes.  By the central
        limit theorem the estimation error is proportional to
        ``σ / √sample_size``, where σ is the standard deviation of
        per-vertex mean distances.  Must be an integer >= 1.  Default ``500``.
    seed : int or None, optional
        Seed for graph-tool's random-number generator (``gt.seed_rng``), which
        picks the sources for the sampled estimate.  Note that this reseeds
        graph-tool's global generator.  ``None`` leaves the generator as it
        is, which gives a non-deterministic result unless it was seeded
        before.  Default ``None``.

    Returns
    -------
    SmallWorldResult
        Named tuple with four fields:

        ``L`` — average shortest path length of the LCC.  Exact for graphs
        with N ≤ 40 000 nodes; estimated from the distances of
        ``sample_size`` randomly chosen sources (drawn without replacement)
        otherwise.

        ``L_norm`` — L / L_random, where L_random is the Fronczak et al.
        (2004) Erdős–Rényi baseline:
        (ln N − γ) / ln ⟨k⟩ + 0.5  (γ = Euler–Mascheroni constant).

        ``C`` — average local clustering coefficient of the LCC (Watts &
        Strogatz 1998): the mean over all vertices of c_i, the fraction of
        pairs of neighbours of vertex i that are themselves connected, with
        c_i = 0 for vertices of degree < 2.

        ``C_norm`` — C / C_random, where C_random is the expected C of an
        Erdős–Rényi graph with the same N and m.  There, a vertex of degree
        ≥ 2 has an expected c_i equal to the edge density p = 2m / (N(N−1)),
        and the other vertices count as zero, so C_random = p · P(k ≥ 2),
        with the binomial share of vertices of degree ≥ 2,
        P(k ≥ 2) = 1 − (1 − p)^(N−1) − (N − 1) p (1 − p)^(N−2).  Returns
        ``nan`` when the LCC is a single edge, where C_random = 0.

    Raises
    ------
    ValueError
        If *g* is directed, *sample_size* is not an integer >= 1, or the
        LCC has fewer than 2 vertices.

    Notes
    -----
    Small-world character is indicated by ``L_norm ≈ 1`` (path lengths
    comparable to a random graph) and ``C_norm >> 1`` (far more clustered).

    C_random is exact for an Erdős–Rényi graph with the LCC's N and m.  The
    LCC of a sparse random graph, however, is denser than the graph in which
    its triangles formed, so its C_norm stays below 1 when ⟨k⟩ is small
    (about 0.87 at ⟨k⟩ = 3 and 0.66 at ⟨k⟩ = 2 in simulations; close to 1
    from ⟨k⟩ ≈ 6 on).

    The exact path length favours speed over memory: it keeps all N²
    pairwise distances, about 6 GB at N = 40 000.  The sampled estimate
    needs memory proportional to N.
    """
    if g.is_directed():
        raise ValueError(
            "g must be undirected; pass gt.GraphView(g, directed=False) "
            "to analyse a directed graph as undirected."
        )
    if not int(sample_size) == sample_size or sample_size < 1:
        raise ValueError(f"sample_size must be an integer >= 1; got {sample_size!r}")

    lcc = gt.extract_largest_component(g, prune=True)
    n = lcc.num_vertices()
    m = lcc.num_edges()

    if n < 2:
        raise ValueError(
            f"LCC has {n} vertex; at least 2 are required to compute path lengths."
        )

    # ------------------------------------------------------------------ #
    # Average shortest path length
    # ------------------------------------------------------------------ #
    if n > _ASPL_SAMPLE_THRESHOLD:
        # Estimate L from the distances of a random sample of source vertices,
        # drawn without replacement; distance_histogram runs their BFS in
        # parallel.  Full all-pairs BFS is O(N(N+M)) and infeasible at this
        # scale.  The LCC is a pruned copy: on filtered graph views,
        # distance_histogram returns wrong counts when run in parallel.
        if seed is not None:
            gt.seed_rng(seed)
        counts, bins = gt.distance_histogram(lcc, samples=min(int(sample_size), n))
        L = float(np.sum(counts * bins[:-1]) / np.sum(counts))
    else:
        # Exact: all-pairs BFS into an N x N distance matrix, the fastest route
        dist = gt.shortest_distance(lcc)
        L = float(np.mean([dist[v].a.sum() / (n - 1) for v in lcc.vertices()]))

    # ------------------------------------------------------------------ #
    # Erdős–Rényi baselines
    # ------------------------------------------------------------------ #
    k = 2 * m / n                                    # mean degree
    L_random = float((np.log(n) - np.euler_gamma) / np.log(k) + 0.5)
    p = m / (n * (n - 1) / 2)                        # edge density
    # Expected C: vertices of degree >= 2 have E[c_i] = p, the others count as 0
    share = 1 - (1 - p) ** (n - 1) - (n - 1) * p * (1 - p) ** (n - 2)
    C_random = p * share

    # ------------------------------------------------------------------ #
    # Average local clustering coefficient, with c_i = 0 for degree < 2
    # ------------------------------------------------------------------ #
    local = gt.local_clustering(lcc).a
    degree = lcc.get_out_degrees(lcc.get_vertices())
    C = float(np.mean(np.where(degree >= 2, local, 0.0)))

    return SmallWorldResult(
        L=L,
        L_norm=L / L_random,
        C=C,
        C_norm=C / C_random if C_random > 0 else np.nan,
    )


# Possible degrees used in scale-free analysis
_VALID_DEGS = ("in", "out", "total")


class ScaleFreeResult(NamedTuple):
    """Return value of :func:`scale_free`."""

    alpha: float   # fitted power-law exponent
    k_min: int     # xmin: lower bound of the fitted power-law regime
    k_max: int     # maximum observed degree
    k_mean: float  # mean degree


def scale_free(
    g: gt.Graph,
    deg: Literal["in", "out", "total"] = "total",
) -> ScaleFreeResult:
    """Fit a power law to the degree distribution of the largest connected component.

    Parameters
    ----------
    g : gt.Graph
        A graph-tool graph.
    deg : {'in', 'out', 'total'}, optional
        Which degree to use.  For an undirected graph all three are
        equivalent.  Default ``'total'``.

    Returns
    -------
    ScaleFreeResult
        Named tuple with four fields:

        ``alpha`` — fitted power-law exponent of the degree distribution.

        ``k_min`` — xmin, the smallest degree at which the power law is
        taken to hold, chosen by ``powerlaw.Fit`` to minimise the
        Kolmogorov–Smirnov distance to the empirical distribution.

        ``k_max`` — maximum observed degree.

        ``k_mean`` — mean degree.

    Raises
    ------
    ValueError
        If *deg* is not one of ``('in', 'out', 'total')``.

    Notes
    -----
    Uses ``powerlaw.Fit`` (Alstott, Bullmore & Plenz 2014).  A network is
    typically called scale-free when 2 < alpha < 3.
    """
    if deg not in _VALID_DEGS:
        raise ValueError(f"deg must be one of {_VALID_DEGS}; got {deg!r}")

    lcc = gt.extract_largest_component(g, prune=True)
    k = lcc.degree_property_map(deg).a

    fit = pl.Fit(k, verbose=False)

    return ScaleFreeResult(
        alpha=float(fit.alpha),
        k_min=int(fit.xmin),
        k_max=int(k.max()),
        k_mean=float(k.mean()),
    )


_BOX_COVER_BIN: Path = Path("./bin/box_cover")


class FractalityResult(NamedTuple):
    """Return value of :func:`fractality`."""

    a: float              # power-law prefactor
    d_B: float            # box-counting (fractal) dimension; caller-fixed or fitted
    l: float | None       # exponential cutoff length; None when truncated=False
    boxes: dict[int, int] # {radius: box_count} as returned by the binary
    centers: dict[int, list[int]]  # {radius: [center_node_ids]}


def fractality(
    g: gt.Graph,
    truncated: bool = False,
    d_B: float | None = None,
    a_bounds: tuple[float, float] = (1.0, 1_000_000.0),
    d_B_bounds: tuple[float, float] = (1.0, 10.0),
    l_bounds: tuple[float, float] = (1.0, 1_000_000.0),
    rad_min: int = 1,
    rad_max: int | None = None,
    random_seed: int = 42,
    bin_path: Path | str = _BOX_COVER_BIN,
) -> FractalityResult:
    """Estimate the fractal dimension of a network via box-counting.

    Parameters
    ----------
    g : gt.Graph
        An undirected graph-tool graph.
    truncated : bool, optional
        Fit a truncated power law
        N_B(r) = a · r^(−d_B) · exp(−r / l) instead of a pure power law.
        Default ``False``.
    d_B : float or None, optional
        Fix the box-counting dimension to this value and fit only the
        prefactor *a* (and, when *truncated* is True, the cutoff *l*).
        When ``None``, *d_B* is estimated from the data.  Default ``None``.
    a_bounds : (float, float), optional
        ``(lower, upper)`` bounds on the fitted prefactor *a*.
        Default ``(1.0, 1_000_000.0)``.
    d_B_bounds : (float, float), optional
        ``(lower, upper)`` bounds on the fitted box-counting dimension.
        Ignored when *d_B* is fixed.  Default ``(1.0, 10.0)``.
    l_bounds : (float, float), optional
        ``(lower, upper)`` bounds on the fitted exponential cutoff *l*.
        Ignored when *truncated* is ``False``.  Default
        ``(1.0, 1_000_000.0)``.
    rad_min : int, optional
        Minimum box radius passed to the binary.  Default ``1``.
    rad_max : int or None, optional
        Maximum box radius passed to the binary.  When ``None`` (default),
        the pseudo-diameter of the LCC is used.
    random_seed : int, optional
        Random seed forwarded to the sketch-based binary.  Default ``42``.
    bin_path : Path or str, optional
        Path to the ``box_cover`` executable (Akiba et al. 2015).
        Default ``./bin/box_cover``.

    Returns
    -------
    FractalityResult
        Named tuple with five fields:

        ``a`` — power-law prefactor.

        ``d_B`` — box-counting (fractal) dimension.  Equals the
        caller-supplied value when *d_B* is given; fitted from data
        otherwise.

        ``l`` — exponential cutoff length.  ``None`` when
        *truncated* is ``False``.

        ``boxes`` — ``{radius: box_count}`` mapping as returned by the
        binary, suitable for direct plotting or inspection.

        ``centers`` — ``{radius: [node_ids]}`` mapping of box center
        node ids per radius.

    Raises
    ------
    FileNotFoundError
        If *bin_path* does not exist or the binary produces no output.
    subprocess.CalledProcessError
        If the box-covering binary exits with a non-zero status.
    ValueError
        If *g* is directed or *d_B* is not a positive number.

    Notes
    -----
    When the LCC diameter is less than 5, too few radius steps exist for
    reliable curve fitting; the result then has ``a`` and ``d_B`` set to
    ``nan``, ``l`` set to ``nan`` (``None`` when *truncated* is ``False``)
    and empty ``boxes`` and ``centers``.

    Requires the sketch-based box-covering binary from
    https://github.com/kenkoooo/graph-sketch-fractality (Akiba et al. 2015).
    """
    if g.is_directed():
        raise ValueError(
            "g must be undirected; pass gt.GraphView(g, directed=False) "
            "to analyse a directed graph as undirected."
        )
    if d_B is not None and d_B <= 0:
        raise ValueError(f"d_B must be a positive number; got {d_B!r}")

    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(f"Box-cover binary not found: {bin_path}")

    lcc = gt.extract_largest_component(g, prune=True)
    diameter = int(gt.pseudo_diameter(lcc)[0])

    if diameter < 5:
        return FractalityResult(
            a=np.nan,
            d_B=np.nan,
            l=np.nan if truncated else None,
            boxes={},
            centers={},
        )

    if rad_max is None:
        rad_max = diameter

    # Write edge list to a temporary file on the Linux filesystem so the
    # binary can read it regardless of where the notebook is stored.
    edges = [(int(e.source()), int(e.target())) for e in lcc.edges()]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tsv", delete=False, prefix="cs_edgelist_"
    ) as tmp:
        tmp_path = Path(tmp.name)
        for src, dst in edges:
            tmp.write(f"{src}\t{dst}\n")

    try:
        result = subprocess.run(
            [
                str(bin_path),
                "-type=tsv",
                f"-graph={tmp_path}",
                "-method=sketch",
                f"-rad_min={rad_min}",
                f"-rad_max={rad_max}",
                f"-random_seed={random_seed}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    # The binary prints "JLOG: <path>" on the last line that contains it.
    jlog_path: Path | None = None
    for line in (result.stdout + result.stderr).splitlines():
        if "JLOG:" in line:
            jlog_path = Path(line.split("JLOG:")[-1].strip())

    if jlog_path is None or not jlog_path.exists():
        # Fallback: most recently modified file in ./jlog/
        jlog_dir = Path("./jlog")
        candidates = [p for p in jlog_dir.iterdir() if p.is_file()]
        if not candidates:
            raise FileNotFoundError("box_cover produced no jlog output.")
        jlog_path = max(candidates, key=lambda p: p.stat().st_mtime)

    with jlog_path.open() as fh:
        data = json.load(fh)

    x = np.asarray(data["radius"], dtype=float)
    y = np.asarray(data["size"], dtype=float)
    boxes = dict(zip(data["radius"], data["size"]))
    centers = {int(k): v for d in data["centers"] for k, v in d.items()}

    # Curve-fitting functions
    def _pl(x, a, d):       return a * x ** (-d)
    def _pl_d(x, a):        return a * x ** (-d_B)       # type: ignore[operator]
    def _tpl(x, a, d, lv):  return a * x ** (-d) * np.exp(-x / lv)
    def _tpl_d(x, a, lv):   return a * x ** (-d_B) * np.exp(-x / lv)  # type: ignore[operator]

    sigma = np.maximum(np.sqrt(y), 1e-12)
    a_lo, a_hi = a_bounds
    d_lo, d_hi = d_B_bounds
    l_lo, l_hi = l_bounds
    if not truncated:
        if d_B is not None:
            (a,), _ = curve_fit(_pl_d, x, y, sigma=sigma, bounds=([a_lo], [a_hi]))
            return FractalityResult(a=float(a), d_B=d_B, l=None, boxes=boxes, centers=centers)
        else:
            (a, d_B_fit), _ = curve_fit(_pl, x, y, sigma=sigma, bounds=([a_lo, d_lo], [a_hi, d_hi]))
            return FractalityResult(a=float(a), d_B=float(d_B_fit), l=None, boxes=boxes, centers=centers)
    else:
        if d_B is not None:
            (a, lv), _ = curve_fit(_tpl_d, x, y, sigma=sigma, bounds=([a_lo, l_lo], [a_hi, l_hi]))
            return FractalityResult(a=float(a), d_B=d_B, l=float(lv), boxes=boxes, centers=centers)
        else:
            (a, d_B_fit, lv), _ = curve_fit(_tpl, x, y, sigma=sigma, bounds=([a_lo, d_lo, l_lo], [a_hi, d_hi, l_hi]))
            return FractalityResult(a=float(a), d_B=float(d_B_fit), l=float(lv), boxes=boxes, centers=centers)


# Number of tightening levels (the 1%..100% grid) over which Fisher Information
# is averaged for each window, following Ahmad et al. (2016).
_FI_TIGHTENING_LEVELS = 100
# Fisher Information of a single-state window; its theoretical maximum (a window
# collapsed into one state yields 4 * ((0 - 1)^2 + (1 - 0)^2) = 8).  Used to find
# the loosest tightening level at which a window first splits into >1 state.
_FI_SINGLE_STATE = 8.0


def _fi_size_of_states(values: np.ndarray, window: int, k: float) -> np.ndarray:
    """Auto-derive the size of states per variable: ``k`` * minimum sliding s.d.

    Reproduces the heuristic of the reference implementation's ``SOST`` step —
    scan every full-length sliding window, take the smallest per-variable
    standard deviation (the most stable stretch) and scale it by *k*
    (Chebyshev's inequality; *k* = 2 keeps ~75% of observations inside the box).
    Missing values (NaN) invalidate any window that contains them.
    """
    n_rows, n_vars = values.shape
    if window > n_rows:
        return np.zeros(n_vars)
    sw = sliding_window_view(values, window, axis=0)   # (n_win, n_vars, window)
    with np.errstate(invalid="ignore"):
        stds = np.std(sw, axis=2, ddof=1)
    stds = np.where(np.isnan(sw).any(axis=2), np.inf, stds)
    sost = np.min(stds, axis=0)
    sost[~np.isfinite(sost)] = 0.0                     # variable had no full window
    return sost * k


def _fi_state_sizes(bin_mat: np.ndarray, threshold: int) -> list[int]:
    """Greedily assign window points to states for one tightening threshold.

    ``bin_mat[m, n]`` holds the number of variables for which points *m* and *n*
    lie within the size of states (the diagonal is masked to -1).  The first
    unassigned point claims every still-unassigned point that matches it in at
    least *threshold* variables; the process repeats.  State sizes are returned
    in discovery order, which the Fisher Information sum below depends on.
    """
    w = bin_mat.shape[0]
    assigned = np.zeros(w, dtype=bool)
    sizes: list[int] = []
    for j in range(w):
        if assigned[j]:
            continue
        members = (~assigned) & (bin_mat[j] >= threshold)
        members[j] = True
        assigned |= members
        sizes.append(int(members.sum()))
    return sizes


def _fi_from_state_sizes(sizes: list[int], n_points: int) -> float:
    """Fisher Information of one window: ``4 * sum (sqrt(p_i) - sqrt(p_i+1))^2``.

    Probabilities ``p_i = size_i / n_points`` are padded with a leading and
    trailing zero so the amplitude ``sqrt(p)`` rises from and falls back to zero.
    """
    p = np.asarray(sizes, dtype=float) / n_points
    q = np.sqrt(np.concatenate(([0.0], p, [0.0])))
    return 4.0 * float(np.sum(np.diff(q) ** 2))


def fisher_information_multivariate(
    data: pd.DataFrame,
    window_size: int,
    window_increment: int = 1,
    time_col: str = "time",
    variable_col: str = "variable",
    value_col: str = "value",
    size_of_states: Sequence[float] | np.ndarray | None = None,
    sost_window: int | None = None,
    sost_k: float = 2.0,
    n_levels: int = _FI_TIGHTENING_LEVELS,
) -> pd.DataFrame:
    """Track system stability with the multivariate Fisher Information of Ahmad et al. (2016).

    Slides a fixed-width window along a multivariate time series and, for each
    window, bins time points into indistinguishable *states* and computes the
    Fisher Information (FI) of the resulting state-probability distribution.
    High FI means the system dwells in few states (stable/ordered); low FI means
    it spreads across many states (unstable/disordered).

    Parameters
    ----------
    data : pandas.DataFrame
        Multivariate time series in **long (tidy) format**: one row per
        observation, with a time column, a variable-name column and a numeric
        value column (see *time_col*, *variable_col*, *value_col*).  It is
        pivoted internally to a time-by-variable table; every
        ``(time, variable)`` combination absent from *data* becomes a missing
        value (NaN).  Time steps and variables keep their order of first
        appearance.  Each computed FI value is assigned to its window's final
        time step.
    window_size : int
        Number of consecutive time steps in each window (``>= 2``).
    window_increment : int, optional
        Number of time steps to advance the window between successive FI values.
        Default ``1``.
    time_col, variable_col, value_col : str, optional
        Column names in the long-format *data* holding, respectively, the time
        label, the variable name, and the (numeric) observation.  Default
        ``'time'``, ``'variable'``, ``'value'``.
    size_of_states : sequence of float or numpy.ndarray or None, optional
        Per-variable box half-width ``Δy`` (one non-negative value per variable,
        in the variables' order of first appearance).  When ``None`` (default)
        it is derived automatically as *sost_k* times the smallest
        sliding-window standard deviation of each variable (see
        :func:`_fi_size_of_states`).
    sost_window : int or None, optional
        Window length used when auto-deriving *size_of_states*.  Defaults to
        *window_size* when ``None``.  Ignored when *size_of_states* is given.
    sost_k : float, optional
        Chebyshev multiplier for the auto-derived size of states.  Default
        ``2.0``.  Ignored when *size_of_states* is given.
    n_levels : int, optional
        Number of tightening levels on the 1%..100% grid over which FI is
        averaged per window.  Default ``100``.

    Returns
    -------
    pandas.DataFrame
        One row per window, indexed by the window's final time label (index name
        taken from *time_col*), with a single column ``'fisher_information'``.
        Empty if no full-length window fits.

    Raises
    ------
    ValueError
        If *data* is not a non-empty DataFrame, if any of *time_col*,
        *variable_col*, *value_col* is missing, if *value_col* is non-numeric, if
        *data* holds duplicate ``(time, variable)`` pairs, if *window_size* < 2
        or exceeds the number of distinct time steps, if *window_increment* < 1,
        if *n_levels* < 1, or if a supplied *size_of_states* has the wrong length
        or holds a negative/non-finite value.

    Notes
    -----
    The long-format input is pivoted to a dense time-by-variable table, so peak
    memory and cost scale with ``n_time_steps * n_variables`` regardless of how
    sparse the long form is.  Missing observations (NaN) never satisfy the
    size-of-state criterion, but the tightening-level denominator uses the full
    variable count, so variables systematically absent across a window make even
    identical points fail to bin and inflate FI in that window.

    The state test compares points with a single *absolute* half-width ``Δy_i``
    per variable, so it assumes each variable varies on one characteristic
    scale.  Heavy-tailed / power-law-distributed variables (mostly small values
    with rare extremes) violate this: a fixed ``Δy_i`` is too loose near zero and
    too tight in the tail, and the auto-derived ``Δy_i`` collapses onto the
    calmest stretch, so the rare extremes dominate the state structure.  Apply a
    variance-stabilising transform (e.g. ``log``/``log1p``, Box-Cox or a rank
    transform) to such variables *before* calling this function; it performs no
    transformation of its own.

    Two time points share a state when ``|y_i(t_a) - y_i(t_b)| <= Δy_i`` for a
    sufficient fraction of variables *i*.  That fraction is the *tightening
    level* (TL): at TL = 100% all variables must match, and it relaxes down the
    grid.  Because the match count is an integer in ``[0, n_variables]``, the
    *n_levels* TLs collapse to at most ``n_variables`` distinct groupings, which
    are computed once each rather than once per level.  The reported FI is the
    mean over the TLs from the loosest level at which the window first splits
    into more than one state up to TL = 100%.

    This is computed independently per window; the reference implementation
    instead reused the single loosest split level found across *all* windows,
    which inflated the FI of the more stable windows.  Comparisons against NaN
    and the pairwise box test are fully vectorised with NumPy broadcasting.

    References
    ----------
    Ahmad, N., Derrible, S., Eason, T. & Cabezas, H. (2016). Using Fisher
    information to track stability in multivariate systems. *Royal Society Open
    Science*, 3:160582. DOI: 10.1098/rsos.160582.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"data must be a pandas DataFrame; got {type(data).__name__}")
    absent = [c for c in (time_col, variable_col, value_col) if c not in data.columns]
    if absent:
        raise ValueError(
            f"data is missing long-format column(s) {absent}; "
            f"present columns are {list(data.columns)}"
        )
    if data.shape[0] == 0:
        raise ValueError("data must be non-empty")
    if not pd.api.types.is_numeric_dtype(data[value_col]):
        raise ValueError(
            f"{value_col!r} must be numeric; got dtype {data[value_col].dtype}"
        )
    if not int(window_size) == window_size or window_size < 2:
        raise ValueError(f"window_size must be an integer >= 2; got {window_size!r}")
    if not int(window_increment) == window_increment or window_increment < 1:
        raise ValueError(f"window_increment must be an integer >= 1; got {window_increment!r}")
    if not int(n_levels) == n_levels or n_levels < 1:
        raise ValueError(f"n_levels must be an integer >= 1; got {n_levels!r}")

    # Reshape long -> wide (time x variable), preserving order of first
    # appearance for both axes.  Absent (time, variable) pairs become NaN.
    times = data[time_col].drop_duplicates().tolist()
    variables = data[variable_col].drop_duplicates().tolist()
    try:
        wide = data.pivot(index=time_col, columns=variable_col, values=value_col)
    except ValueError as exc:
        raise ValueError(
            f"data has duplicate ({time_col}, {variable_col}) pairs; expected one "
            f"value per variable per time step ({exc})"
        ) from exc
    wide = wide.reindex(index=times, columns=variables)

    values = wide.to_numpy(dtype=float)
    n_rows, n_vars = values.shape
    if window_size > n_rows:
        raise ValueError(
            f"window_size ({window_size}) exceeds the number of time steps ({n_rows})"
        )

    if size_of_states is None:
        sost = _fi_size_of_states(values, sost_window or window_size, float(sost_k))
    else:
        sost = np.asarray(size_of_states, dtype=float)
        if sost.shape != (n_vars,):
            raise ValueError(
                f"size_of_states must have one value per variable ({n_vars}); "
                f"got shape {sost.shape}"
            )
        if not np.all(np.isfinite(sost)) or np.any(sost < 0):
            raise ValueError("size_of_states values must be finite and non-negative")

    # Each of the n_levels tightening levels tl maps to an integer match-count
    # threshold ceil(n_vars * tl / n_levels) in [1, n_vars]; distinct thresholds
    # are grouped once and shared across the levels that map to them.
    tls = np.arange(1, n_levels + 1)
    thr_per_level = np.clip(np.ceil(n_vars * tls / n_levels).astype(int), 1, n_vars)
    unique_thresholds = np.unique(thr_per_level)

    end_labels: list = []
    fi_values: list[float] = []
    for start in range(0, n_rows, window_increment):
        win = values[start:start + window_size]
        if win.shape[0] != window_size:
            continue

        # Pairwise count of variables in which two points share a state.
        diff = np.abs(win[:, None, :] - win[None, :, :])   # (w, w, n_vars)
        with np.errstate(invalid="ignore"):                # NaN comparisons -> False
            within = diff <= sost
        bin_mat = within.sum(axis=2).astype(float)
        np.fill_diagonal(bin_mat, -1.0)                    # a point never bins with itself here

        fi_by_thr = {
            int(thr): _fi_from_state_sizes(_fi_state_sizes(bin_mat, int(thr)), window_size)
            for thr in unique_thresholds
        }
        fi_levels = np.array([fi_by_thr[int(thr)] for thr in thr_per_level])

        informative = ~np.isclose(fi_levels, _FI_SINGLE_STATE)
        k_init = int(np.argmax(informative)) if informative.any() else 0
        fi_values.append(float(fi_levels[k_init:].mean()))
        end_labels.append(wide.index[start + window_size - 1])

    index = pd.Index(end_labels, name=time_col)
    return pd.DataFrame({"fisher_information": fi_values}, index=index)


def avalanches(
    data: pd.DataFrame,
    time_col: str = "time",
    variable_col: str = "variable",
    weight_col: str | None = None,
    gap: int = 1,
    drop_censored_left: bool = False,
    drop_censored_right: bool = False,
) -> pd.DataFrame:
    """Extract avalanches from a long-format record of observations.

    An *avalanche* is an uninterrupted sequence of observations of one variable:
    it starts when the variable is observed after having been absent, continues
    for as long as no more than *gap* time steps pass between consecutive
    observations, and ends with the last observation before the next
    interruption.  One row of *data* is one observation.  Each avalanche has a
    *size* (total weight of its observations) and a *duration* (number of time
    steps it spans), whose joint distribution characterises how the process
    releases activity in bursts.

    Parameters
    ----------
    data : pandas.DataFrame
        Observations in **long (tidy) format**: one row per observation, with a
        time column and a variable column (see *time_col*, *variable_col*) and
        optionally a weight column (see *weight_col*).  Not modified; the
        function works on a sorted copy of the columns it needs.
    time_col : str, optional
        Column holding the time step of each observation.  Must be of integer
        dtype and encode *contiguous* units of one resolution (day, month,
        year, ...), because avalanches are cut on the difference between
        successive values.
        Calendar strings or timestamps must be discretised by the caller first,
        e.g. with ``pandas.factorize`` or ``.astype('category').cat.codes``.
        Default ``'time'``.
    variable_col : str, optional
        Column holding the variable whose sequence of observations is traced.
        Any hashable dtype; categorical dtype is preserved in the output.
        Default ``'variable'``.
    weight_col : str or None, optional
        Numeric column holding the weight of each observation.  ``None``
        (default) weights every observation as 1, so ``size`` counts
        observations.
    gap : int, optional
        Largest difference between the time steps of two successive
        observations of a variable that still continues the same avalanche
        (``>= 1``).  With the default ``1``, avalanches break at the first time
        step in which the variable is not observed.  Larger values tolerate
        longer interruptions; note that what counts as an interruption depends
        on the resolution of *time_col*, so a value chosen for daily data does
        not carry over to yearly data.
    drop_censored_left : bool, optional
        Drop avalanches that were already running when observation began (see
        *Returns*).  Default ``False``.
    drop_censored_right : bool, optional
        Drop avalanches that were still running when observation ended.  Default
        ``False``.

    Returns
    -------
    pandas.DataFrame
        One row per avalanche, ordered by variable and then by start time, with
        a fresh ``RangeIndex`` and the columns:

        ``{variable_col}`` - the variable, in the dtype it has in *data*.

        ``time_from``, ``time_to`` - time step of the first and of the last
        observation in the avalanche.

        ``size`` - total weight of the observations of the avalanche; a count
        when *weight_col* is ``None``.

        ``duration`` - ``time_to - time_from + 1``, so a variable observed
        within a single time step has duration 1.

        ``censored_left``, ``censored_right`` - ``True`` when the avalanche
        touches the first (respectively last) time step observed anywhere in
        *data*, and its true extent therefore reaches beyond the observation
        window.  The reported size and duration of such an avalanche are lower
        bounds.  Empty (with these columns and dtypes) if every avalanche is
        dropped.

    Raises
    ------
    ValueError
        If *data* is not a non-empty DataFrame, if *time_col*, *variable_col* or
        *weight_col* is missing from it, if any of those columns holds missing
        values, if *time_col* is not of integer dtype, if *weight_col* is
        non-numeric, or if *gap* is not an integer ``>= 1``.

    Notes
    -----
    Censoring is the reason the two ``censored_*`` columns exist: an avalanche
    that overlaps an edge of the observation window is truncated there, so its
    size and duration are underestimates.  Because large avalanches are the more
    likely to be cut, keeping them biases the tail of the size distribution
    downwards - which is exactly the tail a power-law fit (see
    :func:`scale_free`) is driven by.  Dropping them is not free either: it
    removes the largest events preferentially, and at coarse resolutions (few
    time steps) it can remove a sizeable share of all avalanches.  The columns
    are returned unconditionally so that the choice can be made, and reported,
    per analysis.

    Simultaneous observations are merged rather than separated: two
    observations of the same variable in the same time step contribute their
    weights to one avalanche of duration 1, whatever their order within the
    step.  The result therefore does not depend on how ties in *time_col* are
    ordered in *data*.

    Cost is dominated by the sort, ``O(n log n)`` in the number of
    observations; the avalanche boundaries themselves are found in one
    vectorised pass.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"data must be a pandas DataFrame; got {type(data).__name__}")
    columns = [time_col, variable_col] + ([weight_col] if weight_col is not None else [])
    absent = [c for c in columns if c not in data.columns]
    if absent:
        raise ValueError(
            f"data is missing long-format column(s) {absent}; "
            f"present columns are {list(data.columns)}"
        )
    if data.shape[0] == 0:
        raise ValueError("data must be non-empty")
    incomplete = [c for c in columns if data[c].isna().any()]
    if incomplete:
        raise ValueError(f"column(s) {incomplete} hold missing values")
    if not pd.api.types.is_integer_dtype(data[time_col]):
        raise ValueError(
            f"{time_col!r} must be of integer dtype, encoding contiguous time "
            f"units; got dtype {data[time_col].dtype}"
        )
    if weight_col is not None and not pd.api.types.is_numeric_dtype(data[weight_col]):
        raise ValueError(
            f"{weight_col!r} must be numeric; got dtype {data[weight_col].dtype}"
        )
    if not int(gap) == gap or gap < 1:
        raise ValueError(f"gap must be an integer >= 1; got {gap!r}")

    # Sorting by (variable, time) puts the observations of every variable into
    # one contiguous, chronologically ordered block, so avalanche boundaries
    # become a single comparison between neighbouring rows.
    ordered = data[columns].sort_values([variable_col, time_col], kind="stable")
    variables = ordered[variable_col]
    times = ordered[time_col].to_numpy(dtype=np.int64)
    if weight_col is None:
        weights = np.ones(times.size, dtype=np.int64)
    else:
        weights = ordered[weight_col].to_numpy()

    # A row opens a new avalanche when it is the first row, when the variable
    # changes, or when the time since the previous observation exceeds *gap*.
    # Comparing factorised codes keeps this test dtype-agnostic.
    codes = pd.factorize(variables, sort=False)[0]
    opens = np.empty(times.size, dtype=bool)
    opens[0] = True
    opens[1:] = (codes[1:] != codes[:-1]) | (np.diff(times) > int(gap))

    starts = np.flatnonzero(opens)
    ends = np.append(starts[1:], times.size) - 1

    time_from = times[starts]
    time_to = times[ends]  # times ascend within a block, so the last row is the maximum
    result = pd.DataFrame({
        variable_col: variables.to_numpy()[starts],
        "time_from": time_from,
        "time_to": time_to,
        "size": np.add.reduceat(weights, starts),
        "duration": time_to - time_from + 1,
        "censored_left": time_from == times.min(),
        "censored_right": time_to == times.max(),
    })
    if result[variable_col].dtype != variables.dtype:  # restore categorical etc.
        result[variable_col] = result[variable_col].astype(variables.dtype)

    keep = np.ones(result.shape[0], dtype=bool)
    if drop_censored_left:
        keep &= ~result["censored_left"].to_numpy()
    if drop_censored_right:
        keep &= ~result["censored_right"].to_numpy()
    return result[keep].reset_index(drop=True)
