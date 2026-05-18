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
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix
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

    S_1: int    # size of largest component (0 when only one component)
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
        Named tuple with three fields:

        ``S_1`` — size of the largest component.  Returns ``0`` when
        the graph consists of exactly one component.

        ``S_2`` — size of the second-largest component.  Returns ``0`` when
        the graph consists of exactly one component.

        ``P`` — percolation probability S_1 / N, where S_1 is the largest
        component size and N the total number of vertices.

        ``chi`` — susceptibility χ = (Σ_s s² · n_s − S_1²) / N, where n_s
        is the number of components of size s.  This is the size-weighted
        expected component size of a randomly chosen node, with the single
        largest component excluded.  Returns ``nan`` when the graph consists
        of exactly one component (susceptibility is undefined).

    Notes
    -----
    The susceptibility formula subtracts S_1² from the full weighted sum
    rather than excluding the largest *size class*.  This correctly handles
    the rare case where two components share the maximum size: exactly one
    giant is excluded from χ while P still reflects only the single largest.
    """
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


# Graphs larger than this use sampled ASPL estimation instead of all-pairs BFS.
_ASPL_SAMPLE_THRESHOLD: int = 40_000
# Default number of source vertices for sampled ASPL estimation.
_ASPL_SAMPLE_SIZE: int = 500


class SmallWorldResult(NamedTuple):
    """Return value of :func:`small_world`."""

    L: float       # average shortest path length of the LCC
    L_norm: float  # l / l_random  (Erdős–Rényi baseline)
    C: float       # average clustering coefficient of the LCC
    C_norm: float  # c / c_random  (Erdős–Rényi baseline)


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
        per-vertex mean distances.  Default ``500``.
    seed : int or None, optional
        Seed for the random-number generator used in source sampling.
        ``None`` gives a non-deterministic result.  Default ``None``.

    Returns
    -------
    SmallWorldResult
        Named tuple with four fields:

        ``l`` — average shortest path length of the LCC.  Exact for graphs
        with N ≤ 40 000 nodes; estimated by averaging single-source mean
        distances over ``sample_size`` randomly chosen sources otherwise.

        ``l_ratio`` — l / l_random, where l_random is the Fronczak et al.
        (2004) Erdős–Rényi baseline:
        (ln N − γ) / ln ⟨k⟩ + 0.5  (γ = Euler–Mascheroni constant).

        ``c`` — average clustering coefficient of the LCC computed by
        ``graph_tool.global_clustering``.

        ``c_ratio`` — c / c_random, where c_random = 2m / (N(N−1)) is the
        edge density, equal to the expected clustering of an Erdős–Rényi
        graph with the same N and m.

    Raises
    ------
    ValueError
        If the LCC has fewer than 2 vertices.

    Notes
    -----
    Small-world character is indicated by ``l_ratio ≈ 1`` (path lengths
    comparable to a random graph) and ``c_ratio >> 1`` (far more clustered).
    """
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
        # Estimate l by averaging single-source mean distances over a random
        # sample of source vertices.  Full all-pairs BFS is O(N(N+M)) and
        # infeasible at this scale; each single-source BFS is O(N+M).
        rng = np.random.default_rng(seed)
        source_indices = rng.choice(n, size=min(sample_size, n), replace=False)
        l = float(np.mean([
            gt.shortest_distance(lcc, source=lcc.vertex(int(i))).a.sum() / (n - 1)
            for i in source_indices
        ]))
    else:
        dist = gt.shortest_distance(lcc)
        l = float(np.mean([dist[v].a.sum() / (n - 1) for v in lcc.vertices()]))

    # ------------------------------------------------------------------ #
    # Erdős–Rényi baselines
    # ------------------------------------------------------------------ #
    k = 2 * m / n                                    # mean degree
    l_random = (np.log(n) - np.euler_gamma) / np.log(k) + 0.5
    c_random = m / (n * (n - 1) / 2)                # edge density

    # ------------------------------------------------------------------ #
    # Average clustering coefficient
    # ------------------------------------------------------------------ #
    c = gt.global_clustering(lcc)[0]

    return SmallWorldResult(
        L=l,
        L_norm=l / l_random,
        C=c,
        C_norm=c / c_random,
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
        Named tuple with four fields:

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
        If *d_B* is not a positive number.

    Notes
    -----
    Returns ``FractalityResult(nan, nan, nan, [])`` when the LCC diameter
    is less than 5, as too few radius steps exist for reliable curve fitting.

    Requires the sketch-based box-covering binary from
    https://github.com/kenkoooo/graph-sketch-fractality (Akiba et al. 2015).
    """
    if d_B is not None and d_B <= 0:
        raise ValueError(f"d_B must be a positive number; got {d_B!r}")

    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(f"Box-cover binary not found: {bin_path}")

    lcc = gt.extract_largest_component(g, prune=True)
    diameter = int(gt.pseudo_diameter(lcc)[0])

    if diameter < 5:
        return FractalityResult(a=np.nan, d_B=np.nan, l=np.nan, boxes={}, centers={})

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
