from dataclasses import dataclass, field
from grapp.util.exceptions import UserInputError
from typing import List, Optional, Union
import numpy
import os
import pygrgl
import sys
import time

MISSING_ALLELE = "."
UNKNOWN_ALLELES = {".", "-", "N"}
DEFAULT_BATCH_SIZE = 100


@dataclass
class PolarizationStats:
    unpolarized: List[int] = field(default_factory=list)
    total_seen: int = 0
    emitted: int = 0
    already_polarized: int = 0
    swapped: int = 0
    inconsistent: int = 0
    after_alignment: int = 0
    no_alignment: int = 0
    non_snv_skipped: int = 0
    missing_remapped: int = 0
    mapping_stats: Optional["MutationMappingStatsSummary"] = None


@dataclass
class MutationMappingStatsSummary:
    total_mutations: int = 0
    empty_mutations: int = 0
    mutations_with_one_sample: int = 0
    mutations_with_no_candidates: int = 0
    reused_nodes: int = 0
    reused_node_coverage: int = 0
    reused_exactly: int = 0
    singleton_sample_edges: int = 0
    new_tree_nodes: int = 0
    samples_processed: int = 0
    num_candidates: int = 0
    reuse_size_bigger_than_hist_max: int = 0
    num_with_singletons: int = 0
    max_singletons: int = 0
    reused_mut_nodes: int = 0
    independent_node_visits: int = 0
    shared_node_visits: int = 0
    reuse_size_hist: List[int] = field(default_factory=list)
    independent_node_visits_by_batch: List[int] = field(default_factory=list)
    shared_node_visits_by_batch: List[int] = field(default_factory=list)
    batch_mutation_counts: List[int] = field(default_factory=list)
    batch_sample_counts: List[int] = field(default_factory=list)
    traversal_seconds_by_batch: List[float] = field(default_factory=list)
    candidate_seconds_by_batch: List[float] = field(default_factory=list)
    apply_seconds_by_batch: List[float] = field(default_factory=list)


def accumulate_mapping_stats(total, delta):
    total.total_mutations += delta.total_mutations
    total.empty_mutations += delta.empty_mutations
    total.mutations_with_one_sample += delta.mutations_with_one_sample
    total.mutations_with_no_candidates += delta.mutations_with_no_candidates
    total.reused_nodes += delta.reused_nodes
    total.reused_node_coverage += delta.reused_node_coverage
    total.reused_exactly += delta.reused_exactly
    total.singleton_sample_edges += delta.singleton_sample_edges
    total.new_tree_nodes += delta.new_tree_nodes
    total.samples_processed += delta.samples_processed
    total.num_candidates += delta.num_candidates
    total.reuse_size_bigger_than_hist_max += delta.reuse_size_bigger_than_hist_max
    total.num_with_singletons += delta.num_with_singletons
    total.max_singletons = max(total.max_singletons, delta.max_singletons)
    total.reused_mut_nodes += delta.reused_mut_nodes
    total.independent_node_visits += delta.independent_node_visits
    total.shared_node_visits += delta.shared_node_visits
    if len(total.reuse_size_hist) < len(delta.reuse_size_hist):
        total.reuse_size_hist.extend(
            [0] * (len(delta.reuse_size_hist) - len(total.reuse_size_hist))
        )
    for idx, value in enumerate(delta.reuse_size_hist):
        total.reuse_size_hist[idx] += value
    total.independent_node_visits_by_batch.extend(delta.independent_node_visits_by_batch)
    total.shared_node_visits_by_batch.extend(delta.shared_node_visits_by_batch)
    total.batch_mutation_counts.extend(delta.batch_mutation_counts)
    total.batch_sample_counts.extend(delta.batch_sample_counts)
    total.traversal_seconds_by_batch.extend(delta.traversal_seconds_by_batch)
    total.candidate_seconds_by_batch.extend(delta.candidate_seconds_by_batch)
    total.apply_seconds_by_batch.extend(delta.apply_seconds_by_batch)


@dataclass(frozen=True)
class SiteSwap:
    entries: list
    swap_index: int
    ancestral_allele: str


def profile_add(profile, key, seconds):
    if profile is not None:
        profile[key] = profile.get(key, 0.0) + seconds


def profile_inc(profile, key, amount=1):
    if profile is not None:
        profile[key] = profile.get(key, 0) + amount


def get_descendant_samples(grg, node_id, profile=None):
    if node_id == pygrgl.INVALID_NODE:
        return numpy.array([], dtype=numpy.uint32)
    timer_start = time.perf_counter()
    descendants = pygrgl.get_bfs_order(grg, pygrgl.TraversalDirection.DOWN, [node_id])
    samples = numpy.fromiter(
        (node for node in descendants if node < grg.num_samples),
        dtype=numpy.uint32,
    )
    profile_add(profile, "descendant_samples_s", time.perf_counter() - timer_start)
    profile_inc(profile, "descendant_samples_calls")
    profile_inc(profile, "descendant_samples_total_samples", len(samples))
    return samples


def append_mapping_timings(path, rows, write_header=False):
    if path is None or not rows:
        return
    import csv

    with open(path, "a", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def seconds_to_nanos(seconds):
    return int(round(seconds * 1_000_000_000))


def apply_remaps(
    grg,
    removals,
    remap_mutations,
    remap_samples,
    map_batch_size,
    thread_count,
    dense_membership_cutoff,
    timing_csv=None,
    timing_write_header=False,
    profile=None,
):
    if not removals and not remap_mutations:
        return

    timer_start = time.perf_counter()
    list(grg.get_node_mutation_miss())
    profile_add(profile, "get_node_mutation_miss_s", time.perf_counter() - timer_start)
    timer_start = time.perf_counter()
    for mut_id, node_id in removals:
        grg.remove_mutation(mut_id, node_id)
    profile_add(profile, "remove_mutations_s", time.perf_counter() - timer_start)
    profile_inc(profile, "remove_mutations_count", len(removals))

    if remap_mutations:
        timer_start = time.perf_counter()
        mapping_stats = pygrgl.map_mutations(
            grg,
            remap_mutations,
            remap_samples,
            verbose=False,
            mutation_batch_size=map_batch_size,
            thread_count=thread_count,
            dense_membership_cutoff=dense_membership_cutoff,
        )
        profile_add(profile, "map_mutations_wall_s", time.perf_counter() - timer_start)
        profile_inc(profile, "map_mutations_calls")
        profile_inc(profile, "map_mutations_count", len(remap_mutations))
        timing_rows = []
        batch_count = len(mapping_stats.traversal_seconds_by_batch)
        for batch_index in range(batch_count):
            traversal_seconds = mapping_stats.traversal_seconds_by_batch[batch_index]
            candidate_seconds = mapping_stats.candidate_seconds_by_batch[batch_index]
            apply_seconds = mapping_stats.apply_seconds_by_batch[batch_index]
            traversal_nanos = seconds_to_nanos(traversal_seconds)
            candidate_nanos = seconds_to_nanos(candidate_seconds)
            apply_nanos = seconds_to_nanos(apply_seconds)
            timing_rows.append(
                {
                    "batch_mutations": mapping_stats.batch_mutation_counts[batch_index],
                    "batch_samples": mapping_stats.batch_sample_counts[batch_index],
                    "thread_count": thread_count,
                    "dense_membership_cutoff": dense_membership_cutoff,
                    "traversal_nanos": traversal_nanos,
                    "candidate_nanos": candidate_nanos,
                    "apply_nanos": apply_nanos,
                    "measured_batch_nanos": traversal_nanos
                    + candidate_nanos
                    + apply_nanos,
                    "shared_node_visits": mapping_stats.shared_node_visits_by_batch[
                        batch_index
                    ],
                }
            )
        timer_start = time.perf_counter()
        append_mapping_timings(
            timing_csv,
            timing_rows,
            write_header=timing_write_header,
        )
        profile_add(profile, "write_map_timing_csv_s", time.perf_counter() - timer_start)
        return mapping_stats


# compute mutation removals and remaps for one site
def build_site_swap_remaps(grg, site_swaps, map_batch_size, stats, profile=None):
    if not site_swaps:
        return [], [], []

    removals = []
    remap_mutations = []
    remap_samples = []

    for site_swap in site_swaps:
        entries = site_swap.entries
        swap_mutation = entries[site_swap.swap_index][3]
        has_missing = any(entry[2] != pygrgl.INVALID_NODE for entry in entries)
        state = {
            "ancestral_allele": site_swap.ancestral_allele,
            "old_ref": entries[0][3].ref_allele,
            "position": int(entries[0][3].position),
            "time": swap_mutation.time,
            "full_remap": has_missing,
        }
        node_ids = [entry[1] for entry in entries]
        missing_node_id = entries[site_swap.swap_index][2]
        helper_node_ids = list(node_ids)
        if missing_node_id != pygrgl.INVALID_NODE:
            helper_node_ids.append(missing_node_id)

        timer_start = time.perf_counter()
        carrier_sets, old_ref_carriers = pygrgl.get_descendant_samples_and_complement(
            grg, helper_node_ids
        )
        profile_add(profile, "descendant_samples_s", time.perf_counter() - timer_start)
        profile_inc(profile, "descendant_samples_calls", len(helper_node_ids))
        profile_inc(
            profile,
            "descendant_samples_total_samples",
            sum(len(carrier_set) for carrier_set in carrier_sets),
        )
        profile_inc(profile, "inverse_samples_calls")
        profile_inc(profile, "inverse_samples_total_samples", len(old_ref_carriers))

        missing = (
            carrier_sets[len(entries)]
            if missing_node_id != pygrgl.INVALID_NODE
            else numpy.array([], dtype=numpy.uint32)
        )

        for row_index, entry in enumerate(entries):
            _mut_id, _node_id, _missing_node_id, mutation = entry
            carriers = carrier_sets[row_index]

            if row_index == site_swap.swap_index:
                removals.append((_mut_id, _node_id))
                if len(missing) > 0:
                    remap_mutations.append(
                        pygrgl.Mutation(
                            state["position"],
                            MISSING_ALLELE,
                            state["ancestral_allele"],
                            state["time"],
                        )
                    )
                    remap_samples.append(missing.tolist())
                    stats.missing_remapped += 1
            else:
                updated_mutation = pygrgl.Mutation(
                    state["position"],
                    mutation.allele,
                    state["ancestral_allele"],
                    mutation.time,
                )
                if state["full_remap"]:
                    removals.append((_mut_id, _node_id))
                    remap_mutations.append(updated_mutation)
                    remap_samples.append(carriers.tolist())
                else:
                    grg.set_mutation_by_id(_mut_id, updated_mutation)

        remap_mutations.append(
            pygrgl.Mutation(
                state["position"],
                state["old_ref"],
                state["ancestral_allele"],
                state["time"],
            )
        )
        remap_samples.append(old_ref_carriers.tolist())

    return removals, remap_mutations, remap_samples


# determine whether a site is already polarized or needs a swap
def classify_site(
    site_entries,
    ancestral_sequence,
    stats,
    drop_if_no_match,
    removals,
    site_swaps,
):
    if not site_entries:
        return

    position = int(site_entries[0][3].position)
    old_ref = site_entries[0][3].ref_allele
    stats.total_seen += len(site_entries)

    if len(old_ref) != 1 or any(
        len(mutation.allele) != 1
        for _mut_id, _node_id, _missing_node_id, mutation in site_entries
    ):
        stats.non_snv_skipped += len(site_entries)
        stats.unpolarized.append(position)
        if drop_if_no_match:
            removals.extend(
                (mut_id, node_id)
                for mut_id, node_id, _missing_node_id, _mutation in site_entries
            )
        return

    if ancestral_sequence is None:
        stats.after_alignment += 1
        stats.unpolarized.append(position)
        if drop_if_no_match:
            removals.extend(
                (mut_id, node_id)
                for mut_id, node_id, _missing_node_id, _mutation in site_entries
            )
        return

    ancestral_sequence = ancestral_sequence.upper()
    if ancestral_sequence[0] in UNKNOWN_ALLELES:
        stats.no_alignment += 1
        stats.unpolarized.append(position)
        if drop_if_no_match:
            removals.extend(
                (mut_id, node_id)
                for mut_id, node_id, _missing_node_id, _mutation in site_entries
            )
        return

    matches = []
    if allele_matches_ancestral(old_ref, ancestral_sequence):
        matches.append((-1, old_ref))

    for idx, (_mut_id, _node_id, _missing_node_id, mutation) in enumerate(site_entries):
        if allele_matches_ancestral(mutation.allele, ancestral_sequence):
            matches.append((idx, mutation.allele))

    if not matches:
        stats.inconsistent += len(site_entries)
        stats.unpolarized.append(position)
        if drop_if_no_match:
            removals.extend(
                (mut_id, node_id)
                for mut_id, node_id, _missing_node_id, _mutation in site_entries
            )
        return

    swap_index, ancestral_allele = max(matches, key=lambda item: len(item[1]))

    if swap_index < 0:
        stats.already_polarized += len(site_entries)
        stats.emitted += len(site_entries)
        return

    stats.swapped += 1
    stats.emitted += len(site_entries)
    site_swaps.append(
        SiteSwap(list(site_entries), swap_index, ancestral_allele.upper())
    )


def build_mut_lookup(grg):
    mut_lookup = [None] * grg.num_mutations
    for mut_id, node_id, missing_node_id in grg.get_mutation_node_miss():
        mut_lookup[int(mut_id)] = (int(node_id), int(missing_node_id))
    return mut_lookup


def equals_ignore_case(left, right):
    return left.upper() == right.upper()


def allele_matches_ancestral(allele, ancestral_sequence):
    return (
        bool(allele)
        and len(allele) <= len(ancestral_sequence)
        and equals_ignore_case(
            allele,
            ancestral_sequence[: len(allele)],
        )
    )


def polarize_grg(
    grg: Union[pygrgl.MutableGRG, str],
    ancestral_seq: str,
    drop_if_no_match: bool = True,
    map_batch_size: int = DEFAULT_BATCH_SIZE,
    map_input_batch_size: int = 4096,
    thread_count: int = 1,
    dense_membership_cutoff: float = 0.001,
    map_timing_csv: Optional[str] = None,
    output_file: Optional[str] = None,
):
    profile = {}
    profile_total_start = time.perf_counter()
    if map_batch_size <= 0:
        raise UserInputError("map_batch_size must be greater than zero")
    if map_input_batch_size <= 0:
        raise UserInputError("map_input_batch_size must be greater than zero")
    if thread_count <= 0:
        raise UserInputError("thread_count must be greater than zero")
    if dense_membership_cutoff < 0.0 or dense_membership_cutoff > 1.0:
        raise UserInputError("dense_membership_cutoff must be between 0 and 1")

    if isinstance(grg, str):
        timer_start = time.perf_counter()
        loaded_grg = pygrgl.load_mutable_grg(grg, load_up_edges=True)
        profile_add(profile, "load_mutable_grg_s", time.perf_counter() - timer_start)
        if loaded_grg is None:
            raise UserInputError(f"Failed to load GRG: {grg}")
        grg = loaded_grg

    ancestral_seq_by_position = "-" + ancestral_seq
    stats = PolarizationStats()
    timing_needs_header = bool(map_timing_csv) and not os.path.exists(map_timing_csv)
    timer_start = time.perf_counter()
    mut_lookup = build_mut_lookup(grg)
    profile_add(profile, "build_mut_lookup_s", time.perf_counter() - timer_start)
    total_mutations = grg.num_mutations
    site_entries = []  # type: ignore
    site_key = None
    pending_removals = []  # type: ignore
    pending_site_swaps = []  # type: ignore
    pending_swap_entry_count = 0
    pending_map_removals = []
    pending_remap_mutations = []
    pending_remap_samples = []
    mapping_stats = MutationMappingStatsSummary()

    def materialize_site_swaps():
        nonlocal pending_swap_entry_count
        if not pending_site_swaps:
            return
        timer_start = time.perf_counter()
        site_removals, remap_mutations, remap_samples = build_site_swap_remaps(
            grg, pending_site_swaps, map_input_batch_size, stats, profile=profile
        )
        profile_add(
            profile, "build_site_swap_remaps_s", time.perf_counter() - timer_start
        )
        pending_map_removals.extend(site_removals)
        pending_remap_mutations.extend(remap_mutations)
        pending_remap_samples.extend(remap_samples)
        pending_site_swaps.clear()
        pending_swap_entry_count = 0

    def flush_remaps():
        nonlocal timing_needs_header
        if not pending_map_removals and not pending_remap_mutations:
            return
        timer_start = time.perf_counter()
        remap_stats = apply_remaps(
            grg,
            pending_map_removals,
            pending_remap_mutations,
            pending_remap_samples,
            map_batch_size,
            thread_count,
            dense_membership_cutoff,
            timing_csv=map_timing_csv,
            timing_write_header=timing_needs_header,
            profile=profile,
        )
        profile_add(profile, "apply_remaps_total_s", time.perf_counter() - timer_start)
        if pending_remap_mutations:
            timing_needs_header = False
        if remap_stats is not None:
            accumulate_mapping_stats(mapping_stats, remap_stats)
        pending_map_removals.clear()
        pending_remap_mutations.clear()
        pending_remap_samples.clear()

    def flush_pending(force=False):
        if force:
            materialize_site_swaps()
            if pending_remap_mutations:
                pending_map_removals[:0] = pending_removals
                pending_removals.clear()
                flush_remaps()
            elif pending_removals:
                timer_start = time.perf_counter()
                apply_remaps(
                    grg,
                    pending_removals,
                    [],
                    [],
                    map_batch_size,
                    thread_count,
                    dense_membership_cutoff,
                    profile=profile,
                )
                profile_add(profile, "apply_remaps_total_s", time.perf_counter() - timer_start)
                pending_removals.clear()
            return

        if pending_swap_entry_count >= map_input_batch_size:
            materialize_site_swaps()
        if len(pending_remap_mutations) >= map_input_batch_size:
            pending_map_removals[:0] = pending_removals
            pending_removals.clear()
            flush_remaps()
            return
        if (
            pending_removals
            and not pending_site_swaps
            and len(pending_removals) >= map_input_batch_size
        ):
            timer_start = time.perf_counter()
            apply_remaps(
                grg,
                pending_removals,
                [],
                [],
                map_batch_size,
                thread_count,
                dense_membership_cutoff,
                profile=profile,
            )
            profile_add(profile, "apply_remaps_total_s", time.perf_counter() - timer_start)
            pending_removals.clear()

    def flush_site():
        nonlocal pending_swap_entry_count
        if not site_entries:
            return
        previous_swap_count = len(pending_site_swaps)
        position = int(site_entries[0][3].position)
        ancestral = (
            ancestral_seq_by_position[position]
            if 0 < position < len(ancestral_seq_by_position)
            else None
        )
        timer_start = time.perf_counter()
        classify_site(
            site_entries,
            ancestral,
            stats,
            drop_if_no_match,
            pending_removals,
            pending_site_swaps,
        )
        profile_add(profile, "classify_site_s", time.perf_counter() - timer_start)
        if len(pending_site_swaps) > previous_swap_count:
            pending_swap_entry_count += len(site_entries)
        flush_pending()

    timer_start = time.perf_counter()
    for mut_id in range(total_mutations):
        mutation = grg.get_mutation_by_id(mut_id)
        if mutation.allele == MISSING_ALLELE:
            continue

        lookup_entry = mut_lookup[mut_id]
        if lookup_entry is None:
            continue

        node_id, missing_node_id = lookup_entry
        if node_id == pygrgl.INVALID_NODE:
            continue

        key = (int(mutation.position), mutation.ref_allele)
        if site_key is not None and key != site_key:
            flush_site()
            site_entries.clear()
        site_key = key
        site_entries.append((mut_id, node_id, missing_node_id, mutation))
    profile_add(profile, "scan_mutations_s", time.perf_counter() - timer_start)

    flush_site()
    flush_pending(force=True)

    timer_start = time.perf_counter()
    grg.sort_mutations()
    profile_add(profile, "sort_mutations_s", time.perf_counter() - timer_start)
    if output_file is not None:
        timer_start = time.perf_counter()
        pygrgl.save_grg(grg, output_file)
        profile_add(profile, "save_grg_s", time.perf_counter() - timer_start)
    stats.mapping_stats = mapping_stats
    profile_add(profile, "polarize_total_s", time.perf_counter() - profile_total_start)
    profile["scan_and_classify_s"] = profile.get("scan_mutations_s", 0.0) + profile.get(
        "classify_site_s", 0.0
    )
    profile["materialize_remaps_s"] = profile.get(
        "build_site_swap_remaps_s", 0.0
    )
    profile_fields = (
        "polarize_total_s",
        "load_mutable_grg_s",
        "build_mut_lookup_s",
        "scan_and_classify_s",
        "materialize_remaps_s",
        "descendant_samples_s",
        "inverse_samples_s",
        "map_mutations_wall_s",
        "save_grg_s",
        "descendant_samples_calls",
        "descendant_samples_total_samples",
        "inverse_samples_total_samples",
    )
    print(
        "TIMING polarize_profile "
        + " ".join(
            f"{key}={profile.get(key, 0.0):.6f}"
            if isinstance(profile.get(key, 0.0), float)
            else f"{key}={profile.get(key, 0)}"
            for key in profile_fields
        ),
        file=sys.stderr,
        flush=True,
    )
    return stats
