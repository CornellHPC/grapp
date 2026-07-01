from grapp.popgen.polarize import (
    polarize_grg,
    PolarizationStats,
    DEFAULT_BATCH_SIZE,
    MutationMappingStatsSummary,
    accumulate_mapping_stats,
)
from grapp.util.exceptions import UserInputError
from grapp.util.parallel import split_and_run
from typing import Tuple, Any, Optional, Union
import argparse
import os
import pyfaidx
import pygrgl
import sys


def load_fasta(path: str) -> Tuple[Any, str]:
    fasta = pyfaidx.Fasta(path, as_raw=False, sequence_always_upper=True)
    contigs = list(fasta.keys())
    if len(contigs) != 1:
        raise ValueError(
            "FASTA must contain exactly one contig for grg polarize. "
            f"Found: {', '.join(contigs)}"
        )
    return fasta, contigs[0]


def polarize_grg_from_fasta(
    grg: Union[pygrgl.MutableGRG, str],
    fasta_file: str,
    drop_if_no_match: bool = True,
    map_batch_size: int = DEFAULT_BATCH_SIZE,
    map_input_batch_size: int = 4096,
    thread_count: int = 1,
    dense_membership_mode: str = "cutoff",
    map_timing_csv: Optional[str] = None,
    output_file: Optional[str] = None,
) -> PolarizationStats:
    fasta, contig = load_fasta(fasta_file)
    ancestral_sequence = str(fasta[contig][:])
    return polarize_grg(
        grg,
        ancestral_sequence,
        drop_if_no_match,
        map_batch_size,
        map_input_batch_size=map_input_batch_size,
        thread_count=thread_count,
        dense_membership_mode=dense_membership_mode,
        map_timing_csv=map_timing_csv,
        output_file=output_file,
    )


def _add_stats(total: PolarizationStats, stats: PolarizationStats) -> None:
    total.unpolarized.extend(stats.unpolarized)
    total.total_seen += stats.total_seen
    total.emitted += stats.emitted
    total.already_polarized += stats.already_polarized
    total.swapped += stats.swapped
    total.inconsistent += stats.inconsistent
    total.after_alignment += stats.after_alignment
    total.no_alignment += stats.no_alignment
    total.non_snv_skipped += stats.non_snv_skipped
    total.missing_remapped += stats.missing_remapped
    if stats.mapping_stats is not None:
        if total.mapping_stats is None:
            total.mapping_stats = MutationMappingStatsSummary()
        accumulate_mapping_stats(total.mapping_stats, stats.mapping_stats)


def _grg_part_filename(grg_or_file, context, part_index):
    if isinstance(grg_or_file, str):
        return grg_or_file
    input_file = os.path.join(context["dir"], f"polarize-part-{part_index}.input.grg")
    pygrgl.save_grg(grg_or_file, input_file)
    return input_file


def _polarize_part(grg_or_file, context):
    part_index = context.setdefault("part_index", 0)
    context["part_index"] += 1
    grg_file = _grg_part_filename(grg_or_file, context, part_index)
    base = os.path.basename(grg_file)
    output_file = os.path.join(context["dir"], f"{part_index}.{base}.polar.grg")
    timing_csv = None
    if context["map_timing_csv"] is not None:
        timing_csv = os.path.join(context["dir"], f"{part_index}.{base}.map-timing.csv")
    stats = polarize_grg_from_fasta(
        grg_file,
        context["fasta_file"],
        drop_if_no_match=context["drop_if_no_match"],
        map_batch_size=context["map_batch_size"],
        map_input_batch_size=context["map_input_batch_size"],
        thread_count=context["thread_count"],
        dense_membership_mode=context["dense_membership_mode"],
        map_timing_csv=timing_csv,
        output_file=output_file,
    )
    return output_file, stats, timing_csv


def _merge_timing_csvs(part_results, output_csv):
    if output_csv is None:
        return
    with open(output_csv, "w") as out:
        wrote_header = False
        for _part_file, _stats, timing_csv in part_results:
            if timing_csv is None or not os.path.exists(timing_csv):
                continue
            with open(timing_csv) as inp:
                header = inp.readline()
                if not header:
                    continue
                if not wrote_header:
                    out.write(header)
                    wrote_header = True
                for line in inp:
                    out.write(line)


def _merge_polarized_parts(_grg_parts, part_results, context):
    part_files = [part_file for part_file, _stats, _timing_csv in part_results]
    target = pygrgl.load_mutable_grg(part_files[0], load_up_edges=True)
    if len(part_files) > 1:
        target.merge(part_files[1:])
    pygrgl.save_grg(target, context["output_file"])

    total = PolarizationStats()
    total.mapping_stats = MutationMappingStatsSummary()
    for _part_file, stats, _timing_csv in part_results:
        _add_stats(total, stats)
    _merge_timing_csvs(part_results, context["map_timing_csv"])
    return total


def add_options(subparser):
    subparser.add_argument("grg_input", help="Input GRG file to polarize")
    subparser.add_argument(
        "fasta_file", help="FASTA containing (only) the ancestral sequence"
    )
    subparser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        required=True,
        help="Output GRG file",
    )
    subparser.add_argument(
        "--keep-no-match",
        action="store_true",
        help="Keep mutations with no matching allele in the FASTA (instead of dropping them)",
    )
    subparser.add_argument(
        "--map-batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Internal C++ mutation mapping batch size",
    )
    subparser.add_argument(
        "--map-input-batch-size",
        type=int,
        default=4096,
        help="Number of remap mutations to pass from Python to C++ per call",
    )
    subparser.add_argument(
        "--dense-membership-mode",
        choices=("cutoff", "never"),
        default="cutoff",
        help="Use dense membership sets at the cutoff, or never use them",
    )
    subparser.add_argument(
        "--map-timing-csv",
        help="Optional CSV file for per-call mutation mapping timings",
    )
    subparser.add_argument(
        "--thread-count",
        type=int,
        default=1,
        help="Number of worker threads to use for candidate processing during remaps",
    )
    subparser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of split GRG parts to polarize in parallel",
    )
    subparser.add_argument(
        "--split-threshold",
        type=int,
        default=1_000_000,
        help="Basepair threshold for splitting the GRG before polarizing",
    )
    subparser.add_argument(
        "--temp-dir",
        help="Directory for split GRG parts and intermediate polarized parts",
    )


def run(args):
    if not os.path.isfile(args.grg_input):
        print(f"Input GRG file does not exist: {args.grg_input}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(args.fasta_file):
        print(f"FASTA file does not exist: {args.fasta_file}", file=sys.stderr)
        sys.exit(2)

    try:
        if args.jobs > 1:
            stats = split_and_run(
                args.grg_input,
                _polarize_part,
                _merge_polarized_parts,
                {
                    "fasta_file": args.fasta_file,
                    "drop_if_no_match": not args.keep_no_match,
                    "map_batch_size": args.map_batch_size,
                    "map_input_batch_size": args.map_input_batch_size,
                    "thread_count": args.thread_count,
                    "dense_membership_mode": args.dense_membership_mode,
                    "map_timing_csv": args.map_timing_csv,
                    "output_file": args.output_file,
                },
                jobs=args.jobs,
                temp_dir=args.temp_dir,
                split_threshold=args.split_threshold,
                verbose=True,
            )
        else:
            stats = polarize_grg_from_fasta(
                args.grg_input,
                args.fasta_file,
                drop_if_no_match=not args.keep_no_match,
                map_batch_size=args.map_batch_size,
                map_input_batch_size=args.map_input_batch_size,
                thread_count=args.thread_count,
                dense_membership_mode=args.dense_membership_mode,
                map_timing_csv=args.map_timing_csv,
                output_file=args.output_file,
            )
    except (UserInputError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)

    print("Polarization complete")
    print(f"  Total seen:           {stats.total_seen}")
    print(f"  Emitted:              {stats.emitted}")
    print(f"  Already polarized:    {stats.already_polarized}")
    print(f"  Swapped:              {stats.swapped}")
    print(f"  Inconsistent:         {stats.inconsistent}")
    print(f"  After alignment end:  {stats.after_alignment}")
    print(f"  Alignment mismatch:   {stats.no_alignment}")
    print(f"  Non-SNVs skipped:     {stats.non_snv_skipped}")
    print(f"  Missingness remapped: {stats.missing_remapped}")
    if stats.mapping_stats is not None:
        mapping = stats.mapping_stats
        print("  Mutation mapping:")
        print(f"    Total mutations:     {mapping.total_mutations}")
        print(f"    Empty mutations:     {mapping.empty_mutations}")
        print(f"    One-sample muts:     {mapping.mutations_with_one_sample}")
        print(f"    No candidates:       {mapping.mutations_with_no_candidates}")
        print(f"    Reused nodes:        {mapping.reused_nodes}")
        print(f"    Reused coverage:     {mapping.reused_node_coverage}")
        print(f"    Reused exactly:      {mapping.reused_exactly}")
        print(f"    Singleton edges:     {mapping.singleton_sample_edges}")
        print(f"    New tree nodes:      {mapping.new_tree_nodes}")
        print(f"    Samples processed:   {mapping.samples_processed}")
        print(f"    Candidates:          {mapping.num_candidates}")
        print(f"    Reuse hist overflow:  {mapping.reuse_size_bigger_than_hist_max}")
        print(f"    With singletons:     {mapping.num_with_singletons}")
        print(f"    Max singletons:      {mapping.max_singletons}")
        print(f"    Reused mut nodes:    {mapping.reused_mut_nodes}")
        print(f"    Independent visits:  {mapping.independent_node_visits}")
        print(f"    Shared visits:       {mapping.shared_node_visits}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    run(args)
