from grapp.popgen.polarize import (
    polarize_grg,
    PolarizationStats,
    DEFAULT_BATCH_SIZE,
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
    output_file: Optional[str] = None,
) -> PolarizationStats:
    fasta, contig = load_fasta(fasta_file)
    ancestral_sequence = str(fasta[contig][:])
    return polarize_grg(
        grg,
        ancestral_sequence,
        drop_if_no_match,
        map_batch_size,
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
    stats = polarize_grg_from_fasta(
        grg_file,
        context["fasta_file"],
        drop_if_no_match=context["drop_if_no_match"],
        map_batch_size=context["map_batch_size"],
        output_file=output_file,
    )
    return output_file, stats


def _merge_polarized_parts(_grg_parts, part_results, context):
    part_files = [part_file for part_file, _stats in part_results]
    target = pygrgl.load_mutable_grg(part_files[0], load_up_edges=True)
    if len(part_files) > 1:
        target.merge(part_files[1:])
    pygrgl.save_grg(target, context["output_file"])

    total = PolarizationStats()
    for _part_file, stats in part_results:
        _add_stats(total, stats)
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
        help="Number of flipped mutations to process per graph traversal; larger values use more RAM",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    run(args)
