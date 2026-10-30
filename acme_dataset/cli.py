"""Command line interface: ``acme-dataset <command> ...`` (or ``python -m acme_dataset``)."""

import argparse
import json
import sys
from pathlib import Path

from .constants import COMPONENTS, HUMAN_READABLE_RATE_HZ, RAW_BAGS_DIR


def _cmd_list(args) -> None:
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    from .download import list_remote

    try:
        names = list_remote(args.component, args.institution, revision=args.revision)
    except (EntryNotFoundError, RepositoryNotFoundError) as e:
        sys.exit(f"not available on Hugging Face: {e}")
    for name in names:
        print(name)


def _cmd_download(args) -> None:
    from .download import download

    path = download(args.component, args.institution, args.names, args.local_dir,
                    revision=args.revision, extract=args.extract, max_workers=args.workers)
    print(f"Downloaded to {path.resolve()}")


def _cmd_info(args) -> None:
    import pandas as pd

    path = Path(args.path)
    if (path / "timestamps.txt").exists():
        from .human_readable import Trajectory

        print(json.dumps(Trajectory(path).summary(), indent=2))
        return
    if path.suffix == ".zip" or (path / "metadata.yaml").exists() or path.suffix in (".db3", ".mcap"):
        from .rosbag import metadata_summary

        duration, topics = metadata_summary(path)
        print(f"bag: {path}\nduration: {duration:.1f} s\n")
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(topics.to_string(index=False))
        return
    if path.suffix.lower() in (".xlsx", ".csv"):
        from .tags import load_scenario_tags, tag_counts

        tags = load_scenario_tags(path)
        print(f"{len(tags)} tagged trajectories\n")
        print(tag_counts(tags, by_institution=True).to_string())
        return
    from .bev import BEVIndex
    from .human_readable import find_trajectory_dirs
    from .rosbag import find_bags

    bags = find_bags(path / RAW_BAGS_DIR) if (path / RAW_BAGS_DIR).is_dir() else []
    trajectories = find_trajectory_dirs(path)
    sessions = BEVIndex(path).sessions()
    if bags:
        counts = pd.Series([p.parent.name for p in bags]).value_counts().sort_index()
        print(f"raw bags: {len(bags)}\n{counts.to_string()}\n")
    if trajectories:
        counts = pd.Series([p.parent.name for p in trajectories]).value_counts().sort_index()
        print(f"human-readable trajectories: {len(trajectories)}\n{counts.to_string()}\n")
    if len(sessions):
        print(f"BEV sessions: {len(sessions)}\n{sessions.groupby('dataset').size().to_string()}")
    if not bags and not trajectories and not len(sessions):
        sys.exit(f"nothing recognised at {path}")


def _cmd_stats(args) -> None:
    from .stats import dataset_stats, summarize

    stats = dataset_stats(args.root, args.institution, masks=args.masks, workers=args.workers)
    if stats.empty:
        sys.exit(f"no trajectories found below {args.root}")
    if args.out:
        stats.to_csv(args.out, index=False)
        print(f"Wrote per-trajectory statistics to {args.out}")
    print(summarize(stats).round(2).to_string())


def _convert_one(job):
    from .convert import convert_bag

    bag, args = job
    try:
        return bag, convert_bag(bag, args.out, args.institution, rate=args.rate, pcd_format=args.pcd_format,
                                bbox_encoding=args.bbox_encoding, overwrite=args.overwrite), None
    except Exception as e:  # report and continue with the other bags
        return bag, None, repr(e)


def _cmd_convert(args) -> None:
    from concurrent.futures import ProcessPoolExecutor

    from .rosbag import find_bags

    bags = [b for path in args.bags for b in find_bags(path)]
    if not bags:
        sys.exit("no bags found")
    jobs = [(b, args) for b in bags]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(_convert_one, jobs))
    else:
        results = [_convert_one(job) for job in jobs]
    for bag, out, error in results:
        print(f"{bag} -> {out}" if error is None else f"{bag} FAILED: {error}")


def _cmd_bev_export(args) -> None:
    from .bev import BEVIndex, BEVSession

    index = BEVIndex(args.root)
    sessions = index.sessions()
    if args.dataset:
        sessions = sessions[sessions["dataset"].isin(args.dataset)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kwargs = {"coordinate_bounds": None} if args.no_bounds else {} if args.bounds is None \
        else {"coordinate_bounds": tuple(args.bounds)}
    for dataset, session_id in zip(sessions["dataset"], sessions["session"]):
        try:
            session = BEVSession(index, dataset, session_id)
            df = session.trajectories(filtered=not args.raw, target_fps=args.target_fps, **kwargs)
            if args.tags:
                if session.masks():
                    df = session.tag_segments(df)
                else:
                    print(f"{dataset}/{session_id}: no tag masks, exporting untagged trajectories")
                    df = df.assign(tag_id=None, tag_name=None, segment=None)
        except (FileNotFoundError, ValueError) as e:
            print(f"{dataset}/{session_id}: skipped ({e})")
            continue
        df.insert(0, "session", session_id)
        df.insert(0, "dataset", dataset)
        path = out / dataset / f"{session_id}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"{dataset}/{session_id}: {df['ped_id'].nunique()} pedestrians -> {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acme-dataset", description="Tools for the ACME social navigation dataset.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="list institutions / items available on Hugging Face")
    p.add_argument("component", choices=sorted(COMPONENTS))
    p.add_argument("--institution", "-i")
    p.add_argument("--revision")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("download", help="download (part of) the dataset from Hugging Face")
    p.add_argument("component", choices=sorted(COMPONENTS))
    p.add_argument("--institution", "-i", help="e.g. cmu, gmu, keio, miraikan, uex, umich")
    p.add_argument("--names", "-n", nargs="+", help="specific bags / trajectories / BEV datasets")
    p.add_argument("--local-dir", "-o", default="ACME-dataset")
    p.add_argument("--extract", action=argparse.BooleanOptionalAction, default=None,
                   help="unzip downloaded archives (default: only for human_readable)")
    p.add_argument("--revision")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("info", help="summarise a trajectory folder, bag, tag file or dataset folder")
    p.add_argument("path")
    p.set_defaults(func=_cmd_info)

    p = sub.add_parser("stats", help="per-trajectory statistics of the human-readable data")
    p.add_argument("root", help="dataset root, sampled_data/, institution or trajectory folder")
    p.add_argument("--institution", "-i")
    p.add_argument("--out", help="write per-trajectory statistics to this CSV")
    p.add_argument("--masks", action="store_true", help="also compute person pixel coverage (slow)")
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=_cmd_stats)

    p = sub.add_parser("convert", help="convert raw ROS 2 bags into the human-readable format (no ROS needed)")
    p.add_argument("bags", nargs="+", help="bag folders, .zip archives, or folders containing them")
    p.add_argument("--out", "-o", required=True, help="output folder (one sub-folder per bag)")
    p.add_argument("--institution", "-i", help="topic configuration to use (guessed from the path if omitted)")
    p.add_argument("--rate", type=float, default=HUMAN_READABLE_RATE_HZ, help="sampling rate in Hz (default: 4)")
    p.add_argument("--pcd-format", choices=["npz", "pcd", "both"], default="pcd")
    p.add_argument("--bbox-encoding", choices=["auto", "center", "legacy"], default="auto")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=_cmd_convert)

    p = sub.add_parser("bev-export", help="export BEV pedestrian trajectories in metres to CSV")
    p.add_argument("root", help="dataset root, BEV/ or BEV/BEV_annotations/")
    p.add_argument("--out", "-o", required=True)
    p.add_argument("--dataset", "-d", nargs="+", help="only these BEV datasets (e.g. bonn nus cmu)")
    p.add_argument("--raw", action="store_true", help="no filtering (all annotated points)")
    p.add_argument("--target-fps", type=float, help="downsample like transform_trajectories.py (e.g. 2.5)")
    p.add_argument("--bounds", type=float, nargs=2, metavar=("MIN", "MAX"),
                   help="label points outside [MIN, MAX] metres as out_of_bounds (default: -25 25)")
    p.add_argument("--no-bounds", action="store_true", help="disable the out_of_bounds labelling")
    p.add_argument("--tags", action="store_true", help="split into per-semantic-tag segments")
    p.set_defaults(func=_cmd_bev_export)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
