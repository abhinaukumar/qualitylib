import argparse

from qualitylib.tools import import_python_file, read_dataset
from qualitylib.runner import Runner


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run feature extractors and store results')
    parser.add_argument('--dataset', help='Path to dataset file for which to extract features', type=str)
    parser.add_argument('--fex_name', help='Name of feature extractor', type=str)
    parser.add_argument('--fex_version', help='Version of feature extractor', type=str, default=None)
    parser.add_argument('--processes', help='Number of parallel processes', type=str, default=1)
    return parser


def main() -> None:
    args = get_parser().parse_args()

    dataset = import_python_file(args.dataset)
    assets = read_dataset(dataset, shuffle=True)
    runner = Runner(args.fex_name, args.fex_version, processes=args.processes, use_cache=True)  # Reads from stored results if available, else stores results.
    runner(assets, return_results=False)  # Only extract features, do not use for anything.


if __name__ == '__main__':
    main()