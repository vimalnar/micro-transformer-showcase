"""Example-only state isolation and exact checkpoint comparisons."""
import argparse
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


@contextmanager
def options(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--state', type=Path, help='Keep state here instead of a temporary directory')
    parser.add_argument('--train', action='store_true', help='Run real training before restoring the checkpoint')
    args = parser.parse_args()
    if args.state is not None:
        yield args, args.state
    else:
        with TemporaryDirectory(prefix='micro-transformer-example-') as directory:
            yield args, Path(directory)


def assert_equal(left, right):
    """Compare optimizer tensors and nested metadata without numerical rounding."""
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right
