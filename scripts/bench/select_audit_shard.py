#!/usr/bin/env python3
"""Partition the complete, ordered trust-audit target list into nonempty shards."""
# Support both module invocation and direct script execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import re
import sys


def select(targets, count, index):
    if not re.fullmatch(r'[1-9][0-9]*', count) or not re.fullmatch(r'0|[1-9][0-9]*', index):
        raise ValueError('Set both AUDIT_TRUST_SHARD_COUNT (positive integer) and '
                         'AUDIT_TRUST_SHARD_INDEX (zero-based integer).')
    count, index = int(count), int(index)
    if not 0 <= index < count <= len(targets):
        raise ValueError('Shard index must be below shard count, and shard count '
                         'must not exceed the number of audit targets.')
    # Keep neighboring modules together so proof helpers and their readers
    # share a build cache. Interleaving every task across every worker forces
    # each isolated judge to copy most of the complete corpus.
    start = len(targets) * index // count
    stop = len(targets) * (index + 1) // count
    return targets[start:stop]


if __name__ == '__main__':
    try:
        print('\n'.join(select(sys.stdin.read().splitlines(), *sys.argv[1:])))
    except ValueError as error:
        print(error, file=sys.stderr)
        sys.exit(2)
