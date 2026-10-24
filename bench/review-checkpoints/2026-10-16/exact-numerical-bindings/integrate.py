#!/usr/bin/env python3
"""Verify the frozen adapter package; copy it only with explicit --apply."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

CHECKPOINT = Path('bench/review-checkpoints/2026-10-16/exact-numerical-bindings')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Non-relative package path: ' + str(relative))
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError('Escaping package path: ' + str(relative))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    package, root = args.package.resolve(), args.root.resolve()
    checkpoint = inside(package, CHECKPOINT)
    receipt_path = checkpoint / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    for relative, expected in receipt['artifacts'].items():
        path = inside(package, relative)
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Changed frozen artifact: ' + relative)
    manifest = json.loads((checkpoint / 'integration-manifest.json').read_text())
    replacement = {row['destination']: row for row in manifest['production_files']}
    sources = json.loads((checkpoint / 'evidence/source-hashes.json').read_text())
    for relative, expected in sources['baseline_repository_files'].items():
        destination = inside(root, relative)
        allowed = {expected}
        if relative in replacement:
            allowed.add(sha(inside(package, replacement[relative]['source'])))
        if not destination.is_file() or sha(destination) not in allowed:
            raise ValueError('Changed baseline input/helper/test: ' + relative)
    copies = list(manifest['production_files'])
    for source in sorted(checkpoint.rglob('*')):
        if not source.is_file():
            continue
        relative = str(source.relative_to(package))
        if source != receipt_path and relative not in receipt['artifacts']:
            raise ValueError('Unrecorded checkpoint file: ' + relative)
        copies.append({'source': relative, 'destination': relative, 'before_sha256': None})
    for row in copies:
        source = inside(package, row['source'])
        destination = inside(root, row['destination'])
        allowed = {sha(source)}
        if row['before_sha256'] is not None:
            allowed.add(row['before_sha256'])
        if destination.exists() and (not destination.is_file() or sha(destination) not in allowed):
            raise ValueError('Unexpected destination preimage: ' + row['destination'])
        if not destination.exists() and row['before_sha256'] is not None:
            raise ValueError('Missing destination preimage: ' + row['destination'])
    if args.apply:
        for row in copies:
            source = inside(package, row['source'])
            destination = inside(root, row['destination'])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    print(json.dumps({'status': 'copied' if args.apply else 'ready',
                      'receipt_sha256': sha(receipt_path),
                      'production_files': len(manifest['production_files']),
                      'checkpoint_files': len(copies) - len(manifest['production_files']),
                      'baseline_files_verified': len(sources['baseline_repository_files']),
                      'native_executions': 0, 'original_program_completion_increment': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
