#!/usr/bin/env python3
"""Restore hash-pinned historical audit inputs from the fixed fp-examples archive.

Current Lean/Python sources and current review manifests are tracked normally.
This only materializes old evidence; it never changes a receipt or approves it
for the current source version. The ordinary audit must still validate it.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'bench/main-integration/historical-inputs.json.gz'
REPOSITORY = 'VeriInfra-Lab/VeriTile'
COMMIT = 'd67b1d0444c2e0fb8066414c2041129b53bf2931'


def destination(root, name):
    path = PurePosixPath(name)
    if (path.is_absolute() or '..' in path.parts or str(path) != name
            or path.parts[:2] != ('bench', 'review-checkpoints')):
        raise ValueError(f'Invalid evidence path: {name}')
    result = root / name
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Evidence path escapes root: {name}')
    return result


def valid(path, entry):
    if not path.is_file() or path.stat().st_size != entry['bytes']:
        return False
    digest = hashlib.sha1(f"blob {entry['bytes']}\0".encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == entry['git_blob']


def read_manifest(path=MANIFEST):
    data = json.loads(gzip.decompress(path.read_bytes()))
    if (data.get('schema') != 1 or data.get('repository') != REPOSITORY
            or data.get('archive_commit') != COMMIT):
        raise ValueError('Wrong evidence archive identity')
    for name, entry in data['files'].items():
        destination(ROOT, name)
        if (not isinstance(entry['bytes'], int) or entry['bytes'] < 0
                or not re.fullmatch('[0-9a-f]{40}', entry['git_blob'])):
            raise ValueError(f'Invalid evidence binding: {name}')
    return data['files']


def missing_inputs(root, entries):
    missing = {}
    for name, entry in entries.items():
        path = destination(root, name)
        if path.exists() or path.is_symlink():
            if not valid(path, entry):
                raise ValueError(f'Existing evidence differs; refusing to overwrite: {name}')
        else:
            missing[name] = entry
    return missing


def stage_tar(stream, stage, entries):
    seen = set()
    prefix = f'VeriTile-{COMMIT}/'
    with tarfile.open(fileobj=stream, mode='r|gz') as archive:
        for member in archive:
            if not member.name.startswith(prefix):
                if member.name.rstrip('/') == prefix.rstrip('/'):
                    continue
                raise ValueError('Unexpected archive root')
            name = member.name[len(prefix):]
            # No tar paths are passed to extract/extractall. Only exact manifest
            # members can be written, and links are never followed.
            if name not in entries:
                continue
            if name in seen or not member.isfile() or member.size != entries[name]['bytes']:
                raise ValueError(f'Invalid or duplicate archive member: {name}')
            path = destination(stage, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, path.open('wb') as output:
                shutil.copyfileobj(source, output)
            if not valid(path, entries[name]):
                raise ValueError(f'Archived Git blob differs: {name}')
            seen.add(name)
    if seen != entries.keys():
        raise ValueError(f'Archive omitted {len(entries.keys() - seen)} required files')


def stage_git(repository, stage, entries):
    # Each object request names the pinned archive commit and exact path. A
    # different checkout/branch in the supplied object database cannot affect it.
    process = subprocess.Popen(['git', '-C', str(repository), 'cat-file', '--batch'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for name, entry in entries.items():
            process.stdin.write(f'{COMMIT}:{name}\n'.encode())
            process.stdin.flush()
            header = process.stdout.readline().decode().strip().split()
            if header != [entry['git_blob'], 'blob', str(entry['bytes'])]:
                raise ValueError(f'Archive object unavailable or changed: {name}')
            path = destination(stage, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('wb') as output:
                remaining = entry['bytes']
                while remaining:
                    block = process.stdout.read(min(remaining, 1024 * 1024))
                    if not block:
                        raise ValueError('Truncated git object stream')
                    output.write(block)
                    remaining -= len(block)
            if process.stdout.read(1) != b'\n' or not valid(path, entry):
                raise ValueError(f'Archived Git blob differs: {name}')
        process.stdin.close()
        if process.wait():
            raise ValueError('git cat-file failed')
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='verify local inputs without downloading')
    mode.add_argument('--download', action='store_true', help='download the fixed GitHub commit archive')
    mode.add_argument('--archive', type=Path, help='use a local copy of the GitHub commit tar.gz')
    mode.add_argument('--git-repository', type=Path, help='use a local Git database containing the archive commit')
    args = parser.parse_args()
    entries = read_manifest()
    missing = missing_inputs(ROOT, entries)
    if missing:
        if not (args.download or args.archive or args.git_repository):
            raise SystemExit(f'{len(missing)} historical inputs missing; run this script with --download, --archive, or --git-repository.')
        # Validate the complete batch before publishing any recovered files.
        with tempfile.TemporaryDirectory(prefix='veritile-evidence-') as temporary:
            stage = Path(temporary)
            if args.git_repository:
                stage_git(args.git_repository, stage, missing)
            elif args.archive:
                with args.archive.open('rb') as stream:
                    stage_tar(stream, stage, missing)
            else:
                url = f'https://codeload.github.com/{REPOSITORY}/tar.gz/{COMMIT}'
                with urllib.request.urlopen(url, timeout=120) as stream:
                    stage_tar(stream, stage, missing)
            for name, entry in missing.items():
                dest = destination(ROOT, name)
                if dest.exists() or dest.is_symlink():
                    raise ValueError(f'Evidence appeared during restore: {name}')
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(stage / name, dest)
    print(f'Verified {len(entries)} archived inputs at {COMMIT}; restored {len(missing)}. No current proof result inferred.')


if __name__ == '__main__':
    main()
