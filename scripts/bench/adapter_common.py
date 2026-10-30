"""Small, fail-closed primitives shared by the source and build adapters."""
from __future__ import annotations
# Support both module invocation and direct script execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tomllib
import zlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from scripts.bench.lean_import_scan import lean_code_mask
from scripts.spec_sheet import split_decls, split_statement

BASE = 'bench/tritonbench_g/'
LIBRARIES = ('VeriTile', 'VeriTileMath', 'VeriTileFull', 'TritonBenchGVerified',
             'TritonBenchGInProgress', 'TritonBenchGLegacy',
             'TritonBenchSpecExamples', 'FPFrontendRegressions')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False) + '\n'
    if path.name.endswith('.json.gz'):
        # Pin timestamp and omit the filename so output is reproducible.
        with path.open('wb') as stream:
            with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
                compressed.write(text.encode('utf-8'))
    else:
        path.write_text(text)

def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    if path.name.endswith('.json.gz'):
        try:
            # Read to EOF to verify the gzip trailer before accepting the JSON.
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                text = stream.read()
        except (gzip.BadGzipFile, EOFError, zlib.error) as error:
            raise ValueError('invalid gzip JSON: ' + str(path)) from error
    else:
        text = path.read_text()
    return json.loads(text, object_pairs_hook=unique)

def relative(name):
    if (not isinstance(name, str) or not name or '\\' in name or '\0' in name
            or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
            or str(PurePosixPath(name)) != name or name == '.'):
        raise ValueError('invalid relative path: ' + repr(name))
    return Path(name)

def within(root, name):
    p = root / relative(name)
    if not p.resolve().is_relative_to(root.resolve()) or not p.is_file():
        raise ValueError('missing or escaping file: ' + name)
    return p

def checked(root, name, expected):
    if not isinstance(expected, str) or re.fullmatch('[0-9a-f]{64}', expected) is None:
        raise ValueError('missing/invalid hash: ' + name)
    p = within(root, name)
    if hash_file(p) != expected:
        raise ValueError('hash mismatch: ' + name)
    return p

def unique_rows(rows, key):
    result = {}
    for row in rows:
        k = row[key]
        if k in result:
            raise ValueError('duplicate ' + key + ': ' + str(k))
        result[k] = row
    return result

def module_name(path):
    p = relative(path)
    if p.suffix != '.lean':
        raise ValueError('not a Lean source: ' + path)
    name = '.'.join(p.with_suffix('').parts)
    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*', name) is None:
        raise ValueError('unsupported module name: ' + name)
    return name

def imports(text):
    """Actual project import syntax; comments, strings and quotations are blanked.

    Multiline imports are supported until the next command. Unknown import
    tokens are rejected instead of silently omitting a dependency.
    """
    code = lean_code_mask(text)
    result = []
    lines = code.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r'^\s*(?:public\s+)?(?:meta\s+)?import\s+(.*)', line)
        if not m:
            continue
        body = m[1].strip()
        j = i + 1
        while j < len(lines) and re.fullmatch(r'\s+[A-Z][A-Za-z0-9_.]*(?:\s+[A-Z][A-Za-z0-9_.]*)*\s*', lines[j]):
            body += ' ' + lines[j].strip()
            j += 1
        for token in body.split():
            if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*', token) is None:
                raise ValueError('unsupported import syntax: ' + token)
            result.append(token)
    if len(result) != len(set(result)):
        raise ValueError('duplicate imports')
    return sorted(result)

def declarations(path):
    return {d['name']: {**d, 'statement_sha256': sha(split_statement(d['text'])[0].encode())}
            for d in split_decls(path.read_text())}

def registered(root):
    """Expand exactly the eight existing libraries, without changing Lake."""
    config = tomllib.loads(within(root, 'lakefile.toml').read_text())
    libs = {lib['name']: lib for lib in config['lean_lib']}
    if not set(LIBRARIES) <= set(libs):
        raise ValueError('required library omitted')
    result = {}
    for name in LIBRARIES:
        lib = libs[name]
        if lib.get('srcDir', '.') != '.':
            raise ValueError('unsupported library srcDir: ' + name)
        patterns = [*lib.get('roots', []), *lib.get('globs', [])]
        if not patterns:
            patterns = [name]
        modules = set()
        for pattern in patterns:
            if pattern.endswith('.+') or pattern.endswith('.*'):
                directory = root / Path(*pattern[:-2].split('.'))
                files = directory.rglob('*.lean') if pattern.endswith('.+') else directory.glob('*.lean')
                modules.update(module_name(p.relative_to(root).as_posix()) for p in files)
            elif '+' in pattern or '*' in pattern:
                raise ValueError('unsupported Lake glob: ' + pattern)
            else:
                source = Path(*pattern.split('.')).with_suffix('.lean').as_posix()
                within(root, source)
                modules.add(pattern)
        for module in modules:
            result.setdefault(module, []).append(name)
    return {m: sorted(libs) for m, libs in sorted(result.items())}

def task_directories(root):
    return sorted(p.name for p in (root / BASE).iterdir()
                  if p.is_dir() and not p.name.startswith(('.', '_')))

def task_files(root):
    """No convention excludes extra Lean files from the explicit inventory."""
    return sorted(p.relative_to(root).as_posix() for name in task_directories(root)
                  for p in (root / BASE / name).rglob('*.lean'))
