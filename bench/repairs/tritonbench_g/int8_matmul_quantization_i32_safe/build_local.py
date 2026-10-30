"""Build only this candidate; inherited object directories are read-only inputs."""
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).resolve().parent
OUT = ROOT / '.veritile-cache/int8-matmul-quantization-i32-safe-build/lib/lean'
INHERITED = ROOT / '.veritile-cache/kernel-build/lib/lean'
PACKAGES = Path('/tmp/veritile-tritonbench-lake/packages')
LIBS = [str(PACKAGES / p / '.lake/build/lib/lean') for p in
    ['Cli', 'batteries', 'Qq', 'aesop', 'proofwidgets', 'importGraph', 'LeanSearchClient', 'plausible', 'mathlib']]
LIBS += ['/tmp/veritile-lean-toolchain/lib/lean', str(OUT), str(INHERITED), str(ROOT / '.lake/build/lib/lean')]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def imports_of(source):
    records = []
    for module in re.findall(r'^import\s+([A-Za-z0-9_.]+)', source.read_text(), re.M):
        rel = Path(*module.split('.'))
        obj = next((Path(lib) / rel.with_suffix('.olean') for lib in LIBS
                    if (Path(lib) / rel.with_suffix('.olean')).is_file()), None)
        assert obj is not None, module
        record = dict(module=module, object=str(obj.resolve()), object_sha256=sha(obj))
        src = ROOT / rel.with_suffix('.lean')
        if src.is_file():
            record.update(source=str(src.relative_to(ROOT)), source_sha256=sha(src))
        records.append(record)
    return records

def mirror_inputs(base, output):
    """Lean selects a namespace root, so expose siblings through private links."""
    output.mkdir(parents=True, exist_ok=True)
    for entry in base.iterdir():
        target = output / entry.name
        if not os.path.lexists(target):
            target.symlink_to(entry.absolute(), target_is_directory=entry.is_dir())
        elif target.is_dir() and not target.is_symlink() and entry.is_dir():
            mirror_inputs(entry, target)

def main():
    env = dict(os.environ, LEAN_PATH=':'.join(LIBS), LEAN_NUM_THREADS='2')
    # All links are created under OUT; inherited targets are never written.
    (OUT / SOURCE.relative_to(ROOT)).mkdir(parents=True, exist_ok=True)
    mirror_inputs(INHERITED, OUT)
    for name in sys.argv[1:]:
        source = SOURCE / name
        assert source.resolve().is_relative_to(SOURCE) and source.suffix == '.lean'
        target = OUT / source.relative_to(ROOT).with_suffix('.olean')
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.is_symlink()
        original = sha(source)
        imports = imports_of(source)
        cmd = ['/tmp/veritile-lean-toolchain/bin/lean', '-s', '32768', '--root=' + str(ROOT),
               str(source), '-o', str(target), '-i', str(target.with_suffix('.ilean'))]
        logdir = ROOT / '.veritile-cache/kernel-evidence/int8_matmul_quantization_i32_safe/build-logs'
        logdir.mkdir(parents=True, exist_ok=True)
        log = logdir / (source.stem + '-' + original[:16] + '.log')
        with log.open('a') as stream:
            stream.write(json.dumps(dict(command=cmd, lean_path=LIBS)) + '\n')
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            for number, line in enumerate(process.stdout):
                stream.write(line)
                if number < 250:
                    print(line, end='', flush=True)
            code = process.wait()
            stream.write(json.dumps(dict(exit_code=code)) + '\n')
        print(json.dumps(dict(log=str(log.relative_to(ROOT)), exit_code=code)), flush=True)
        if code:
            raise SystemExit(code)
        assert sha(source) == original
        assert imports_of(source) == imports
        objects = {str(p.relative_to(ROOT)): sha(p) for p in target.parent.glob(target.stem + '.*')
                   if p.is_file() and not p.name.endswith('.workspace-build.json')}
        record = dict(source=str(source.relative_to(ROOT)), source_sha256=original, exit_code=0,
                      objects=objects, lean_path=LIBS, command=cmd,
                      direct_imports=imports,
                      log=dict(path=str(log.relative_to(ROOT)),sha256=sha(log)))
        target.with_suffix('.workspace-build.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(module=record['source'], exit_code=0)), flush=True)

if __name__ == '__main__':
    main()
