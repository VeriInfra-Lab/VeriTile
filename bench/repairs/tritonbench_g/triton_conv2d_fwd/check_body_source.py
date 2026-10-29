"""Check all four literal bodies and their existing public invocation aliases.

This source check neither imports Triton/Torch nor executes a kernel. Lean's
execution, safety and public IO proofs are separate required checks.
"""
from pathlib import Path
import json
import re
from bench.jit_correspondence import check_manifest, quotation

ROOT = Path(__file__).resolve().parents[4]
BASE = Path('bench/repairs/tritonbench_g/triton_conv2d_fwd')


def check_aliases(text: str, definition: str, half: bool, candidate: bool) -> None:
    signature, _ = quotation(text, definition)
    entries = re.findall(r'\blet\s+(\w+)\s*:\s*(\w+)\s*:=\s*([^\n]+)', signature)
    actual = {name: (ty, ' '.join(value.split())) for name, ty, value in entries}
    expected = {name: ('Nat', 'p.' + name) for name in (
        'kernel_height', 'kernel_width', 'stride_height', 'stride_width', 'groups',
        'BLOCK_SIZE_BATCH_HEIGHT_WIDTH', 'BLOCK_SIZE_IN_FEAT', 'BLOCK_SIZE_OUT_FEAT')}
    expected.update({name: ('Int', '(p.' + name + ' : Int)')
                     for name in ('padding_height', 'padding_width')})
    expected.update({name: ('RegionName', 'p.' + field) for name, field in (
        ('input_pointer', 'Input'), ('weight_pointer', 'Weight'), ('output_pointer', 'Output'))})
    expected.update(fp16=('Bool', 'Bool.true' if half else 'Bool.false'), tf32=('Bool', 'p.tf32'))
    if candidate:
        expected['BLOCK_SIZE_BATCH_HEIGHT_WIDTH'] = ('Nat', '128')
    if len(actual) != len(entries) or actual != expected:
        raise ValueError('source invocation aliases differ from the reviewed public parameters')


def check_sources(root: Path = ROOT) -> dict:
    manifest = root / BASE / 'jit-correspondence.json'
    rows = json.loads(manifest.read_text())
    if len(rows) != 4:
        raise ValueError('both source variants and both operand-conversion branches are required')
    results = check_manifest(root, manifest)
    for row in rows:
        check_aliases((root / row['lean']).read_text(), row['lean_definition'],
                      row['expected_specialized_booleans']['fp16'],
                      row['lean'].endswith('/CandidateKernels.lean'))
    return {'status': 'PASS', 'bodies': results,
            'scope': 'Exact Python statements and declared invocation aliases; not a compiler or hardware certificate.'}


if __name__ == '__main__':
    print(json.dumps(check_sources(), indent=2))
