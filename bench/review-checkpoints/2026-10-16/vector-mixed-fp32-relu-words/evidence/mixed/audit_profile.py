"""Bind the 18 mixed fp32 materializations to already-frozen exact-source IR."""
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
old = root.parent / 'vector-addition-dtype-extension-wip'
audit = old / 'evidence/extension-audit'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(old / 'evidence/receipt.json') == '85dc08106852c4a3f69516046182954693eb5afd0158064db358c708a70b9100'
source = json.loads((audit / 'frontend-probes.json').read_text())
ints = {'bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'}
output = root / 'evidence/source-profile'
output.mkdir(exist_ok=True)
rows = []
for row in source:
    x, y = row['x'], row['y']
    if not (x in ints and y == 'float32' or y in ints and x == 'float32'):
        continue
    assert row['status'] == 'frontend_generated'
    raw = audit / row['ttir_path']
    assert sha(raw) == row['ttir_sha256']
    ir = raw.read_text()
    integral = x if x in ints else y
    width = 1 if integral == 'bool' else int(re.search(r'\d+', integral).group())
    to_float = 'uitofp' if integral == 'bool' or integral.startswith('uint') else 'sitofp'
    assert re.search(r'arith\.' + to_float + r'.*tensor<1024xi' + str(width) + r'> to tensor<1024xf32>', ir)
    assert re.search(r'arith\.addf.*tensor<1024xf32>', ir)
    if x in ints:
        store_width = 8 if x == 'bool' else width
        from_float = 'fptoui' if x.startswith('uint') else 'fptosi'
        assert re.search(r'arith\.' + from_float + r'.*tensor<1024xf32> to tensor<1024xi' + str(store_width) + r'>', ir)
        assert re.search(r'tt\.store.*tt.ptr<i' + str(store_width) + r'>', ir)
    else:
        store_width, from_float = 32, None
        assert re.search(r'tt\.store.*tt.ptr<f32>', ir)
    if integral == 'bool':
        assert 'arith.cmpi ne' in ir and 'tt.ptr<i8>' in ir
    (output / raw.name).write_bytes(raw.read_bytes())
    rows.append({'x': x, 'y': y, 'integer_first': x in ints,
                 'integer_to_fp32': to_float, 'source_integer_width': width,
                 'fp32_to_integer_store': from_float, 'store_width': store_width,
                 'ttir_path': raw.name, 'ttir_sha256': sha(raw),
                 'status': 'exact_source_frontend_route_checked_not_native_refinement'})
assert len(rows) == 18
receipt = {'profiles': rows, 'count': len(rows),
           'frozen_integer_package_receipt_sha256': sha(old / 'evidence/receipt.json'),
           'frozen_frontend_receipt_sha256': sha(audit / 'receipt.json'),
           'frozen_frontend_profiles_sha256': sha(audit / 'frontend-probes.json'),
           'source_sha256': json.loads((audit / 'receipt.json').read_text())['source_sha256'],
           'new_frontend_runs': 0, 'native_backend_runs': 0, 'gpu_runs': 0,
           'options': 'inherited original sm90/debug=false/sanitize_overflow=true profile',
           'limitations': ['observed frontend conversion graph only',
                           'no native data/address overflow, poison, debug or admission theorem']}
p = output / 'receipt.json'
p.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'count': len(rows), 'receipt_sha256': sha(p)}))
