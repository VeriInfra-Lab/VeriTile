"""Independent frozen-byte, source-shape and finite partition review.

This does not execute Triton or claim Python/compiler semantic refinement.
"""
import ast
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
TMP = Path('/tmp/veritile-standardization-20261015')
PKG = TMP / 'vector-dtype-admission-wip'
DEST = TMP / 'vector-dtype-admission-independent-review.json'
AUDIT = PKG / 'evidence/source-audit'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_text())

def check(path, expected):
    assert path.is_file() and not path.is_symlink(), str(path)
    assert sha(path) == expected, str(path)
    return path

assert not DEST.exists(), 'refuse review overwrite'
r = load(check(PKG / 'evidence/receipt.json', '13237a74d459a1fd6967155bb167bf4418617d3bbf4a2ee1a199e5b10712cf46'))
s = load(check(PKG / 'evidence/scope-review.json', '7d735125f66f10fc09d788a8f1fa57536bd1e45dd70967590ea128a09db2f888'))
a = load(check(AUDIT / 'audit-result.json', 'c6ce9a938df490f6a6264cee87107cb1f771c646f2456b9bf01f10602a645a83'))
for p, h in r['artifacts'].items():
    check(PKG / p, h)
deps = load(PKG / 'evidence/attempt-03/dependencies-before.json')
assert deps == load(PKG / 'evidence/attempt-03/dependencies-after.json')
assert len(deps['files']) == 154
assert sum('olean_path' in v for v in deps['files'].values()) == 148
assert len(deps['external_direct_import_oleans']) == 35
for p, d in deps['files'].items():
    check(Path(d['source_path']), d['sha256'])
    assert Path(d['source_path']) == (PKG if d['source_kind'] == 'package' else ROOT) / p
    if 'olean_path' in d:
        check(Path(d['olean_path']), d['olean_sha256'])
for d in deps['external_direct_import_oleans'].values():
    check(Path(d['olean_path']), d['olean_sha256'])
for p, h in deps['toolchain_binary_sha256'].items():
    check(Path(p), h)
results = load(PKG / 'evidence/attempt-03/verification-results.json')
assert len(results) == 2 and [x['exit_code'] for x in results] == [0, 0]
logs = []
for row in results:
    assert row['source_sha256_before'] == row['source_sha256_after'] == r['modules'][row['module']]
    logs.append(check(PKG / row['log'], row['log_sha256']).read_text())
source = '\n'.join((PKG / p).read_text() for p in r['modules'])
log = '\n'.join(logs)
assert len(re.findall(r'^#axiomsClean ', source, re.M)) == log.count('axiom footprint ⊆ standard base ✓') == 17
assert len(re.findall(r'^#auditModuleAxioms$', source, re.M)) == 2
assert '#specNonCircular ' not in source and 'non-circular ✓' not in log
assert not re.search(r'\b(?:warning|error):|sorryAx|DISALLOWED', log)
assert not re.search(r'\b(?:sorry|admit)\b|^axiom\s', source, re.M)
for name in ('attempt-01', 'attempt-02'):
    p = PKG / 'evidence' / name
    failed = load(p / 'verification-results.json')
    assert len(failed) == 1 and failed[0]['exit_code'] == 1
    f = failed[0]
    check(p / 'DTypeAdmission.source.lean', f['source_sha256_before'])
    assert f['source_sha256_before'] == f['source_sha256_after']
    assert 'error:' in check(PKG / f['log'], f['log_sha256']).read_text()
    assert load(p / 'dependencies-before.json') == load(p / 'dependencies-after.json')

# Re-extract each exact source unit and AST from its source, without importing it.
trees = {}
texts = {}
for name, info in a['files'].items():
    p = check(Path(info['path']), info['sha256'])
    if p.suffix == '.py':
        texts[name] = p.read_text()
        trees[name] = ast.parse(texts[name])

def named(node, name):
    for child in node.body:
        label = getattr(child, 'name', None)
        if isinstance(child, ast.Assign) and len(child.targets) == 1 and isinstance(child.targets[0], ast.Name):
            label = child.targets[0].id
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            label = child.target.id
        if label == name:
            return child
    raise AssertionError(name)

units = {}
for u in a['source_units']:
    node = trees[u['source']]
    for part in u['qualified_name'].split('.'):
        node = named(node, part)
    assert (node.lineno, node.end_lineno) == (u['line_start'], u['line_end'])
    raw = '\n'.join(texts[u['source']].splitlines()[node.lineno-1:node.end_lineno]) + '\n'
    assert check(AUDIT / u['text_file'], u['text_sha256']).read_bytes() == raw.encode()
    assert hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() == u['ast_sha256']
    units[u['source'], u['qualified_name']] = node
assert len(units) == 28
assert len(a['checks']) == 24 and all(x['passed'] is True for x in a['checks'])

# Independently check the decisive pre-body ordering, unsupported scalar guard,
# source pointer recursion and frozen default supported set.
function = units['triton_codegen', 'CodeGenerator.visit_FunctionDef']
body = [ast.unparse(x) for x in function.body]
assert body.index('fn_ty = self.prototype.serialize(self.builder)') < body.index('self.visit_compound_statement(node.body)')
assert not units['original_python', 'add_kernel'].args.defaults
assert ast.unparse(units['triton_codegen', 'ASTFunction.serialize'].body[0]) == 'arg_types_ir = self.flatten_ir_types(builder, self.arg_types)'
assert ast.unparse(units['triton_core', 'pointer_type.to_ir'].body[0]) == 'return builder.get_ptr_ty(self.element_ty.to_ir(builder), self.address_space)'
guard = units['triton_core', 'dtype.to_ir'].body[0]
assert ast.unparse(guard.test) == "self.name.startswith('fp8')"
assert ast.unparse(guard.body[0].test) == "hasattr(builder, 'options') and self.name not in builder.options.supported_fp8_dtypes"
assert isinstance(guard.body[0].body[0], ast.Raise)
opts = units['triton_backend', 'CUDAOptions']
default = next(ast.literal_eval(x.value) for x in opts.body if isinstance(x, ast.AnnAssign) and ast.unparse(x.target) == 'supported_fp8_dtypes')
assert default == ('fp8e5', 'fp8e4b15')
assert a['target'] == s['target'] == {'backend':'cuda','arch':90,'block_size':1024,'supported_fp8_dtypes':['fp8e4b15','fp8e4nv','fp8e5']}

# A separate transcription of the frozen promotion branches independently
# recomputes every row of the actual Lean executable export and stored probes.
canonical = ast.literal_eval(units['triton_canonicalization', 'type_canonicalisation_dict'].value)
def backend(name):
    sig = canonical[name]
    if sig == 'u1': return 'int1'
    if sig.startswith('i'): return 'int' + sig[1:]
    if sig.startswith('u'): return 'uint' + sig[1:]
    return sig

def promote(x, y):
    for d in ('fp64', 'fp32', 'fp16'):
        if d in (x, y): return d
    if x == y == 'bf16': return 'bf16'
    if 'bf16' in (x, y): return 'fp32'
    if x.startswith('fp8') and y.startswith('fp8'): return x if x == y else 'fp16'
    if x.startswith('fp8') or y.startswith('fp8'): return None
    def parts(d):
        assert re.fullmatch(r'(?:u?int)(?:1|8|16|32|64)', d)
        return d != 'int1' and d.startswith('int'), int(re.search(r'\d+', d)[0])
    sx, wx = parts(x); sy, wy = parts(y)
    if sx == sy: return x if wx > wy else y
    if not sx: return x if wx >= wy else y
    return y if wy >= wx else x

model = {}
for line in (PKG / 'evidence/model-rows.tsv').read_text().splitlines():
    x, y, pr, stage, detail, store = line.split('|')
    assert (x, y) not in model
    model[x, y] = {'promotion_result':None if pr == 'none' else pr,'stage':stage,'detail':detail,'store':store}
assert len(model) == 289
rows = load(check(AUDIT / 'rows.json', a['rows_sha256']))
assert len(rows) == 289
stages = Counter(); cross = Counter()
seen = set()
for row in rows:
    x, y = backend(row['x']), backend(row['y'])
    assert (x, y) not in seen
    seen.add((x, y))
    p = promote(x, y)
    assert model[x, y]['promotion_result'] == p == row['promotion_result']
    assert row['finite_lean_model'] == model[x, y]
    bad = next(((arg, d) for arg, d in [('x', x), ('y', y), ('output', x), ('nElements', 'int32')] if d in {'fp8e4b8', 'fp8e5b16'}), None)
    stage = 'signature' if bad else 'addition' if p is None else 'admitted'
    assert model[x, y]['stage'] == stage
    assert (row['observed_frontend_status'] == 'frontend_generated') == (stage == 'admitted')
    if bad:
        assert model[x, y]['detail'].endswith('.' + bad[0]) and model[x, y]['store'] == bad[1]
        assert 'ValueError' in row['error'] and 'not supported in this architecture' in row['error']
    elif stage == 'addition':
        assert 'TypeError' in row['error'] and 'unexpected type' in row['error']
    else:
        assert model[x, y]['detail'] == p and model[x, y]['store'] == ('int8' if x == 'int1' else x)
    assert row['new_value_execution_profile'] is False and row['source_native_refinement_proved'] is False
    stages[stage] += 1
    cross[stage + ':' + ('promoted' if p is not None else 'promotion_error')] += 1
assert dict(stages) == a['stage_counts'] == {'admitted':189,'addition':36,'signature':64}
assert dict(cross) == a['stage_promotion_cross_counts'] == {'admitted:promoted':189,'addition:promotion_error':36,'signature:promotion_error':36,'signature:promoted':28}
assert seen == set(model)

sys.path.insert(0, str(ROOT / 'bench'))
from adapter_common import declarations
parsed = declarations(PKG / 'bench/tritonbench_g/vector_addition/DTypeAdmissionCorrect.lean')
assert len(parsed) == len(r['public_contracts']) == 4 and r['public_contracts'] == s['public_contracts']
for contract in r['public_contracts']:
    assert parsed[contract['parser_name']]['statement_sha256'] == contract['statement_sha256']
    assert contract['model_kind'] == contract['role'] == 'protocol_contract'
    assert contract['covers_stages'] == [] and contract['scope_relation'] == 'not_an_execution_contract'
    assert contract['unclosed_numeric_callbacks'] == [] and contract['source_semantics_refinement_proved'] is False
assert all(g['deferrable_as_numerical_assumption'] is False for g in s['structural_remaining'])
assert s['numerical_assumptions_added'] == [] and not r['whole_original_program_accepted']
assert r['original_program_completion_increment'] == r['coverage_summary']['new_execution_contracts'] == r['coverage_summary']['new_value_dtype_profiles'] == 0
assert a['formal_python_to_model_refinement'] is False and a['observations_are_proof_premises'] is False
for d in a['pinned_inherited_value_inventory_anchors'].values():
    check(Path(d['path']), d['sha256'])
export = load(PKG / 'evidence/model-rows-result.json')
assert export['exit_code'] == 0 and export['row_count'] == 289
check(PKG / 'evidence/emit-model-rows.lean', export['source_sha256'])
check(PKG / 'evidence/model-rows.tsv', export['output_sha256'])
check(PKG / 'evidence/model-rows.stderr', export['stderr_sha256'])
execution = load(PKG / 'evidence/source-audit-execution.json')
assert execution['exit_code'] == 0
check(PKG / 'evidence/source-audit-final.log', execution['log_sha256'])
check(AUDIT / 'audit-result.json', execution['audit_sha256'])
w = load(PKG / 'evidence/whitespace-check.json')
assert w['exit_code'] == 0 and w['stdout'] == w['stderr'] == ''
temp = Path(w['temporary_index_root'])
staged = subprocess.check_output(['git','ls-files','-z'], cwd=temp).decode().strip('\0').split('\0')
assert len(staged) == w['staged_files'] == 69
for name in staged:
    assert subprocess.check_output(['git','show',':' + name], cwd=temp) == (PKG / name).read_bytes()
assert (AUDIT / '.gitattributes').read_text() == '/original_python__test_add.txt whitespace=-blank-at-eol\n'
assert subprocess.run(['git','diff','--cached','--check'], cwd=temp, capture_output=True).returncode == 0
assert load(PKG / 'evidence/failed-whitespace-check.json')['exit_code'] != 0

review = {
    'schema':1,'status':'accepted_partial_finite_dtype_admission_protocol_model','reviewer':'root',
    'reviewed_at_utc':datetime.now(timezone.utc).isoformat(),
    'receipt_sha256':sha(PKG / 'evidence/receipt.json'), 'scope_review_sha256':sha(PKG / 'evidence/scope-review.json'),
    'source_audit_sha256':sha(AUDIT / 'audit-result.json'), 'reviewed_source_sha256':r['modules'],
    'checker_sha256':sha(Path(__file__)), 'proof_defects_found':[],
    'whole_original_program_accepted':False, 'original_program_completion_increment':0,
    'execution_contract_increment':0, 'protocol_contract_increment':4, 'value_profile_increment':0,
    'verification':{'actual_producer_lean_exits':[0,0],'independent_lean_rerun':False,
        'named_axiom_checks':17,'noncircular_checks':0,'module_axiom_audits':2,'warnings':0,
        'frozen_artifacts_checked':len(r['artifacts']),'source_and_anchor_fingerprints':154,
        'project_oleans_checked':148,'direct_external_oleans_checked':35,'preserved_failed_compilations':2,
        'exact_python_source_units_reextracted':28,'producer_source_checks_inspected':24,
        'finite_rows_independently_recomputed':289,'staged_blobs_rechecked':69,
        'new_frontend_runs':0,'native_backend_runs':0,'gpu_runs':0},
    'semantic_review':[
        'Both Lean sources and producer verification, source audit and finalizer read. Finite case proofs have independent exhaustiveness, disjointness and first-error statements. No new axiom or numerical callback is introduced.',
        'Signature serialization precedes the original no-default kernel body. Pointer element to_ir checks the frozen default sm90 supported fp8 set. The first fnuz rejection is x, otherwise y; output=x and n_elements=i32 add no earlier rejection.',
        'All 289 actual Lean-exported rows independently recomputed from the inspected promotion rules and compared with retained observations. Partition:189 admitted,64 signature failures,36 addition failures. Standalone promotion217/72 is a different partition;36 promotion failures are preempted by signature failure.',
        'All four public statements describe a finite protocol model. They do not prove Python/AST semantics, JIT/cache/host reachability, native execution, exception wrapping or GPU memory behavior. They add no executed value profile. All100 rejection cases, including28 promotion-success fnuz cases, stay separate from value coverage.',
        'The narrowly anchored whitespace attribute preserves the original test_add source bytes. Both failed compiles and the failed pre-exception whitespace check remain recorded.'],
    'stage_counts':dict(stages),'stage_promotion_cross_counts':dict(cross),
    'remaining_structural_obligations':s['structural_remaining']
}
DEST.write_text(json.dumps(review, indent=2) + '\n')
print(json.dumps({'review':str(DEST),'sha256':sha(DEST),'status':review['status']}, indent=2))
