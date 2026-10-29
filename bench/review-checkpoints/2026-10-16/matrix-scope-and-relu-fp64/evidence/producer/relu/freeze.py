"""Freeze exact additive source, imported object and inherited audit bindings."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parent
repo = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
old = root.parent/'relu-original-wip/evidence/dtype-audit'
modules = ['bench/tritonbench_g/relu_triton_kernel/'+s+'.lean' for s in
           ['Proofs/FP64Source', 'Proofs/FP64Grid', 'Proofs/FP64Host', 'FP64Correct']]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
dump = lambda p, v: p.write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n')

def dependencies():
    queue = modules[:]
    project = {}
    external = {}
    while queue:
        rel = queue.pop()
        if rel in project or rel in external:
            continue
        local = root/rel
        p = local if local.is_file() else repo/rel
        if not p.is_file():
            obj = Path(rel).with_suffix('.olean')
            choices = [Path('/tmp/veritile-lean-toolchain/lib/lean')/obj]
            choices += [d/'.lake/build/lib/lean'/obj for d in (repo/'.lake/packages').iterdir()]
            found = next((q for q in choices if q.is_file()), None)
            assert found, rel
            external[rel] = {'olean':str(found.resolve()), 'sha256':sha(found)}
            continue
        row = {'source':str(p), 'sha256':sha(p), 'package':p == local}
        if p != local:
            obj = root/Path(rel).with_suffix('.olean')
            assert obj.is_file() and obj.is_symlink(), obj
            row.update(olean=str(obj.resolve()), olean_sha256=sha(obj))
        project[rel] = row
        for line in p.read_text().splitlines():
            if line.startswith('import '):
                for name in line[7:].split():
                    if name.startswith('--'):
                        break
                    queue.append(name.replace('.', '/')+'.lean')
    files = [repo/'lakefile.toml', repo/'lake-manifest.json', repo/'lean-toolchain',
             Path('/tmp/veritile-lean-toolchain/bin/lean'), Path('/tmp/veritile-lean-toolchain/bin/lake')]
    files += sorted((repo/'bench/tritonbench_g/relu_triton_kernel').rglob('*.lean'))
    files += [repo/'bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py']
    files += [old/n for n in ['receipt.json', 'frontend-profiles.json', 'float64.raw.ttir',
                             'extracted_relu_kernel.py', 'semantic-excerpts.txt']]
    calibration = root.parent/'acceptance-standard-calibration'
    files += [calibration/n for n in ['receipt.json', 'calibration.json', 'evidence.json']]
    return {'project':dict(sorted(project.items())), 'external_objects':dict(sorted(external.items())),
            'input_sha256':{str(p):sha(p) for p in files}}

def route_check():
    python = repo/'bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py'
    receipt = read(old/'receipt.json')
    assert sha(python) == receipt['source_sha256']
    tree = ast.parse(python.read_text())
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'relu_kernel')
    extracted = next(n for n in ast.parse((old/'extracted_relu_kernel.py').read_text()).body
                     if isinstance(n, ast.FunctionDef) and n.name == 'relu_kernel')
    assert ast.dump(kernel, include_attributes=False) == ast.dump(extracted, include_attributes=False)
    statements = [ast.unparse(n) for n in kernel.body]
    assert statements == [
        'pid = tl.program_id(0)', 'block_start = pid * block_size',
        'offsets = block_start + tl.arange(0, block_size)', 'mask = offsets < N',
        'x = tl.load(x_ptr + offsets, mask=mask)', 'result = tl.where(x >= 0, x, 0.0)',
        'if pid == 0:\n    tl.store(out_ptr + offsets, result, mask=mask)']
    ttir = (old/'float64.raw.ttir').read_text()
    ordered = ['tt.get_program_id', 'tt.make_range', 'arith.cmpi slt', 'tt.load',
               'arith.cmpf oge', 'arith.select', 'scf.if', 'arith.truncf', 'tt.store']
    locations = [ttir.index(token) for token in ordered]
    assert locations == sorted(locations)
    for token in ['!tt.ptr<f64>', '!tt.ptr<f32>', 'tensor<1024xf64>', 'to tensor<1024xf32>']:
        assert token in ttir
    profiles = read(old/'frontend-profiles.json')
    assert len(profiles) == 17
    success = [p['input'] for p in profiles if 'ttir' in p]
    rejected = {p['input']:p['error'] for p in profiles if 'error' in p}
    assert len(success) == 13 and len(rejected) == 4 and 'float64' in success
    assert all('unexpected type' in rejected[p] for p in ['float8_e4m3fn', 'float8_e5m2'])
    assert all('not supported in this architecture' in rejected[p]
               for p in ['float8_e4m3fnuz', 'float8_e5m2fnuz'])
    result = {'source':str(python), 'source_sha256':sha(python), 'kernel_ast_identical':True,
              'source_statements':statements, 'ttir_sha256':sha(old/'float64.raw.ttir'),
              'ttir_order_tokens':ordered, 'ttir_order_positions':locations,
              'source_route_kind':'explicit hand-materialized statement interpreter, reviewed against AST and frozen TTIR',
              'frontend_source':'inherited frozen audit; no rerun', 'admitted_profiles':success,
              'comparison_rejections':['float8_e4m3fn','float8_e5m2'],
              'signature_rejections':['float8_e4m3fnuz','float8_e5m2fnuz'],
              'observed_frontend_config':{'target':receipt['target'],'constexprs':receipt['constexprs'],
                  'options':receipt['options'],'environment':receipt['environment']},
              'new_native_compilation':False, 'new_gpu_execution':False}
    dump(root/'evidence/source-route-check.json', result)

def numerical_proposal():
    source = modules[0]
    config = {'dtype':'float64', 'output_dtype':'float32', 'block_size':1024,
              'N_domain':'0 <= N < 2^31', 'programs':'all complete ceil-grid permutations',
              'word_domain':'all 2^64 input patterns; arbitrary masked-load words',
              'native_target':'inherited CUDA sm90/Triton 3.7.1 domain only; other configurations require review'}
    ns = 'VeriTile.Bench.TritonBenchG.ReluTritonKernel.FP64Source.'
    rows = [
      {'id':'relu.fp64.ordered-ge-zero', 'statement':'result = tl.where(x >= 0, x, 0.0)',
       'native_operation':'arith.cmpf oge f64 against +0', 'port':ns+'orderedGeZero64',
       'binding':'FP64Source.step.whereGeZero64, argument s.regs.loaded i',
       'status':'proposed deferred numerical correspondence',
       'obligation':'Concrete sign/exponent/fraction comparator agrees with native ordered >=+0, including NaNs, infinities, subnormals and both zero signs.'},
      {'id':'relu.fp64.zero-literals', 'statement':'result = tl.where(x >= 0, x, 0.0)',
       'native_operation':'sitofp i32 zero to f64 and promoted f64 +0 literal',
       'port':ns+'select64', 'binding':'step.whereGeZero64 literal 0#64 and orderedGeZero64 zero comparison',
       'status':'proposed deferred exact numerical binding',
       'obligation':'Source int zero and float zero materialize as f64 positive zero. Select itself preserves the chosen complete f64 word.'},
      {'id':'relu.fp64.store-narrow', 'statement':'tl.store(out_ptr + offsets, result, mask=mask)',
       'native_operation':'arith.truncf f64 to f32 inside pid==0 gate',
       'port':ns+'Numerics.narrow64to32',
       'binding':'FP64Source.step.gatedStore32 -> scatter32 values, after f64 result',
       'status':'proposed deferred numerical callback',
       'obligation':'The supplied total word function equals the native f64→f32 conversion, including exact rounding, overflow, underflow/denormal, zero-sign and special-value behavior.'}]
    for row in rows:
        row.update(source=source, source_sha256=sha(root/source), configuration=config,
                   public_contracts=['VeriTile.Bench.TritonBenchG.ReluTritonKernel.FP64Correct.relu_fp64_source_grid_correctness',
                                     'VeriTile.Bench.TritonBenchG.ReluTritonKernel.FP64Correct.relu_fp64_host_source_correctness'])
    dump(root/'evidence/numerical-bindings-proposal.json',
         {'integrated_in_repository_queue':False, 'independent_registration_review':'pending root review',
          'native_numerical_validation':False, 'bindings':rows,
          'existing_twelve_routes':'unchanged; their existing numerical obligations are neither discharged nor widened by this proposal'})

def finalize():
    receipt = root/'evidence/receipt.json'
    assert not receipt.exists(), 'Refuse to overwrite frozen receipt'
    assert read(root/'evidence/dependencies-before.json') == read(root/'evidence/dependencies-after.json') == dependencies()
    history = read(root/'evidence/verification-attempts.json')
    results = read(root/'evidence/verification-results.json')
    assert [r['module'] for r in results] == modules
    assert history[-4:] == results
    for row in history:
        assert row['source_sha256_before'] == row['snapshot_sha256'] == sha(root/row['source_snapshot'])
        assert row['log_sha256'] == sha(root/row['log'])
    for row in results:
        assert row['exit_code'] == 0
        assert row['source_sha256_before'] == row['source_sha256_after'] == sha(root/row['module'])
        assert not re.search(r'\berror:|sorryAx|DISALLOWED|warning:', (root/row['log']).read_text())
    route_check()
    numerical_proposal()
    scope = {'status':'additive compiled partial extension; original-program acceptance pending',
      'acceptance_increment':0, 'successful_dtype_route_increment':1,
      'logical_success_routes_after':13, 'canonical_domain_count':17,
      'known_rejected_profiles':4, 'existing_twelve_routes_rewritten':False,
      'new_public_contracts':['FP64Correct.relu_fp64_source_grid_correctness', 'FP64Correct.relu_fp64_host_source_correctness'],
      'proved':['explicit f64 source statement execution and arbitrary masked words',
                'every pid pre-gate load/compare/select', 'pid0-only narrowing/store',
                'complete grid permutations with initial-memory formula and frame',
                'fixed1024 signed-i32 offsets/masks/gate on N<2^31',
                'finite host dispatch invokes unchanged twelve routes plus new fp64 grid',
                'host success composed with HostWrapper.relu_success; four fp8 rejected launch branches stop before return'],
      'trusted_primitive_boundary':['total typed memory views with valid input addresses and fresh fp32 output region',
          'Torch device/empty_like/numel outcomes with their metadata contracts',
          'frozen-target frontend dtype admission/rejection category',
          'launch execution represented by complete sequential program permutations under separate input/output views'],
      'remaining_finite_bindings':['Tensor metadata/freshness and actual valid typed input/output view applicability',
          'Justified full source/configuration domain and reviewed frontend primitive classification',
          'Exact numerical queue integration and independent registration review',
          'Schema2 public stage/dtype/branch scope integration and independent substantive review'],
      'known_source_bug':'For N>1024 only first block is written; arbitrary initial output tail persists. No whole-tensor mathematical ReLU claim.',
      'memory_carriers':'Runtime product: old typed branch observes final.1.mem; fp64 observes final.2; each proves the unselected carrier unchanged.',
      'harness':'Existing generic host helper/module semantics preserved; no generic harness implementation proof obligation added.',
      'native_execution':False, 'repository_modified':False, 'git_commands_run':False}
    dump(root/'evidence/scope-review.json',scope)
    outputs = ['README.md','verify.py','freeze.py','evidence/source-route-check.json',
               'evidence/numerical-bindings-proposal.json','evidence/scope-review.json',
               'evidence/dependencies-before.json','evidence/dependencies-after.json',
               'evidence/verification-attempts.json','evidence/verification-results.json']
    dump(receipt, {'schema_version':1,'frozen':True,'acceptance_increment':0,
      'modules':{p:sha(root/p) for p in modules},
      'module_objects':{str(Path(p).with_suffix('.olean')):sha((root/p).with_suffix('.olean')) for p in modules},
      'artifacts':{p:sha(root/p) for p in outputs}, 'verification_attempt_count':len(history),
      'failed_attempts_preserved':sum(r['exit_code'] != 0 for r in history),
      'final_clean_attempts':results,
      'independent_review_status':'parent review of draft exists; frozen package review/integration pending'})
    print(json.dumps({'receipt':str(receipt),'receipt_sha256':sha(receipt),'final_modules':4,'attempts':len(history)}))

mode = sys.argv[1]
if mode in ['before','after']:
    current = dependencies()
    if mode == 'after':
        assert read(root/'evidence/dependencies-before.json') == current
    dump(root/('evidence/dependencies-'+mode+'.json'),current)
    print(json.dumps({'mode':mode,'project_sources':len(current['project']), 'external_objects':len(current['external_objects'])}))
elif mode == 'finalize':
    finalize()
else:
    raise SystemExit('Expected before, after or finalize')
