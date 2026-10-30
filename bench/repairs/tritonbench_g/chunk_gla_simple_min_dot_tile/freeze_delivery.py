"""Freeze the complete candidate after its candidate-only build and saved checks."""
import hashlib
import json
import re
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=ROOT/'.veritile-cache/task-deliveries/chunk_gla_simple_min_dot_tile'
BUILD=ROOT/'.veritile-cache/kernel-build/lib/lean'
BASE=ROOT/'.lake/build/lib/lean'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):
    p=Path(p).absolute()
    return {'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'sha256':digest(p)}
def imported(p):
    return [name for line in p.read_text().splitlines() if line.startswith('import ')
            for name in line.removeprefix('import ').split('--')[0].split()]

entry=json.loads((OUT/'entry.json').read_text())
assert all(digest(ROOT/p)==h for p,h in entry['predecessor_files'].items())
assert digest(ROOT/entry['original_defect_packet']['path'])==entry['original_defect_packet']['sha256']
source=HERE/'chunk_gla_simple_min_dot_tile.py'
assert digest(source)==entry['candidate_source']['sha256']
proofs=sorted(HERE.rglob('*.lean'))
receipts=[]
for p in proofs:
    receipt=BUILD/p.relative_to(ROOT).with_suffix('.workspace-build.json')
    r=json.loads(receipt.read_text())
    assert r['source_sha256']==digest(p),(p,'candidate source receipt mismatch')
    assert r['exit_code']==0
    for name,sha in r['objects'].items():assert digest(name)==sha,(name,'candidate object changed')
    receipts.append(pin(receipt))

modules={};frontier=set()
queue=['bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Correct',
       'bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.FPEquiv']
while queue:
    name=queue.pop()
    if name in modules or name in frontier:continue
    src=ROOT/(name.replace('.','/')+'.lean')
    if not src.is_file():frontier.add(name);continue
    imports=imported(src)
    family=Path(name.replace('.','/'))
    obj=next((base/family.with_suffix('.olean') for base in (BUILD,BASE) if (base/family.with_suffix('.olean')).is_file()),None)
    assert obj is not None,('missing effective object',name)
    record={'source':pin(src),'direct_imports':imports,'effective_olean':pin(obj),'artifacts':[]}
    for suffix in ('.olean.private','.olean.server','.ilean','.workspace-build.json'):
        f=obj.with_suffix(suffix)
        if f.is_file():record['artifacts'].append(pin(f))
    receipt=obj.with_suffix('.workspace-build.json')
    if receipt.is_file():
        r=json.loads(receipt.read_text())
        record['current_source_matches_receipt']=r.get('source_sha256')==digest(src)
        assert record['current_source_matches_receipt'],('dependency source receipt mismatch',name)
    modules[name]=record
    queue.extend(imports)
closure={'modules':modules,'external_cached_frontier':sorted(frontier),
         'scope':'Current workspace source closure and first-existing object in the approved driver search order. Imported library/toolchain packages are the pinned build-driver frontier. This is not a cold rebuild of inherited dependencies.'}
(OUT/'dependencies.json').write_text(json.dumps(closure,indent=2)+'\n')

native=json.loads((OUT/'native-admission/result.json').read_text())
assert native['source']['sha256']==digest(source)
assert native['checker']['sha256']==digest(HERE/'check_kernel_admission.py')
assert all(c['status'] in ('compile_success','empty_return_no_launch','cancelled_not_completed','not_started_after_cancellation') for c in native['cases'])
compiler_refs=[Path('/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py'),
 Path('/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py'),
 Path('/usr/local/lib/python3.12/site-packages/triton/backends/nvidia/compiler.py'),
 Path('/usr/local/lib/python3.12/site-packages/triton/language/semantic.py'),
 Path('/usr/local/lib/python3.12/site-packages/triton/_utils.py'),
 Path('/usr/local/lib/python3.12/site-packages/nvidia/cu13/include/cccl/cuda/__device/arch_traits.h')]
resource=[]
for c in native['cases']:
    if c['status']=='compile_success':
        resource.append({'shape':{n:c['admitted'][n] for n in ('B','H','T','K','V','BT')},
                         'shared_bytes':c['metadata']['shared'],'sm80_max_shared_per_block':166912,
                         'fits_sm80_shared_memory':c['metadata']['shared']<=166912})
resources={'limits':resource,'evidence':[pin(p) for p in compiler_refs],
 'basis':'CUDA arch_traits sm80: 164*1024 per SM minus 1024 reserved per block = 166912. CompiledKernel._init_handles rejects metadata.shared greater than the device maximum.',
 'interpretation':'External device resource applicability; no new source guard, domain change, numerical TODO, physical GPU launch claim, or resource experiment.'}
(OUT/'resources.json').write_text(json.dumps(resources,indent=2)+'\n')

source_binding=json.loads((OUT/'source-binding.json').read_text())
assert source_binding['candidate_source']['sha256']==digest(source)
assert source_binding['readable_quote']['sha256']==digest(HERE/'Kernels.lean')
assert source_binding['checker']['sha256']==digest(HERE/'check_source.py')
assert source_binding['complete_AST_match']
contract=(OUT/'contract.log').read_text()
assert '68 rejects; 9 admits; 16 exact CPU controls' in contract
lowering=json.loads((OUT/'lowering.json').read_text())
assert lowering['source']['sha256']==digest(source)
assert lowering['checker']['sha256']==digest(HERE/'check_lowering.py')
artifacts=[]
for p in sorted(OUT.iterdir()):
    if p.is_file() and p.name!='delivery.json':artifacts.append(pin(p))
for c in native['cases']:
    for p in c.get('artifacts',[]):
        assert digest(ROOT/p['path'])==p['sha256']
        artifacts.append(p)
packet={
 'candidate':'chunk_gla_simple_min_dot_tile','status':'FROZEN_FOR_INDEPENDENT_REVIEW',
 'source':pin(source),'candidate_files':[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()],
 'source_delta':'Only BK=max(16,min(nextpow2(max(K,1)),64)); all original checked input domain and K0 retained.',
 'predecessor':entry['predecessor_source'],'original_defect_packet':entry['original_defect_packet'],
 'preservation':{'files_checked':len(entry['predecessor_files']),'all_preserved':True,'old_defect_packet_preserved':True},
 'public_endpoints':[
  'VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Correct.whole_kernel',
  'VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Correct.returned_tensor',
  'VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.FPEquiv.whole_kernel',
  'VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.FPEquiv.returned_tensor'],
 'scope':{'full_readable_source':True,'source_success_constructed':True,'host_success_premise':False,
   'independent_Real_target':True,'independent_scalar_native_target':True,'actual_pointer_Tensor_bytes_frame':True,
   'K0_retained':True,'empty_output_retained':True,'input_copies':False,'compatibility_layer':False,
   'predecessor_machine_equivalence_on_uncompiled_domain':False,'GPU_execution':False,
   'self_acceptance':False,'core_Lake_ledger_edits':False},
 'model_boundaries':[
  'Five fixed scalar primitive families: literal, binary, unary, cast, FMA. Raw word leaves are identity; no dot/reduction/output numeric port.',
  'Real arithmetic and native word graph are separate semantics; no universal native=Real or exceptional-K0-zero claim.',
  'Compiler realization of the inspected scalar FMA/exp2/carry graph and native ABI is an explicit boundary.',
  'Fresh output allocation, available device resources, GPU runtime/driver and scheduling delimit physical applicability.'],
 'numerical_status':{'outcome':'modeled fixed primitive boundary','numerical_TODOs':0,'primitive_experiments_added':0},
 'build':{'driver':pin(ROOT/'.veritile-cache/build-tools/workspace_lean_local_first_v2.py'),
   'candidate_modules':len(proofs),'candidate_receipts':receipts,'dependency_closure':pin(OUT/'dependencies.json'),
   'repository_dependency_modules':len(modules),'external_cached_frontier_count':len(frontier)},
 'validation':{'source_binding':pin(OUT/'source-binding.json'),'contract_controls':pin(OUT/'contract.log'),
  'native_admission':pin(OUT/'native-admission/result.json'),'lowering':pin(OUT/'lowering.json'),
  'resources':pin(OUT/'resources.json'),'compiler_cases':len(native['cases']),'completed_compile_successes':native['completed_compile_successes'],'cancelled_partial_compiles':native['cancelled_partial_compiles'],'unattempted':native['unattempted']},
 'evidence':artifacts,
 'review_request':'Independently review the complete candidate and source→successful execution→independent Real/native→actual Tensor/bytes/frame chain; do not replace or mutate the original defect packet.'}
(OUT/'delivery.json').write_text(json.dumps(packet,indent=2)+'\n')
print(json.dumps({'delivery':pin(OUT/'delivery.json'),'candidate_modules':len(proofs),
                  'repository_modules':len(modules),'compiler_cases':len(native['cases']),'completed_compile_successes':native['completed_compile_successes'],'cancelled_partial_compiles':native['cancelled_partial_compiles'],'unattempted':native['unattempted']},indent=2))
