"""Freeze candidate-only delivery and pin the effective read-only dependency closure."""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).resolve().parent
EVIDENCE = ROOT / '.veritile-cache/kernel-evidence/int8_matmul_quantization_i32_safe'
DEST = ROOT / '.veritile-cache/task-deliveries/int8_matmul_quantization_i32_safe/v1'
spec = importlib.util.spec_from_file_location('i32_safe_builder', SOURCE/'build_local.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def rel(path):
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

def imports(path):
    # Workspace documentation can contain example import lines. Ignore
    # nested Lean block comments, line comments and string literals.
    source=path.read_text();clean=[];i=0;depth=0;quoted=False
    while i<len(source):
        if depth:
            if source.startswith('/-',i):depth+=1;i+=2;continue
            if source.startswith('-/',i):depth-=1;i+=2;continue
            clean.append('\n' if source[i]=='\n' else ' ');i+=1;continue
        if quoted:
            if source[i]=='\\':clean.extend('  ');i+=2;continue
            if source[i]=='"':quoted=False
            clean.append('\n' if source[i]=='\n' else ' ');i+=1;continue
        if source.startswith('/-',i):depth=1;i+=2;continue
        if source.startswith('--',i):
            end=source.find('\n',i)
            if end<0:break
            i=end;continue
        if source[i]=='"':quoted=True;clean.append(' ');i+=1;continue
        clean.append(source[i]);i+=1
    return re.findall(r'^(?:public[ \t]+)?import[ \t]+([A-Za-z0-9_.]+)',''.join(clean),re.M)

def main():
    assert not DEST.exists(), 'Frozen v1 already exists; never overwrite a frozen delivery.'
    pins={}
    preserved=[]
    def add(path,expected=None):
        path=Path(path)
        if not path.is_absolute():path=ROOT/path
        assert path.is_file(),str(path)
        digest=sha(path)
        if expected is not None:assert digest==expected,(str(path),digest,expected)
        pins[str(path)]={'path':rel(path),'sha256':digest,'bytes':path.stat().st_size}
        return pins[str(path)]
    def refs(value):
        if isinstance(value,dict):
            if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
                yield value
            for child in value.values():yield from refs(child)
        elif isinstance(value,list):
            for child in value:yield from refs(child)
    scalar=ROOT/'.veritile-cache/task-deliveries/int8_matmul_quantization_scalar_cast/delivery.json'
    frozen=json.loads(scalar.read_text())
    exact={
      str(scalar.relative_to(ROOT)):'a237d7e26b73286fbd69dbc86944cde86e4c24c57f62fc026fe97656995840d0',
      '.veritile-cache/task-deliveries/int8_matmul_quantization_scalar_cast/independent-review.json':'2a0cbb6bc2beff0bb0f805281de88b8ad19a04cd203245ae7f5f5b78e33d067d',
      '.veritile-cache/kernel-evidence/int8_matmul_quantization_scalar_cast/accumulation-scope-assessment-20261020.json':'74e3d5f0fd58ba4cc6ac7e4ea8cf6189c54cf8d4bcc61094e2b80c7b2d4975ef',
      'bench/repairs/tritonbench_g/int8_matmul_quantization/Proofs/Ideal.lean':'8ebcf2da4bf7ab09a9389fdf379db3dacc7374f59fd2f2d7ee68c2bea4a82a60',
    }
    for path,digest in exact.items():preserved.append(add(path,digest))
    for record in refs(frozen):
        preserved.append(add(record['path'],record['sha256']))
    # The frozen compiler profile directories contain the retained emitted
    # TTIR/TTGIR/LLVM/PTX products, not newly generated evidence.
    for directory in [scalar.parent/'native',scalar.parent/'large-producer']:
        for p in directory.rglob('*'):
            if p.is_file():add(p)
    candidates=sorted(SOURCE.rglob('*.lean'))
    receipts=[]
    for src in candidates:
        add(src)
        obj=builder.OUT/src.relative_to(ROOT).with_suffix('.olean')
        receipt=obj.with_suffix('.workspace-build.json')
        record=json.loads(receipt.read_text())
        assert record['exit_code']==0 and record['source']==rel(src)
        assert record['source_sha256']==sha(src)
        assert record.get('direct_imports')==builder.imports_of(src),str(src)
        for p,digest in record['objects'].items():add(p,digest)
        add(record['log']['path'],record['log']['sha256'])
        receipts.append(add(receipt))
    # Pin transitive workspace source/object closure at the same effective
    # LEAN_PATH used by the final candidate builds. Package modules are
    # recorded as explicit external read-only object inputs.
    dependencies=[];external=[];seen=set();missing_receipts=[]
    queue=[m for src in candidates for m in imports(src)]
    while queue:
        module=queue.pop()
        if module in seen:continue
        seen.add(module)
        modulepath=Path(*module.split('.'))
        src=ROOT/modulepath.with_suffix('.lean')
        obj=next((Path(lib)/modulepath.with_suffix('.olean') for lib in builder.LIBS
                  if (Path(lib)/modulepath.with_suffix('.olean')).is_file()),None)
        assert obj is not None,module
        resolved=obj.resolve()
        if src.is_file():
            row={'module':module,'source':add(src),'object':add(resolved),
                 'effective_import_path':str(obj)}
            for suffix in ['.ilean','.workspace-owned','.workspace-build.json']:
                p=resolved.with_suffix(suffix)
                if p.is_file():add(p)
            if not resolved.with_suffix('.workspace-build.json').is_file():missing_receipts.append(module)
            dependencies.append(row)
            queue.extend(imports(src))
        else:
            external.append({'module':module,'object':str(resolved),'sha256':sha(resolved),'bytes':resolved.stat().st_size})
    for path in SOURCE.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:add(path)
    for name in ['candidate-check.json','scope-precision-20261020.json','environment-startup.json']:
        add(EVIDENCE/name)
    for name in ['lean-toolchain','lake-manifest.json']:
        p=ROOT/name
        if p.is_file():add(p)
    toolchain=Path('/tmp/veritile-lean-toolchain/bin/lean')
    check=json.loads((EVIDENCE/'candidate-check.json').read_text())
    assert check['status']=='PASS' and check['source_sha256']==sha(SOURCE/'int8_matmul_quantization_i32_safe.py')
    assert check['exact_single_guard_delta'] and check['complete_AST_binding']
    assert len(check['cases'])==62 and len(check['mutants_rejected'])==7
    assert {'Kernels.lean','Contract.lean','Correct.lean','FPEquiv.lean'} <= {p.name for p in candidates}
    packet={
      'schema_version':1,'task':'int8_matmul_quantization','candidate':'int8_matmul_quantization_i32_safe',
      'status':'FROZEN_CANDIDATE_FOR_INDEPENDENT_WHOLE_TASK_REVIEW',
      'source_delta':'Only active (M and N) GEMM rejects K > 131071 on frozen scalar_cast baseline.',
      'public_entries':['Kernels.lean','Contract.lean','Correct.lean','FPEquiv.lean'],
      'public_theorems':['Public.correct','Public.fpEquiv','Public.composedCorrect','Public.composedFPEquiv','Public.producerCorrect','Public.producerFPEquiv'],
      'numerical_scope':{'ideal':'Independent dense Real formula; complete source interpretation.',
        'native':'Exact source-ordered signed32/FP32/mandatory-FP16/destination-typed-atomic word graph.',
        'native_equals_real':False,'payload_or_scale_guards':False,
        'local_bound':'Every distinct masked source subset fits signed32 at K <= 131071; concrete scalar/K32/fused/nonfused/loop recurrence.'},
      'boundary':['Public admission includes input liveness, shared Tensor layout/pointer laws, output separation, and actual prepared-backing simulation.',
        'Composed public liveness includes original backing plus both actual producer allocations; frame quantifies all original live storage.',
        'Native max/divide/typed atomics are explicit single-instruction policies and reductions are structural trees.',
        'Inherited unchanged Torch helper retains its explicit local Torch/compiler ABI; no baseline full-AST identity is asserted for candidate module.',
        'No new compiler probes, numerical experiments, GPU runs or shared-core/ledger/Lake/canonical-selection changes.'],
      'validation':{'candidate_lean_modules':len(candidates),'host_metadata_cases':len(check['cases']),
        'mutants_rejected':len(check['mutants_rejected']),'gemm_configs':34,'producer_configs':4,
        'native_equals_real_claim':False,'source_ast_report':rel(EVIDENCE/'candidate-check.json')},
      'candidate_receipts':receipts,'workspace_dependencies':dependencies,'external_import_objects':external,
      'inherited_modules_without_receipt':sorted(missing_receipts),'preserved_records':preserved,
      'toolchain':{'path':str(toolchain),'sha256':sha(toolchain),'version':'Lean 4.29.0 / 98dc76e3c0a9b856c9b98726b713fb04fab16740'},
      'files':[]
    }
    DEST.mkdir(parents=True)
    for path,row in sorted(pins.items()):
        p=Path(path)
        # Inherited workspace modules may resolve through a read-only build
        # tree in /tmp. Snapshot those files under a separate absolute-path
        # namespace while preserving their original path in the manifest.
        target=(DEST/'files'/p.relative_to(ROOT) if p.is_relative_to(ROOT)
                else DEST/'external-files'/p.relative_to(p.anchor))
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target)
        assert sha(target)==row['sha256']
        row=dict(row,frozen_path=str(target.relative_to(DEST)))
        packet['files'].append(row)
    manifest=DEST/'packet.json';manifest.write_text(json.dumps(packet,indent=2)+'\n')
    # Immutable snapshot verification checks every live and frozen record,
    # the exact source/build/import pins, and all preserved prior artifacts.
    for row in packet['files']:
        assert sha(ROOT/row['path'])==row['sha256']
        assert sha(DEST/row['frozen_path'])==row['sha256']
    verification={'status':'PASS','files':len(packet['files']),'candidate_modules':len(candidates),
      'preserved_records':len(preserved),'packet_sha256':sha(manifest),
      'missing_inherited_receipts':len(missing_receipts),'external_import_objects':len(external)}
    (DEST/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    archive=DEST.parent/'packet-v1.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        tar.add(DEST,arcname='int8_matmul_quantization_i32_safe-v1')
    delivery={'task':packet['task'],'candidate':packet['candidate'],'status':packet['status'],
      'packet':{'path':rel(manifest),'sha256':sha(manifest)},
      'archive':{'path':rel(archive),'sha256':sha(archive)},
      'verification':{'path':rel(DEST/'verification.json'),'sha256':sha(DEST/'verification.json')},
      'validation':packet['validation'],'independent_review':'pending_root_designated_reviewer',
      'root_actions':'Independent whole-task review, acceptance and any ledger/Lake/canonical-selection action remain with root.'}
    path=DEST.parent/'delivery.json';path.write_text(json.dumps(delivery,indent=2)+'\n')
    print(json.dumps({'delivery':rel(path),'sha256':sha(path),**delivery},indent=2))

if __name__=='__main__':main()
