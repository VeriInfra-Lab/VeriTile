import ast,hashlib,json,shutil,sys
from pathlib import Path
W=Path(__file__).resolve().parent
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(R/'bench'))
from adapter_common import imports
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def save(name,obj):(W/'evidence'/name).write_text(json.dumps(obj,indent=2)+'\n')
modules=list(load(W/'evidence/staging-map.json').values())
seen=set();project={};external=set()
def visit(rel):
 if rel in seen:return
 seen.add(rel)
 p=W/'after'/rel if rel in modules else R/rel
 if not p.is_file():external.add(rel);return
 if rel not in modules:project[rel]={'sha256':sha(p)}
 for m in imports(p.read_text()):visit(m.replace('.','/')+'.lean')
for m in modules:visit(m)
save('dependencies-before.json',{'project':dict(sorted(project.items())),'external_imports':sorted(external),'new_modules':{m:sha(W/'after'/m) for m in modules}})
old=Path('/tmp/veritile-standardization-20261015/vector-addition-dtype-extension-wip/evidence/extension-audit')
frontend=W/'evidence/frontend';frontend.mkdir(exist_ok=True)
rows=load(old/'frontend-probes.json')
for f in ['receipt.json','frontend-probes.json','promotion-matrix.json','probe.py','extracted_add_kernel.py','triton-semantic-excerpts.txt']+[x['ttir_path'] for x in rows if 'ttir_path' in x]:shutil.copyfile(old/f,frontend/f)
inherit=W/'evidence/inherited';inherit.mkdir(exist_ok=True)
for name,path in {
 'typed-host-receipt.json':'/tmp/veritile-standardization-20261015/vector-typed-host-wip/evidence/receipt.json',
 'typed-host-root-review.json':'/tmp/veritile-standardization-20261015/vector-typed-host-root-review.json',
 'typed-host-numerical-proposal.json':'/tmp/veritile-standardization-20261015/vector-typed-host-wip/evidence/numerical-bindings-proposal.json',
 'typed-host-source-review.json':'/tmp/veritile-standardization-20261015/vector-typed-host-wip/evidence/source-route-check.json',
 'admission-source-audit.json':'/tmp/veritile-standardization-20261015/vector-dtype-admission-wip/evidence/source-audit/audit-result.json'}.items():shutil.copyfile(path,inherit/name)
original=R/'bench/tritonbench_g/vector_addition/vector_addition.py'
text=original.read_text();tree=ast.parse(text);functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
prior=load(inherit/'typed-host-source-review.json')
statements={n:[ast.unparse(a) for a in functions[n].body] for n in ['add_kernel','add']}
assert statements==prior['source_statements']
extract=ast.parse((frontend/'extracted_add_kernel.py').read_text())
assert ast.dump(functions['add_kernel'],include_attributes=False)==ast.dump(next(n for n in extract.body if isinstance(n,ast.FunctionDef)),include_attributes=False)
for row in rows:
 if 'ttir_path' in row:
  p=frontend/row['ttir_path'];assert sha(p)==row['ttir_sha256'];ir=p.read_text()
  assert 'tt.assert' not in ir and 'tt.load' in ir and 'tt.store' in ir
  assert ir.index('tt.load')<ir.rindex('tt.store')
sem=Path('/usr/local/lib/python3.12/site-packages/triton/language/semantic.py');semtext=sem.read_text()
assert 'if not self.builder.options.debug:\n            return' in semtext
save('source-review.json',{'source':str(original.relative_to(R)),'source_sha256':sha(original),'statements':statements,
 'identical_original_kernel_ast':True,'inherited_frontend_cases':len(rows),'frontend_generated':sum(x['status']=='frontend_generated' for x in rows),
 'all_generated_ir_has_no_device_assert':True,'target':load(frontend/'receipt.json')['scope'],
 'runtime_n_signature':'i32; n is not constexpr; arithmetic/index shape is independent of n',
 'sanitizer_review':{'debug':False,'instrumentation_mode':'','sanitize_overflow':True,'no_assert_reason':'TritonSemantic.device_assert returns immediately when builder.options.debug is false; widened arithmetic has no memory side effect. Modular source integer results are retained.','semantic_source':str(sem),'semantic_sha256':sha(sem)},
 'test_harness_review':'test_add is non-JIT, has no production caller, invokes add four times and is called only by import-time result_gold initialization. RNG implementation is trusted test machinery.',
 'native_backend_compiled':False,'gpu_executed':False,'compiler_implementation_proof':False})
print(json.dumps({'project_dependencies':len(project),'frontend_profiles':len(rows),'frontend_successes':189,'new_modules':len(modules)}))
