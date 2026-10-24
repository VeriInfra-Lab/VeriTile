from pathlib import Path
import hashlib,json,shutil
T=Path('/tmp/veritile-standardization-20261015');W=T/'add-value-finite-acceptance-revision1-wip';R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');OLD=T/'add-value-finite-acceptance-wip';V=T/'vector-finite-acceptance-revision2-wip'
W.mkdir();A=W/'after/bench/tritonbench_g/add_value';(A/'Proofs').mkdir(parents=True);E=W/'evidence';E.mkdir();I=E/'inherited';I.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ['compile.py','audit_setup_order.py']:shutil.copyfile(V/name,W/name)
shutil.copyfile(OLD/'probe_native.py',W/'probe_native.py')
for name in ['receipt.json','domain-review.json','numerical-authority-delta.json','scalar-operation-inventory.json','catalog-supplement-task.proposed.json']:
 p=OLD/'evidence'/name
 if p.exists():shutil.copyfile(p,I/('prior-'+name))
for p in (OLD/'after').rglob('*.lean'):
 dst=I/'prior-sources'/(str(p.relative_to(OLD/'after'))+'.txt');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
scope=T/'annotated-i32-scope-review';receipt=json.loads((scope/'receipt.json').read_text());dst=I/'annotated-i32-scope-review';dst.mkdir()
for rel,h in receipt['artifacts'].items():
 p=scope/rel;assert sha(p)==h;q=dst/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
shutil.copyfile(scope/'receipt.json',dst/'receipt.json')
review=T/'add-value-finite-root-validation';rd=I/'prior-root-validation';rd.mkdir()
for p in review.iterdir():
 if p.is_file():shutil.copyfile(p,rd/p.name)
(E/'predecessor-policy.json').write_text(json.dumps({'prior_receipt':str(OLD/'evidence/receipt.json'),'prior_receipt_sha256':sha(OLD/'evidence/receipt.json'),'prior_root_review':str(review),'scope_gap_receipt_sha256':sha(scope/'receipt.json'),'old_candidate_registration':False,'policy':'Preserve the frozen five-module bounded candidate and root checks as historical evidence. New native proposal has no dependency on its uninstalled modules and corrects actual annotated i32/full-domain behavior.'},indent=2)+'\n')
scalar=(OLD/'after/bench/tritonbench_g/add_value/Proofs/FiniteScalar.lean').read_text().replace('namespace AddValueFinite','namespace AddValueNative').replace('end AddValueFinite','end AddValueNative')
(A/'Proofs/NativeScalar.lean').write_text(scalar)
(A/'Proofs/NativeArithmetic.lean').write_text('''import bench.tritonbench_g.vector_addition.Proofs.NativeArithmetic
import bench.tritonbench_g.add_value.Proofs.NativeScalar

namespace AddValueNative
open VeriTile.Bench.TritonBenchG.VectorAddition
open DTypeScope TypedHost Finite
abbrev hostExtent := Native.hostExtent
abbrev grid := Native.grid
abbrev gridSupported := Native.gridSupported
abbrev bound := Native.bound
abbrev offset := Native.offset
abbrev mask := Native.mask
abbrev pointer := Native.pointer
abbrev activeTile := Native.activeTile
abbrev Buffer := Native.Buffer
abbrev dtype (d : Storage) : DType := wideDType d
structure Layout where
  x : Buffer
  output : Buffer
  device : Nat
def Layout.Fresh (p : Layout) : Prop := p.x.Disjoint p.output

theorem all_masks_false_at_two31 (pid lane : Nat) : mask (2^31) pid lane=Bool.false := Native.all_masks_false_at_two31 pid lane
theorem active_tile_empty_at_two31 (pid : Nat) : activeTile (2^31) pid=[] := Native.active_tile_empty_at_two31 pid
theorem counterexample_grid : grid (2^31)=2^21 ∧ gridSupported (2^31) ∧ hostExtent (2^31) := Native.counterexample_grid
#axiomsClean all_masks_false_at_two31
#axiomsClean counterexample_grid
end AddValueNative
''')
s=(R/'bench/tritonbench_g/vector_addition/Proofs/NativeMemory.lean').read_text().replace('import bench.tritonbench_g.vector_addition.Proofs.NativeScalar','import bench.tritonbench_g.add_value.Proofs.NativeArithmetic').replace('namespace VeriTile.Bench.TritonBenchG.VectorAddition.Native','namespace AddValueNative').replace('end VeriTile.Bench.TritonBenchG.VectorAddition.Native','end AddValueNative').replace('open VeriTile.Triton DTypeScope TypedHost Finite','open VeriTile.Triton\nopen VeriTile.Bench.TritonBenchG.VectorAddition\nopen DTypeScope TypedHost Finite Native.Buffer').replace('  yAddress : Nat\n','')
(A/'Proofs/NativeMemory.lean').write_text(s)
s=(R/'bench/tritonbench_g/vector_addition/Proofs/NativeGrid.lean').read_text().replace('import bench.tritonbench_g.vector_addition.Proofs.NativeProgram','import bench.tritonbench_g.add_value.Proofs.NativeProgram').replace('namespace VeriTile.Bench.TritonBenchG.VectorAddition.Native','namespace AddValueNative').replace('end VeriTile.Bench.TritonBenchG.VectorAddition.Native','end AddValueNative').replace('open VeriTile.Triton DTypeScope TypedHost Finite','open VeriTile.Triton\nopen VeriTile.Bench.TritonBenchG.VectorAddition\nopen DTypeScope TypedHost Finite').replace('(r : Route)','(d : Storage)').replace('(Carrier r)','(Word (.storage d))').replace('r.signature.1','dtype d').replace(' r ',' d ')
(A/'Proofs/NativeGrid.lean').write_text(s)
print(W)
