import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.CheckedBridge
import bench.tritonbench_g.context_attn_nopad.Proofs.HostCoverage

namespace VeriTile.Bench.Repairs.ContextAttnNopad.Checked
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section

/-- Independent causal natural-exp formula over arbitrary Real input arrays.
The actual binary64 host scale is retained; no encoder bounds the Real domain. -/
def mathematical (c : Call) (a : Formula.Inputs) (b h t d : Nat) : ℝ :=
  (∑ j ∈ Finset.range (t+1),Real.exp
      ((∑ e : Fin c.q.dims,a.q (c.q.offset (c.start b+t) h e.val)*
        a.k (c.k.offset (c.start b+j) h e.val))*c.scale)*
      a.v (c.v.offset (c.start b+j) h d)) /
    ∑ j ∈ Finset.range (t+1),Real.exp
      ((∑ e : Fin c.q.dims,a.q (c.q.offset (c.start b+t) h e.val)*
        a.k (c.k.offset (c.start b+j) h e.val))*c.scale)

/-- Checked whole-call mathematical result, all grid programs, complete logical
output coverage, original read roles, exact complement-of-stores frame and None. -/
theorem whole_mathematical_real (c : Call) (r : Regions) (v : Views) (A : FlatAlloc)
    (s : BlockState) (requestedMax : Nat) (start length : Nat → Int)
    (checked : checkInputs c r v A.extent requestedMax start length=Bool.true)
    (observed : MetadataObservation c r s start length) (labels : Labels r)
    (allocation : A.Disjoint) (covered : ∀ rg,rg∉A.regions → A.extent rg=0) :
    Spec.Real (∃ result,c.Returns r A s Bool.false result ∧
      (∀ b h t d,b < c.batches → h < c.q.heads → t < c.length b → d < c.q.dims →
        result.1.readMem A.flat (A.addr r.out (c.output b h t d))=
          mathematical c (inputs s r) b h t d) ∧
      BlockState.WriteWithin (c.writes r A s) (A.flattenState s) result.1 ∧
      (∀ rg∈c.inputRegions r,∀ off,off < A.extent rg →
        result.1.mem A.flat (A.addr rg off)=(A.flattenState s).mem A.flat (A.addr rg off)) ∧
      result.2=none) := by
  have guards := (checkInputs_iff c r v A.extent requestedMax start length).mp checked
  have hm := memory c r A s start length guards.2.1 guards.2.2.2 observed labels
  obtain ⟨result,run,values,frame,reads⟩ := host_complete hm allocation covered
    (complete c r A.extent guards.2.1) Bool.false
  exact ⟨result,run,values,frame,reads,run.2⟩

#specNonCircular mathematical avoiding [Call.Returns,Call.source,VeriTile.Triton.exec,Kernel.mergeFrames]
#axiomsClean whole_mathematical_real
end
end VeriTile.Bench.Repairs.ContextAttnNopad.Checked
