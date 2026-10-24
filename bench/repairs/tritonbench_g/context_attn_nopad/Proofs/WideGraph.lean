import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.NativeGraph
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.WideFP

namespace VeriTile.Bench.Repairs.ContextAttnNopad.WideGraph
open VeriTile.Triton FP.Structural
open NativeGraph
noncomputable section

/-- Complete staged native graph on the int32/int64 domain. Initial logical
words are bound by Prepared. All native integer sites and final backedges,
metadata byte sizes, successful execution/readback and full frames are checked.
The Plan and Operations remain local native structural/primitive parameters. -/
theorem whole_call (plan : Plan) {c : Call} {r : Regions} {A : FlatAlloc}
    {layoutState : BlockState} {s : State Expr} (input : Inputs)
    (hm : WideDomain.Memory c r A layoutState) (ha : NativeProgram.ABI c A)
    (types : WideNative.MetadataTypes) (fits : WideNative.MetadataFits c types)
    (alloc : WideNative.MetadataAlloc r A types)
    (view : FPHostSource.OutputView Expr c) (before : Nat → Expr)
    (prepared : Prepared c r s input view before) :
    FPHostPrecise.Contract (backend plan).algebra c r A layoutState s view before ∧
    (∀ j : HostPlan.Job (FPHostSource.effective c r s),
      WideNative.Projects (FPHostSource.effective c r s) r A layoutState types j ∧
      ∀ iter, iter ≤ (HostPlan.global j.2.2).val →
        (NativeProgram.word (iter*128) + NativeProgram.word 128).toInt = ((iter*128+128 : Nat) : Int)) ∧
    (∀ b, b < c.batches →
      (NativeProgram.word b * NativeProgram.word types.start.bytes).toInt = ((b*types.start.bytes : Nat) : Int) ∧
      (NativeProgram.word b * NativeProgram.word types.length.bytes).toInt = ((b*types.length.bytes : Nat) : Int)) := by
  refine ⟨WideFP.whole_host_precise (backend plan) hm ha fits prepared.metadata view before prepared.initialized,
    ?_,WideNative.metadata_byte_offsets hm types alloc⟩
  have he : FPHostSource.effective c r s = HostPlan.effective c r layoutState := by
    unfold FPHostSource.effective HostPlan.effective
    rw [FPHostSource.metadataMaximum_eq prepared.metadata,WideDomain.metadataMaximum_eq hm]
  rw [he]
  exact WideNative.complete_projection hm ha types fits

/-- Local graph interpretation preserves the successful whole result and frame.
The primitive interpreter contains an independent ternary FMA and seeded dot;
no equality with separately rounded source operations is used. -/
theorem evaluated_call (plan : Plan) (ops : Operations) {c : Call} {r : Regions} {A : FlatAlloc}
    {layoutState : BlockState} {s : State Expr} (input : Inputs)
    (hm : WideDomain.Memory c r A layoutState) (ha : NativeProgram.ABI c A)
    (types : WideNative.MetadataTypes) (fits : WideNative.MetadataFits c types)
    (alloc : WideNative.MetadataAlloc r A types)
    (view : FPHostSource.OutputView Expr c) (before : Nat → Expr)
    (prepared : Prepared c r s input view before) :
    ∃ (raw : State Expr) (result : FPHostSource.Result Expr),
      FPHostSource.runHost (backend plan).algebra c r s view before = some result ∧
      FPHostSource.reference (backend plan).algebra c r s view before = some result ∧
      result.objectId = view.objectId ∧ result.allocationId = view.allocationId ∧ result.pythonReturn = none ∧
      (∀ i, eval ops (view.action (result.memory (view.address i))) =
        eval ops ((raw.mem r.out (Logical.coordOffset c i)).read .real)) ∧
      (∀ addr, (¬ ∃ i, Logical.Written c i ∧ view.address i = addr) →
        eval ops (result.memory addr) = eval ops (before addr)) ∧
      (∀ rg off, ¬ (FPHostSource.effective c r s).writes r A layoutState (A.flat,A.addr rg off) →
        raw.mem rg off = s.mem rg off) := by
  obtain ⟨raw,result,_,hr,href,_,hid,halloc,hret,hread,hframe,hflat⟩ :=
    (whole_call plan input hm ha types fits alloc view before prepared).1
  refine ⟨raw,result,hr,href,hid,halloc,hret,?_,?_,hflat⟩
  · intro i
    rw [hread i]
  · intro addr h
    rw [hframe addr h]

#print axioms whole_call
#print axioms evaluated_call
end
end VeriTile.Bench.Repairs.ContextAttnNopad.WideGraph
