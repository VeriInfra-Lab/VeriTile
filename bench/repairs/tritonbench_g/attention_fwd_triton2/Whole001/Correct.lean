import bench.repairs.tritonbench_g.attention_fwd_triton2.Whole001.Proofs.RealReturned
namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001
open VeriTile Triton FP.Structural Source Whole Machine
noncomputable section
/-- Actual full-source exact-Real execution, allocations, arbitrary legal grid order,
finite returned BF16-tagged tensor and frame, with an independent initial-input target.
This is the mathematical interpretation, not a claim of zero native rounding error. -/
theorem whole_returned_real (p : Parameters) (bases : Bases) (domain : Domain p bases)
    (abi : InputABI p) (before : RealReturned.InputValues) (fits : Physical.InputFits p before.store)
    (alloc : Allocation p abi before.store.extend) (work : List Program)
    (legal : (programs (grid p)).Perm work) :
    ∃ result,Physical.invoke (fun _=>Real.algebra) (fun _=>Real.dotModel) p abi before.store alloc work=some result ∧
      result.tensor=alloc.output ∧ result.memory.capacity p.Out=words p ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<96),
        ∃ index : Fin (result.memory.capacity p.Out),index.val=logicalAddress p b h n d ∧
          result.memory.cells p.Out index=.mk .bf16 (some (RealReturned.target p before.logical b h n d))) ∧
      (∀ r (i : Fin (before.capacity r)),r≠p.Out → r≠alloc.unusedM.region →
        ∃ j : Fin (result.memory.capacity r),j.val=i.val ∧ result.memory.cells r j=before.store.cells r i) ∧
      Access.Safe p ∧ Closed p bases ∧ Physical.InputAccess p before.store :=
  RealReturned.returned_real p bases domain abi before.store fits alloc before.logical
    (RealReturned.input_values_reads p before) work legal

#axiomsClean whole_returned_real
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001
