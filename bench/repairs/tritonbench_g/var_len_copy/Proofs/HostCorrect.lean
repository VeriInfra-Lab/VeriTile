import bench.repairs.tritonbench_g.var_len_copy.Proofs.Host
import bench.repairs.tritonbench_g.var_len_copy.Proofs.Interleaving
import VeriTile.Meta.Specification

/-! Raw FP32 copy, all bit patterns, with signed-int32 metadata. No floating
arithmetic, rounding, numerical assumption or finite-payload restriction occurs. -/
namespace VeriTile.Bench.Repairs.VarLenCopy.FPEquiv
noncomputable section

specification var_len_copy_fp32 (m : Inputs) (segments : Nat → Segment)
    (h : InputContract m segments) (object : Nat) (before : Memory)
    (binding : TensorBinding m segments before) :
    ∃ out, runWrapper m segments object before = .ok out ∧
      out.returned = () ∧ out.outputObject = object ∧ out.outputMetadata = m .output ∧
      SegmentCopy (request m segments)
        (fun i => before (addr (base m .source) i)) (fun i => before (addr (base m .output) i))
        (fun i => out.memory (addr (base m .output) i)) ∧
      (∀ byte, (∀ x, Touched (request m segments) x →
        byte < addr (base m .output) x ∨ addr (base m .output) x + 4 ≤ byte) →
        observeByte out.memory byte = observeByte before byte) := by
  obtain ⟨out,he,hn,ho,hm,hv,hf⟩ := wrapper_correct m segments h object before binding
  exact ⟨out,he,hn,ho,hm,hv,fun byte miss => byte_frame m segments h before out.memory hf byte miss⟩

/-- Complete event permutations preserve the same source result. -/
theorem every_interleaving (m : Inputs) (segments : Nat → Segment)
    (h : InputContract m segments) (before : Memory) (binding : TensorBinding m segments before)
    (events : List Event)
    (order : events.Perm (allEvents (directLayout m) (request m segments)
      (List.range (m .oldStarts).numel))) :
    execute events before = launch (directLayout m) before :=
  interleaving_equal (directLayout m) (request m segments) before
    (prepared_direct m segments h before binding) (contract_valid m segments h) events order

#axiomsClean var_len_copy_fp32
#axiomsClean every_interleaving
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VarLenCopy.FPEquiv
