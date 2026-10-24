import bench.repairs.tritonbench_g.var_len_copy.Proofs.Contract
import bench.repairs.tritonbench_g.var_len_copy.Proofs.Conformance

namespace VeriTile.Bench.Repairs.VarLenCopy
noncomputable section

structure Result where
  returned : Unit
  outputObject : Nat
  outputMetadata : VeriTile.Torch.TensorMetadata
  memory : Memory

def runWrapper (m : Inputs) (segments : Nat → Segment) (outputObject : Nat)
    (before : Memory) : Except InputError Result := do
  checkInputs m segments
  if (m .oldStarts).numel = 0 then return ⟨(),outputObject,m .output,before⟩
  return ⟨(),outputObject,m .output,Native.sourceLaunch (directLayout m) before⟩

theorem wrapper_correct (m : Inputs) (segments : Nat → Segment)
    (h : InputContract m segments) (object : Nat) (before : Memory)
    (binding : TensorBinding m segments before) :
    ∃ out, runWrapper m segments object before = .ok out ∧
      out.returned = () ∧ out.outputObject = object ∧ out.outputMetadata = m .output ∧
      SegmentCopy (request m segments)
        (fun i => before (addr (base m .source) i)) (fun i => before (addr (base m .output) i))
        (fun i => out.memory (addr (base m .output) i)) ∧
      (∀ cell, (¬∃ x, Touched (request m segments) x ∧ cell=addr (base m .output) x) →
        out.memory cell = before cell) := by
  have accepted := (checkInputs_iff m segments).mpr h
  have prep := prepared_direct m segments h before binding
  have valid := contract_valid m segments h
  have source := Native.source_launch (directLayout m) (request m segments) before prep valid
  have result := launch_correct (directLayout m) (request m segments) before prep valid
  by_cases empty : (m .oldStarts).numel = 0
  · refine ⟨⟨(),object,m .output,before⟩,by simp [runWrapper,accepted,empty],rfl,rfl,rfl,?_,?_⟩
    · constructor
      · intro p i hp
        simp [request,empty] at hp
      · intro x _ _; rfl
    · intro cell _; rfl
  · exact ⟨⟨(),object,m .output,launch (directLayout m) before⟩,
      by simp [runWrapper,accepted,empty,source],rfl,rfl,rfl,result⟩

/-- Little-endian byte observation of the naturally aligned FP32/int32 heap. -/
def observeByte (memory : Memory) (address : Nat) : BitVec 8 :=
  (memory (address / 4 * 4)).extractLsb' (address % 4 * 8) 8

theorem byte_frame (m : Inputs) (segments : Nat → Segment) (h : InputContract m segments)
    (before after : Memory)
    (frame : ∀ cell, (¬∃ x, Touched (request m segments) x ∧ cell=addr (base m .output) x) →
      after cell = before cell) (byte : Nat)
    (outside : ∀ x, Touched (request m segments) x →
      byte < addr (base m .output) x ∨ addr (base m .output) x + 4 ≤ byte) :
    observeByte after byte = observeByte before byte := by
  unfold observeByte
  rw [frame]
  rintro ⟨x,hx,he⟩
  obtain ⟨p,i,hp,hi,rfl⟩ := hx
  simp only [request] at hp hi
  have nonempty : 0 < (m .output).numel := by
    have bound := (h.outputBound p hp (by omega)).2
    omega
  have align := h.alignment .output nonempty
  have ext := outside _ ⟨p,i,hp,hi,rfl⟩
  unfold addr at he ext
  omega

#axiomsClean wrapper_correct
#axiomsClean byte_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VarLenCopy
