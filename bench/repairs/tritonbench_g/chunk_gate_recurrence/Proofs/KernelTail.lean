import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelMemory

/-! The backward kernel leaves its final DI/DG slices untouched. The original
host zeroing is outside the two kernel programs. -/
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelTail
open VeriTile.Triton FP.Structural KernelProgram KernelMemory
noncomputable section

theorem input_outside (c : Shape) (p : Program c) (e : Fin 64) (j : Fin 16) :
    ¬footprint .backward c p "DI" (address c p (c.N-1) e.val j.val) := by
  intro active
  rcases active with ⟨_,t,k,v,equal⟩ | ⟨wrong,_⟩ | ⟨wrong,_⟩
  · have same := (address_unique c p (c.N-1) t.val e k j v equal).1
    exact (Nat.ne_of_lt t.isLt) same.symm
  · exact absurd wrong (by decide)
  · exact absurd wrong (by decide)

theorem gate_outside (c : Shape) (p : Program c) :
    ¬footprint .backward c p "DG" (gradientAddress c p (c.N-1)) := by
  intro active
  rcases active with ⟨wrong,_⟩ | ⟨_,t,equal⟩ | ⟨wrong,_⟩
  · exact absurd wrong (by decide)
  · have same := gradient_unique c p (c.N-1) t.val equal
    exact (Nat.ne_of_lt t.isLt) same.symm
  · exact absurd wrong (by decide)

variable {α : Type} [Inhabited α]
theorem untouched (c : Shape) (p : Program c) (u v : State α)
    (frame : Frame .backward c p u v) :
    (∀ (e : Fin 64) (j : Fin 16),
      v.mem "DI" (address c p (c.N-1) e.val j.val)=u.mem "DI" (address c p (c.N-1) e.val j.val)) ∧
    v.mem "DG" (gradientAddress c p (c.N-1))=u.mem "DG" (gradientAddress c p (c.N-1)) :=
  ⟨fun e j => frame _ _ (input_outside c p e j),frame _ _ (gate_outside c p)⟩

#axiomsClean untouched
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelTail
