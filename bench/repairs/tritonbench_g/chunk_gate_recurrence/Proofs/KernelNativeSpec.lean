import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelExecutionSpec
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelWords

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelNative
open VeriTile.Triton FP.Structural KernelProgram KernelCarrier
noncomputable section
def expected (isa : Instructions) (plan : Reduction.ProductPlan 1024) (kind : Kind)
    (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c) (u : State Node)
    (r : RegionName) (t : Nat) (i : Lane) : Word :=
  let a := KernelWords.inputs (KernelReference.read c s p u)
  match kind with
  | .forward => KernelWords.forward isa (swapped .forward c) a seed t i
  | .backward =>
    if r="DI" then KernelWords.reverse isa (swapped .backward c) a c.N (c.N-1-t) i
    else if r="DG" then KernelWords.contraction isa plan
      (KernelWords.reverse isa (swapped .backward c) a c.N (c.N-1-t)) (a.saved t)
    else KernelWords.reverse isa (swapped .backward c) a c.N c.N i

theorem expected_lowering (isa : Instructions) (plan : Reduction.ProductPlan 1024) (kind : Kind)
    (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c) (u : State Node)
    (r : RegionName) (t : Nat) (i : Lane) :
    KernelExecution.expected (model isa plan kind c) kind c s seed p u r t i=
      atom (expected isa plan kind c s seed p u r t i) := by
  cases kind with
  | forward => simp only [KernelExecution.expected,expected,model,KernelWords.forward_lowering]
  | backward =>
    simp only [KernelExecution.expected,expected,model,KernelWords.reverse_lowering]
    split_ifs
    · rfl
    · rw [KernelWords.contraction_lowering]
      rfl
    · rfl

def Outputs (isa : Instructions) (plan : Reduction.ProductPlan 1024) (kind : Kind)
    (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c) (u v : State Node) : Prop := match kind with
  | .forward => ∀ (t : Fin c.N) (e : Fin 64) (j : Fin 16),
      v.mem "O" (address c p t.val e.val j.val)=
        storedWord .real (expected isa plan .forward c s seed p u "O" t.val (e,j,PUnit.unit))
  | .backward =>
    (∀ (t : Fin (c.N-1)) (e : Fin 64) (j : Fin 16),
      v.mem "DI" (address c p t.val e.val j.val)=
        storedWord .real (expected isa plan .backward c s seed p u "DI" t.val (e,j,PUnit.unit))) ∧
    (∀ t : Fin (c.N-1),v.mem "DG" (gradientAddress c p t.val)=
      storedWord .real (expected isa plan .backward c s seed p u "DG" t.val (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit))) ∧
    (∀ (e : Fin 64) (j : Fin 16),v.mem "DL" (dseedAddress c p e.val j.val)=
      storedWord .fp32 (expected isa plan .backward c s seed p u "DL" 0 (e,j,PUnit.unit)))

#specNonCircular expected avoiding [FP.Structural.exec,FP.Structural.run,KernelExecution.expected]
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelNative
