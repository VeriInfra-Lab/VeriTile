import bench.repairs.tritonbench_g.matmul_kernel.Proofs.Pair
import bench.repairs.tritonbench_g.matmul_kernel.Proofs.KernelReplay

namespace VeriTile.Bench.Repairs.MatmulKernel.Pair
open VeriTile.Triton FP.Structural KernelSource KernelTensor KernelEntries
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Same actual ordered byte stores and allocation adapter; each launch now
executes the complete remaining-length source quotation. -/
def candidateReplay (M : Algebra Word) (D : DotAcc Word) (p : Prepared) :
    List Program → State Word → KernelStorage.Heap → Option (State Word × KernelStorage.Heap)
  | [],s,heap => some (s,heap)
  | pid::tail,s,heap => do
      let next ← FP.Structural.exec (backend M D) (remaining p.config) (atProgram s pid)
      candidateReplay M D p tail next (KernelReplay.programBytes M D p s pid heap)

theorem replay_equal (M : Algebra Word) (D : DotAcc Word) (p : Prepared)
    (order : List Program) (s : State Word) (heap : KernelStorage.Heap) :
    KernelReplay.replay M D p order s heap=candidateReplay M D p order s heap := by
  induction order generalizing s heap with
  | nil => rfl
  | cons pid rest ih =>
    simp only [KernelReplay.replay,candidateReplay,kernel_equal]
    cases hr : FP.Structural.exec (backend M D) (remaining p.config) (atProgram s pid) with
    | none => rfl
    | some t => exact ih t _

/-- The new scalar subtraction is represented exactly throughout every
admitted reached iteration; the original signed32 guard already suffices. -/
theorem remaining_bounds (p : Prepared) (pid : Program) (launch : pid∈programs p.config)
    (q : Nat) (reached : q<p.config.steps) :
    q*p.config.BK<p.config.K ∧ p.config.K-q*p.config.BK<2^31 ∧
      (p.config.K : Int)-(q*p.config.BK : Nat)=(p.config.K-q*p.config.BK : Nat) := by
  have rounded := Nat.div_mul_le_self (p.config.K+p.config.BK-1) p.config.BK
  have offset := Nat.mul_le_mul_right p.config.BK (Nat.succ_le_of_lt reached)
  change (q+1)*p.config.BK≤((p.config.K+p.config.BK-1)/p.config.BK)*p.config.BK at offset
  rw [Nat.add_mul,Nat.one_mul] at offset
  have positive := p.valid.bk
  have dimensions := (KernelBounds.signed p pid launch).dims
  omega

#axiomsClean replay_equal
#axiomsClean remaining_bounds
end
end VeriTile.Bench.Repairs.MatmulKernel.Pair
