import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelReference
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelNative
namespace VeriTile.Bench.Repairs.Conv2d.KernelNative
open Native KernelNumeric
noncomputable section

/-- Closed production-default instruction set: only the IEEE RN-even scalar
FMA is reachable. The fixed unused values cannot affect the false/false path. -/
def defaultInstructions : Instructions :=
  ⟨Software.fmaRN,fun _ _=>0,fun _ _ c=>c,fun _ _ c=>c⟩

def defaultReference (s : Shape) (m : Memory Word) (x : Conv2d.Cell) : Word :=
  reference (operations defaultInstructions ⟨Bool.false,Bool.false⟩) s ⟨Bool.false,Bool.false⟩ m x

theorem fmas_congr (p q : Instructions) (h : ∀ a b c,p.fmaRN a b c=q.fmaRN a b c)
    (a b : Fin 32 → Word) (k : Nat) (carry : Word) : fmas p a b k carry=fmas q a b k carry := by
  induction k with
  | zero=>rfl
  | succ k ih=>rw [fmas,fmas,ih,h]

theorem default_reference (p : Instructions)
    (h : ∀ a b c,p.fmaRN a b c=Software.fmaRN a b c)
    (s : Shape) (m : Memory Word) (x : Conv2d.Cell) :
    reference (operations p ⟨Bool.false,Bool.false⟩) s ⟨Bool.false,Bool.false⟩ m x=defaultReference s m x := by
  unfold reference defaultReference
  simp only [reference,convert,operations,lower32,Bool.false_eq_true,if_false]
  simp_rw [fmas_congr p defaultInstructions h]

#specNonCircular defaultReference avoiding [KernelSource.kernel,KernelGrid.launch,KernelGrid.value]
#axiomsClean default_reference
end
end VeriTile.Bench.Repairs.Conv2d.KernelNative
