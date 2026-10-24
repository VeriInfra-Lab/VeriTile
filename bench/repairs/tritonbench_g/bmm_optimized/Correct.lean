import bench.repairs.tritonbench_g.bmm_optimized.Contract
import bench.repairs.tritonbench_g.bmm_optimized.Proofs.Host
import bench.repairs.tritonbench_g.bmm_optimized.Proofs.SourceIndex
namespace VeriTile.Bench.Repairs.BmmOptimized
theorem correct_real (call : Host.Call) (c : Config) (callerDevice : Nat) (initial : Memory ℝ)
    (x : Cell) (hx : call.shape.Active x) :
    (Host.invokeReal call c callerDevice initial).memory .output (call.shape.o x.batch x.row x.column)=
      Host.product call initial x := Host.whole_real call c callerDevice initial x hx
#axiomsClean correct_real
end VeriTile.Bench.Repairs.BmmOptimized

namespace VeriTile.Bench.Repairs.BmmOptimized.Checked
open Host Native
noncomputable def runReal (p : Metadata) (c : Config) (device : Nat) (m : Memory ℝ) := do
  checkInputs p
  return invokeReal p.call c device m

specification checked_real (p : Metadata) (admitted : InputContract p) (ordinary : Ordinary p.call)
    (c : Config) (device : Nat) (m : Memory ℝ) :
    runReal p c device m=.ok (invokeReal p.call c device m) ∧
      Public.RealClaim p.call c device m (invokeReal p.call c device m) ∧ IndexABI p.call.shape := by
  refine ⟨by simp [runReal,(checkInputs_iff p).2 admitted],?_,admitted.indices⟩
  refine ⟨rfl,⟨ordinary.freshA,ordinary.freshB⟩,rfl,rfl,rfl,⟨rfl,rfl⟩,
    invokeReal_memory p.call c device m,Host.whole_real p.call c device m,?_,?_,?_,?_⟩
  · intro a;rw [invokeReal_memory,source_frame _ c realScalar _ .callerA a (Or.inl (by simp))];rfl
  · intro a;rw [invokeReal_memory,source_frame _ c realScalar _ .callerB a (Or.inl (by simp))];rfl
  · intro id a;rw [invokeReal_memory,source_frame _ c realScalar _ (.other id) a (Or.inl (by simp))];rfl
  · intro a ha;rw [invokeReal_memory,source_frame _ c realScalar _ .output a (Or.inr ha)];rfl
#axiomsClean checked_real
end VeriTile.Bench.Repairs.BmmOptimized.Checked
