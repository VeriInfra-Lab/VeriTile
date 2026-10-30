import bench.pretrain_kernels.tasks.local_mtp_roll_packed_fwd.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_mtp_roll_packed_fwd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Contract
open scoped BigOperators Classical

def Allowed (xs : IO.Inputs) : Prop :=
  (∀ (j0 : Fin 5), xs.constant53d3e3c65e j0 = ([0, 16, 48, 80, 128] : List Int).getD (j0.val) 0)

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs (i0,i1) =>
    let input0 := xs.input0
    let constant53d3e3c65e := xs.constant53d3e3c65e
    let vcat : Fin 2 → Fin 16 → ℝ := fun i0 i1 => (if i1.val < 15 then input0 i0 ⟨(((i1.val - 0) + 1)) % 128, Nat.mod_lt _ (by decide)⟩ else 0)
    let vcat_1 : Fin 2 → Fin 32 → ℝ := fun i0 i1 => (if i1.val < 31 then input0 i0 ⟨(((i1.val - 0) + 17)) % 128, Nat.mod_lt _ (by decide)⟩ else 0)
    let vcat_2 : Fin 2 → Fin 32 → ℝ := fun i0 i1 => (if i1.val < 31 then input0 i0 ⟨(((i1.val - 0) + 49)) % 128, Nat.mod_lt _ (by decide)⟩ else 0)
    let vcat_3 : Fin 2 → Fin 48 → ℝ := fun i0 i1 => (if i1.val < 47 then input0 i0 ⟨(((i1.val - 0) + 81)) % 128, Nat.mod_lt _ (by decide)⟩ else 0)
    let vcat_4 : Fin 2 → Fin 128 → ℝ := fun i0 i1 => (if i1.val < 16 then vcat i0 ⟨((i1.val - 0)) % 16, Nat.mod_lt _ (by decide)⟩ else (if i1.val < 48 then vcat_1 i0 ⟨((i1.val - 16)) % 32, Nat.mod_lt _ (by decide)⟩ else (if i1.val < 80 then vcat_2 i0 ⟨((i1.val - 48)) % 32, Nat.mod_lt _ (by decide)⟩ else vcat_3 i0 ⟨((i1.val - 80)) % 48, Nat.mod_lt _ (by decide)⟩)))
    vcat_4 i0 i1

end Contract
end
end VeriTile.Bench.PretrainKernels.local_mtp_roll_packed_fwd
