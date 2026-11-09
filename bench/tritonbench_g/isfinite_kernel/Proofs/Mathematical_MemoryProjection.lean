import VeriTile.Triton.Float.ScatterProjection
import bench.tritonbench_g.isfinite_kernel.Proofs.Mathematical_BitBackend

/-! Projection lemmas keep the actual store fold intact during execution.
Readback can subsequently discard registers and inspect only memory. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
open VeriTile.Triton FP.Structural

@[simp] theorem fold_writes_regs {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active]
    (region : ι → RegionName) (offset : ι → Nat) (value : ι → Cell α)
    (s : State α) :
    (xs.foldl (fun t i => if active i then t.write (region i) (offset i) (value i)
      else t) s).regs = s.regs :=
  State.fold_writes_regs xs active region offset value s

theorem fold_writes_mem {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active]
    (region : ι → RegionName) (offset : ι → Nat) (value : ι → Cell α)
    (s : State α) :
    (xs.foldl (fun t i => if active i then t.write (region i) (offset i) (value i)
      else t) s).mem =
    xs.foldl (fun mem i => if active i then
      fun r o => if r = region i ∧ o = offset i then value i else mem r o
      else mem) s.mem :=
  State.fold_writes_mem xs active region offset value s

#axiomsClean fold_writes_regs
#axiomsClean fold_writes_mem
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
