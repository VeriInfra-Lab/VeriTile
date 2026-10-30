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
      else t) s).regs = s.regs := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, ih]
    split <;> rfl

theorem fold_writes_mem {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active]
    (region : ι → RegionName) (offset : ι → Nat) (value : ι → Cell α)
    (s : State α) :
    (xs.foldl (fun t i => if active i then t.write (region i) (offset i) (value i)
      else t) s).mem =
    xs.foldl (fun mem i => if active i then
      fun r o => if r = region i ∧ o = offset i then value i else mem r o
      else mem) s.mem := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, ih]
    by_cases hi : active i
    · simp only [hi, ↓reduceIte]
      rfl
    · simp only [hi, ↓reduceIte]

#axiomsClean fold_writes_regs
#axiomsClean fold_writes_mem
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
