import bench.tritonbench_g.context_attn_nopad.FPBackend

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]
theorem masked_fold_mem_congr {I : Type} (rg : RegionName) (off : I → Nat)
    (v : I → Cell α) (on : I → Prop) [DecidablePred on] (lanes : List I)
    (s t : State α) (h : s.mem = t.mem) :
    (lanes.foldl (fun u i => if on i then u.write rg (off i) (v i) else u) s).mem =
    (lanes.foldl (fun u i => if on i then u.write rg (off i) (v i) else u) t).mem := by
  induction lanes generalizing s t with
  | nil => exact h
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : on i
    · simp only [hi,↓reduceIte]
      apply ih
      simp only [State.write]
      rw [h]
    · simp only [hi,↓reduceIte]
      exact ih s t h


end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
