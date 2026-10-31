import bench.tritonbench_g.rms_rbe_matmul.Proofs.KernelRotation

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem fold_memory_congr {ι : Type} (xs : List ι) (region : RegionName)
    (off : ι → Nat) (mask : ι → Prop) [DecidablePred mask] (value : ι → Cell α)
    (s t : State α) (mem : s.mem=t.mem) :
    (xs.foldl (fun u i => if mask i then u.write region (off i) (value i) else u) s).mem=
    (xs.foldl (fun u i => if mask i then u.write region (off i) (value i) else u) t).mem := by
  induction xs generalizing s t with
  | nil => exact mem
  | cons i xs ih =>
    simp only [List.foldl_cons]
    by_cases hm : mask i
    · simp only [if_pos hm]
      exact ih _ _ (by simp only [State.write,mem])
    · simpa only [if_neg hm] using ih s t mem

theorem store_run (a : Operations α) (fp8 : Bool) (p : Parameters)
    (s t : State α) (values : TileIndex [16,64] → α) (h : TailReady p s t values) :
    ∃ u, run (engine a) (stores fp8 p) t=some u ∧
      u.mem=(scatter a fp8 p s s values).mem := by
  cases fp8
  all_goals
    simp [stores,kernel,TypedSource.kernel,TypedSource.source,arguments,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
      Region.cast,ofFloat,toFloat,State.setReg,store,
      engine,Algebra.withDefaultPrecision,resolvePrecision,
      h.product,h.batch,h.tileM,h.tileN]
  all_goals simp [scatter,active,outputOffset,outputCell,outputDType,
    ofFloat,decide_eq_true_eq,Bool.and_eq_true,row,col]
  all_goals apply fold_memory_congr
  all_goals exact h.mem

#axiomsClean store_run
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource
