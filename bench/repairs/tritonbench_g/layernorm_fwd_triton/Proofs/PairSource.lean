import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.PairSource
open VeriTile.Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Independent register assignments commute when their names differ. -/
theorem setReg_comm {α : Type} (s : State α) (a b : RegName)
    (da db : TileDType) (sha shb : TileShape)
    (va : Values α da sha) (vb : Values α db shb) (different : a ≠ b) :
    (s.setReg a da sha va).setReg b db shb vb =
      (s.setReg b db shb vb).setReg a da sha va := by
  cases s
  unfold State.setReg
  congr 1
  funext d sh n
  by_cases ha : n = a
  · subst n
    simp [State.setReg, different]
  · by_cases hb : n = b
    · subst n
      simp [State.setReg, different, Ne.symm different]
    · simp [State.setReg, ha, hb]

def outputBody (p : Parameters) : List ComputeStmt :=
  match ((candidateKernel p).surfaceBody : List ComputeStmt)[24]? with
  | some (ComputeStmt.forRangeInt _ _ _ _ body) => body
  | _ => []

/-- The weight vector is loaded after x_hat, shortening its lifetime. Every
operand, cast, arithmetic operation, mask and store remains the same. -/
theorem output_body_eq {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (off : Nat) (mem : RegionName → Nat → Cell α)
    (ready : OutputReady M p s t mem) :
    run M (loopBody p 24) (t.setReg "off" .int [] (fun _ => (off : Int))) =
      run M (outputBody p) (t.setReg "off" .int [] (fun _ => (off : Int))) := by
  simp [outputBody, candidateKernel, runtimeSourceCandidate, loopBody, kernel,
    runtimeSource, ComputeKernel.surfaceBody, Random.i32ArangeSize, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, evalOffsets, store, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, numericLt, natLt, Region.cast, Int.toNat_add, Int.toNat_mul, Int.natCast_nonneg, Int.toNat_of_nonpos,
    ready.x, ready.nValue, ready.zeroValue, ready.stepValue, ready.lanes, ready.xStride, ready.yStride, ready.wStride, ready.y, ready.w, ready.mean, ready.rstd]
  rw [setReg_comm _ "w" "x" _ _ _ _ _ _ (by decide)]
  rw [setReg_comm _ "w" "x_hat" _ _ _ _ _ _ (by decide)]

#axiomsClean setReg_comm
#axiomsClean output_body_eq
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.PairSource
