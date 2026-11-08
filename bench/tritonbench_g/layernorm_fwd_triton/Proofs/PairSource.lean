import bench.tritonbench_g.layernorm_fwd_triton.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.PairSource
open VeriTile.Triton FP.Structural Source
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_mem fold_writes_regs)
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

/-- The scheduled load has its own successful final state. Both versions
follow the same ordered arithmetic and update the same complete memory;
generated load temporaries are private and need not contain the same values. -/
theorem candidate_output_body_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (off : Nat) (mem : RegionName → Nat → Cell α)
    (hx : p.x ≠ p.y) (hw : p.weight ≠ p.y) (h : OutputReady M p s t mem) :
    ∃ u, run M (outputBody p) (t.setReg "off" .int [] (fun _ => (off : Int))) = some u ∧
      OutputReady M p s u (outputStep M p s off mem) := by
  simp [outputBody, candidateKernel, runtimeSourceCandidateKernel, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, evalOffsets, store, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, numericLt, natLt, Region.cast,
    h.integers.1, h.integers.2.1, h.integers.2.2, h.readX, h.readW, h.pids, h.undef, h.x, h.y, h.w, h.mean, h.rstd]
  constructor
  · simp only [fold_writes_mem, State.setReg_mem, h.memory]
    rfl
  · funext o
    rw [State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inl hx)]
    simp [h.readX]
  · funext o
    rw [State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inl hw)]
    simp [h.readW]
  · simp [h.pids]
  · simp only [fold_writes_undef, State.setReg_undef, h.undef]
  · simp [fold_writes_regs, h.x]
  · simp [fold_writes_regs, h.y]
  · simp [fold_writes_regs, h.w]
  · simp [fold_writes_regs, h.mean]
  · simp [fold_writes_regs, h.rstd]
  · simp [integerSetup, fold_writes_regs, h.integers.1, h.integers.2.1, h.integers.2.2]

#axiomsClean setReg_comm
#axiomsClean candidate_output_body_run
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.PairSource
