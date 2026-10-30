import bench.tritonbench_g.layer_norm_triton.Proofs.SourceGroupedSafety

namespace VeriTile.Bench.Repairs.LayerNormTriton.PairGrouped
open VeriTile.Triton FP.Structural SourceModel SourceGroupedGraph
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

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

def kernel (stride M N G B : Nat) :=
  TypedSource.groupedBackwardCandidate "DX" "DY" "DW" "DB" "X" "W" "Mean" "Rstd" stride M N G B

def body (stride M N G B : Nat) : List ComputeStmt :=
  match ((kernel stride M N G B).surfaceBody : List ComputeStmt)[7]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
  | _ => []

theorem split (stride M N G B : Nat) : (kernel stride M N G B).surfaceBody =
    SourceGrouped.prelude stride M N G B ++
      [.forRangeDyn "row" (.ref .nat [] "group") (.constNat M) (.constNat G) (body stride M N G B)] ++
      SourceGrouped.suffix stride M N G B := by rfl

theorem body_eq (m : Model α) (s t : State α) (stride M N G B group row : Nat)
    (a : Carries α B) (h : Invariant s N B group a t) :
    run (engine m) (body stride M N G B) (t.setReg "row" .nat [] (fun _ => row)) =
      run (engine m) (SourceGrouped.body stride M N G B) (t.setReg "row" .nat [] (fun _ => row)) := by
  obtain ⟨hmem,hu,hgroup,hcols,hmask,hdwp,hdbp,hdw,hdb⟩ := h
  simp [body,kernel,TypedSource.groupedBackwardCandidate,SourceGrouped.body,SourceGrouped.code,
    TypedSource.groupedBackward,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,
    engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
    FP.Structural.store,ComputeDType.eraseDType,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    Region.cast,Option.bind,ofFloat,toFloat,hgroup,hcols,hmask,hdwp,hdbp,hdw,hdb]
  rw [setReg_comm _ "xhat" "w" _ _ _ _ _ _ (by decide)]
  rw [setReg_comm _ "rstd" "w" _ _ _ _ _ _ (by decide)]
  rw [setReg_comm _ "mean" "w" _ _ _ _ _ _ (by decide)]

#axiomsClean body_eq
end
end VeriTile.Bench.Repairs.LayerNormTriton.PairGrouped
