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
  TypedSource.groupedBackwardCandidateKernel "DX" "DY" "DW" "DB" "X" "W" "Mean" "Rstd" stride M N G B

def body (stride M N G B : Nat) : List ComputeStmt :=
  match ((kernel stride M N G B).surfaceBody : List ComputeStmt)[11]? with
  | some (ComputeStmt.forRangeInt _ _ _ _ body) => body
  | _ => []

theorem split (stride M N G B : Nat) : (kernel stride M N G B).surfaceBody =
    SourceGrouped.prelude stride M N G B ++
      [.forRangeInt "row" (.ref .int [] "row") (.castNatToInt (.constNat M)) (.constNat G) (body stride M N G B)] ++
      SourceGrouped.suffix stride M N G B := by rfl

def commonPrefix (stride M N G B : Nat) := (SourceGrouped.body stride M N G B).take 4
def weightLoad (stride M N G B : Nat) := ((SourceGrouped.body stride M N G B).drop 4).take 2
def statistics (stride M N G B : Nat) := ((SourceGrouped.body stride M N G B).drop 6).take 3
def commonSuffix (stride M N G B : Nat) := (SourceGrouped.body stride M N G B).drop 9

theorem source_body_split (stride M N G B : Nat) : SourceGrouped.body stride M N G B =
    commonPrefix stride M N G B ++ (weightLoad stride M N G B ++ statistics stride M N G B) ++
      commonSuffix stride M N G B := by rfl
theorem candidate_body_split (stride M N G B : Nat) : body stride M N G B =
    commonPrefix stride M N G B ++ (statistics stride M N G B ++ weightLoad stride M N G B) ++
      commonSuffix stride M N G B := by rfl

theorem load_statistics_commute (m : Model α) (s : State α) (stride M N G B : Nat) :
    run (engine m) (statistics stride M N G B ++ weightLoad stride M N G B) s =
      run (engine m) (weightLoad stride M N G B ++ statistics stride M N G B) s := by
  cases hx : s.regs .real [B] "x" <;> cases hr : s.regs .int [] "row" <;>
    cases hc : s.regs .int [B] "cols" <;> cases hm : s.regs .bool [B] "mask" <;>
    simp [statistics,weightLoad,SourceGrouped.body,SourceGrouped.code,
      TypedSource.groupedBackwardKernel,groupedBackwardKernel,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,engine,numeric,natLt,numericLt,
      numericLe,numericEq,bop,FP.Structural.store,ComputeDType.eraseDType,
      TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
      Region.cast,Option.bind,ofFloat,toFloat,hx,hr,hc,hm]
  all_goals
    repeat first
      | rw [setReg_comm _ "w" "mean" _ _ _ _ _ _ (by decide)]
      | rw [setReg_comm _ "w" "rstd" _ _ _ _ _ _ (by decide)]
      | rw [setReg_comm _ "w" "xhat" _ _ _ _ _ _ (by decide)]
      | rw [setReg_comm _ "__veritile_load_2" "mean" _ _ _ _ _ _ (by decide)]
      | rw [setReg_comm _ "__veritile_load_2" "rstd" _ _ _ _ _ _ (by decide)]
      | rw [setReg_comm _ "__veritile_load_2" "xhat" _ _ _ _ _ _ (by decide)]

theorem body_eq (m : Model α) (s t : State α) (stride M N G B group row : Nat)
    (a : Carries α B) (h : Invariant s N B group a t) :
    run (engine m) (body stride M N G B) (t.setReg "row" .int [] (fun _ => (row : Int))) =
      run (engine m) (SourceGrouped.body stride M N G B) (t.setReg "row" .int [] (fun _ => (row : Int))) := by
  rw [source_body_split,candidate_body_split,run_append,run_append,run_append,run_append]
  cases ran : run (engine m) (commonPrefix stride M N G B) (t.setReg "row" .int [] (fun _ => (row : Int))) with
  | none => rfl
  | some middle =>
    simp only [Option.bind_some]
    rw [load_statistics_commute]
#axiomsClean body_eq
end
end VeriTile.Bench.Repairs.LayerNormTriton.PairGrouped
