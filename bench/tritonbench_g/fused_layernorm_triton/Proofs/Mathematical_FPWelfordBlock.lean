import bench.tritonbench_g.fused_layernorm_triton.Proofs.SourcePieces
import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_FPWelfordTree
namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def firstLaneValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X : RegionName) (rnumel RBLOCK : Nat) : Fin RBLOCK → WelfordTriple α := fun i =>
  if i.val<rnumel then
    ⟨M.cast (some .fp32) .real .real
      (M.cast (some .fp32) .real .real (M.fp32Load ((s.mem X (i.val+rnumel*s.pids 0)).read .real))),
      M.literal (some .fp32) .real 0,M.literal (some .fp32) .real 1⟩
  else ⟨M.literal (some .fp32) .real 0,M.literal (some .fp32) .real 0,M.literal (some .fp32) .real 0⟩

/-- Index values established by the literal prefix and retained by both loops. -/
structure SourceIndexPins {α : Type} (B : Nat) (t : State α) : Prop where
  zero : t.regs .int [] "zero" = some (fun _ => 0)
  base : t.regs .int [1,B] "rbase" = some (fun i => (i.2.1.val : Int))

structure WelfordLanesReady {α : Type} [Inhabited α] {N : Nat} (s t : State α)
    (values : Fin N → WelfordTriple α) : Prop where
  mem : t.mem=s.mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  mean : t.regs .real [1,N] "tmp3_mean" = some (fun i => (values i.2.1).mean)
  m2 : t.regs .real [1,N] "tmp3_m2" = some (fun i => (values i.2.1).m2)
  weight : t.regs .real [1,N] "tmp3_weight" = some (fun i => (values i.2.1).weight)
  row : t.regs .int [1,1] "x0" = some (fun _ => (s.pids 0 : Int))
  indices : SourceIndexPins N t

/-- A tuple reduction's outputs have shape [1], matching reduce(axis=1) of
[1,RBLOCK]. Each scalar expression is the exact chosen helper tree. -/
def welfordTupleCode {N : Nat} (tree : ReductionTree N) : List ComputeStmt :=
  [ .assign .real [1] "tmp3_tmp" (.compute (.alg .fp32 (.full [1] (treeExpressions tree).mean))),
    .assign .real [1] "tmp4_tmp" (.compute (.alg .fp32 (.full [1] (treeExpressions tree).m2))),
    .assign .real [1] "tmp5_tmp" (.compute (.alg .fp32 (.full [1] (treeExpressions tree).weight))) ]

structure WelfordTupleReady {α : Type} [Inhabited α] (N : Nat) (s t : State α) (values : WelfordTriple α) : Prop where
  mem : t.mem=s.mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  mean : t.regs .real [1] "tmp3_tmp" = some (fun _ => values.mean)
  m2 : t.regs .real [1] "tmp4_tmp" = some (fun _ => values.m2)
  weight : t.regs .real [1] "tmp5_tmp" = some (fun _ => values.weight)
  row : t.regs .int [1,1] "x0" = some (fun _ => (s.pids 0 : Int))
  indices : SourceIndexPins N t

theorem welford_tuple_run {α : Type} [Inhabited α] {N : Nat} (M : Algebra α)
    (equal32 : α→α→Bool) (s t : State α) (values : Fin N→WelfordTriple α)
    (h : WelfordLanesReady s t values) (tree : ReductionTree N) :
    ∃ final, run (withEquality M equal32) (welfordTupleCode tree) t = some final ∧
      WelfordTupleReady N s final (treeValue M equal32 values tree) := by
  let v := treeValue M equal32 values tree
  let t1 := t.setReg "tmp3_tmp" .real [1] (fun _ => v.mean)
  let t2 := t1.setReg "tmp4_tmp" .real [1] (fun _ => v.m2)
  have ht := tree_expressions_eval M equal32 t values h.mean h.m2 h.weight tree
  have ht1 := tree_expressions_eval M equal32 t1 values
    (by simpa [t1,State.setReg] using h.mean)
    (by simpa [t1,State.setReg] using h.m2)
    (by simpa [t1,State.setReg] using h.weight) tree
  have ht2 := tree_expressions_eval M equal32 t2 values
    (by simpa [t2,t1,State.setReg] using h.mean)
    (by simpa [t2,t1,State.setReg] using h.m2)
    (by simpa [t2,t1,State.setReg] using h.weight) tree
  have hfull (u : State α) (e : Op .real []) (x : α)
      (he : evalOp (withEquality M equal32) (some .fp32) e u = some (fun _ => x)) :
      evalExpr (withEquality M equal32) (.compute (.alg .fp32 (.full [1] e))) u = some (fun _ => x) := by
    simp only [evalExpr,evalComputeOp]
    rw [evalOp_unfold]
    simp only [ComputeDType.eraseDType, bind, pure, Option.bind]
    rw [he]
  have e0 := hfull t _ _ ht.1
  have e1 := hfull t1 _ _ ht1.2.1
  have e2 := hfull t2 _ _ ht2.2.2
  simp only [ComputeDType.eraseDType, t2, t1, v] at e0 e1 e2
  refine ⟨t2.setReg "tmp5_tmp" .real [1] (fun _ => v.weight), ?_, ?_⟩
  · simp only [welfordTupleCode, run, step, ComputeDType.eraseDType, e0, e1, e2, bind, pure, Option.bind]
    rfl
  · constructor <;> simp [t2,t1,v,State.setReg,h.mem,h.pids,h.undef,h.row,h.indices.zero,h.indices.base]
    all_goals constructor <;> simp [t2,t1,v,State.setReg,h.indices.zero,h.indices.base]

/-- The literal source tail following the ordered helper tuple. The selected
reference tree and input pointers are outside this tail; no second source
implementation is maintained. -/
def welfordSuffix (Rstd Mean : RegionName) (rnumel : Nat) : List ComputeStmt :=
  (VeriTile.Bench.Repairs.FusedLayernormTriton.SourcePieces.normalization Bool.false
    "X" "W" "Bias" "Y" Mean Rstd 0 rnumel 1 (.leaf ⟨0,by decide⟩)).before.drop 1

def rstdValue {α : Type} (M : Algebra α) (rnumel : Nat) (values : WelfordTriple α) : α :=
  M.unary (some .fp32) .libdeviceRsqrt (M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .div values.m2 (M.fromInt (some .fp32) (rnumel : Int)))
    (M.literal none .real (1e-05 : ℝ)))

def statisticsMemory {α : Type} (M : Algebra α) (s : State α) (Mean Rstd : RegionName)
    (rnumel : Nat) (values : WelfordTriple α) : State α :=
  (s.write Mean (s.pids 0) (.mk .real values.mean)).write Rstd (s.pids 0)
    (.mk .real (rstdValue M rnumel values))

theorem welford_suffix_run {α : Type} [Inhabited α] (M : Algebra α)
    (equal32 : α→α→Bool) (s t : State α) (Mean Rstd : RegionName)
    (rnumel B : Nat) (values : WelfordTriple α) (h : WelfordTupleReady B s t values) :
    ∃ final, run (withEquality M equal32) (welfordSuffix Rstd Mean rnumel) t = some final ∧
      final.mem = (statisticsMemory M s Mean Rstd rnumel values).mem := by
  simp only [welfordSuffix, VeriTile.Bench.Repairs.FusedLayernormTriton.SourcePieces.normalization, VeriTile.Bench.Repairs.FusedLayernormTriton.SourcePieces.statistics, VeriTile.Bench.Repairs.FusedLayernormTriton.SourcePieces.source, VeriTile.Bench.Repairs.FusedLayernormTriton.Kernels.kernel, VeriTile.Bench.Repairs.FusedLayernormTriton.Pasted.originalKernel, Source.intRange?, List.drop, Option.getD, Bool.false_eq_true, ite_false, ComputeKernel.surfaceBody, Option.map, String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp [ run, step, evalExpr, evalComputeOp,
    evalOp_unfold, Option.bind, ComputeDType.eraseDType, numeric, bop, store,
    withEquality, Region.cast, ofFloat, toFloat,h.mean,h.m2,h.weight,h.row,
    TileShape.allIndices,List.finRange_succ,Fin.eq_zero,State.write,statisticsMemory,rstdValue,h.mem]

#axiomsClean welford_tuple_run
#axiomsClean welford_suffix_run
end
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.FPStages
