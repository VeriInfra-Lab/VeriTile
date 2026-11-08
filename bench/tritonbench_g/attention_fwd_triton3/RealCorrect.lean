import bench.tritonbench_g.attention_fwd_triton3.Proofs.LogicalModelReader

/-! Complete repaired-source Real specification. All six finite input windows,
the padded M/L/O outputs, aliases and ordinary grid are fixed in IO.lean.
Floating arithmetic is interpreted as exact WithBot Real; the signed32 pointer
conversion remains in the mathematical data indices. This makes no IEEE claim.
-/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open scoped BigOperators VeriTile.Triton.LogicalIO
noncomputable section

def maximumScore : List (ℝ × ℝ) → WithBot ℝ
  | []=>⊥
  | x::xs=>(x.1 : WithBot ℝ) ⊔ maximumScore xs

/-- Explicit closed maximum and two sums over the input pairs. Empty blocks
are retained: executing an all-masked block differs from executing no block.
`Real.blockScore` is the Q·K finite sum times scale·log₂(e); its adjacent
mathematical definitions retain the source window, tail and signed32 index. -/
def rowSummary (c : Call) (tile : Nat) (xs : RealInputs.Inputs c.BM c.D c.NK)
    (row : Fin c.BM) (initialValue : ℝ) (value : Nat → Fin c.BN → ℝ) : RealFormula.RowState :=
  let seed:=RealLoopFormula.rowSeed (Real.seed c xs) row initialValue
  let blocks:=(List.range (Real.count c tile)).map (fun n=>
    ((List.finRange c.BN).filter (Real.blockActive c tile n row)).map
      (fun key=>(Real.blockScore c xs tile n row key,value n key)))
  if blocks=[] then seed else
    let pairs:=blocks.flatten
    let m:=seed.center ⊔ maximumScore pairs
    { center:=m
      denominator:=seed.denominator*RealFormula.weight seed.center m+
        (pairs.map (fun pair=>RealFormula.weight (some pair.1) m*1)).sum
      numerator:=seed.numerator*RealFormula.weight seed.center m+
        (pairs.map (fun pair=>RealFormula.weight (some pair.1) m*pair.2)).sum }

def carry (c : Call) (tile : Nat) (xs : RealInputs.Inputs c.BM c.D c.NK) : RealLoopFormula.Carry c.BM c.D where
  center:=fun row=>(rowSummary c tile xs row 0 (fun _ _=>0)).center
  denominator:=fun row=>(rowSummary c tile xs row 0 (fun _ _=>0)).denominator
  numerator:=fun i=>(rowSummary c tile xs i.1 ((Real.seed c xs).numerator i)
    (fun n key=>Real.blockValue c xs tile n (key,i.2.1,()))).numerator

/-- Continuation writes M/L/O. END writes M+log₂(L) and O/L; its L output is
masked out by IO. Mathematical log and division retain their total meanings. -/
def expected (c : Call) (tile : Nat) (xs : RealInputs.Inputs c.BM c.D c.NK) : RealIO.Outputs c.BM c.D :=
  let a:=carry c tile xs
  { center:=fun r=>(a.center r).elim 0 (fun m=>if c.finish≠0 then m+Real.log (a.denominator r)/Real.log 2 else m)
    denominator:=a.denominator
    output:=fun i=>if c.finish≠0 then a.numerator i/a.denominator i.1 else a.numerator i }

def formula (c : Call) (xs : LogicalIO.Inputs c) : LogicalIO.Outputs c where
  center:=fun program row=>(expected c (c.tile program) (xs.programs program)).center row
  denominator:=fun program row=>(expected c (c.tile program) (xs.programs program)).denominator row
  output:=fun program row col=>(expected c (c.tile program) (xs.programs program)).output (row,col,())

private theorem maximum_eq : maximumScore=RealFormula.maxScore := by
  funext pairs
  induction pairs with
  | nil=>rfl
  | cons x xs ih=>simp only [maximumScore,RealFormula.maxScore,ih]

private theorem carry_eq (c : Call) (tile : Nat) (xs : RealInputs.Inputs c.BM c.D c.NK) :
    carry c tile xs=Real.finalCarry c xs tile := by
  unfold carry rowSummary Real.finalCarry Real.closed RealLoopFormula.closed
  rw [maximum_eq]
  rfl

private theorem center_observe (finish : Nat) (m : WithBot ℝ) (mass : ℝ) :
    m.elim 0 (fun x=>if finish≠0 then x+Real.log mass/Real.log 2 else x)=
    (if finish≠0 then m.map (fun x=>x+Real.log mass/Real.log 2) else m).unbotD 0 := by
  cases m <;> by_cases hf : finish≠0 <;> simp [hf,Option.elim,WithBot.unbotD,WithBot.recBotCoe] <;> rfl

private theorem formula_connection (c : Call) : (LogicalIO.io c).lift (formula c)=RealInterface.modelFormula c := by
  funext raw o j
  fin_cases o
  all_goals simp only [LogicalIO.io,VeriTile.Triton.LogicalIO.lift,LogicalIO.inputsEquiv,LogicalIO.outputsEquiv,
    formula,RealInterface.modelFormula,RealInterface.modelOutputLane,RealInterface.modelLaneValue,expected,
    Fin.cases_zero,Fin.cases_succ,carry_eq,Real.laneValue]
  · exact center_observe _ _ _
  · rfl
  · change (if c.finish≠0 then _ else _)=WithBot.unbotD 0 ((some (if c.finish≠0 then _ else _)) : WithBot ℝ)
    rfl

specification correct (c : Call) (valid : Valid c) :
    Spec.Real (LogicalIO.io c ⊨ formula c) :=
  LogicalReader.model_correct c valid _ (formula_connection c)

#axiomsClean correct
#specNonCircular formula avoiding
  [Kernels.source,FP.Structural.exec,FP.Structural.run,Reference.result,Reference.iterate,Reference.gridResult]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast
