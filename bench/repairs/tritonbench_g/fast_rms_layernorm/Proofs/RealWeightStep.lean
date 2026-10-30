import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RealWeightEval

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealWeightStep
open VeriTile.Triton RealWeightProgram RealWeightOps
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

def columns (K pid : Nat) : Tile .int [K] := Tile.vec (fun j => ((pid*K+j.val : Nat) : Int))
def masks (h : Call) (K pid : Nat) : Tile .bool [K] := Tile.vec (fun j => decide (pid*K+j.val<h.cols))
def raw (h : Call) (K pid : Nat) (u : BlockState) (role : RegionName) (row : Nat) (j : Fin K) : ℝ :=
  if pid*K+j.val<h.cols then u.readMem role (row*h.cols+(pid*K+j.val)) else 0
def saved (u : BlockState) (row : Nat) : ℝ := u.readMem "r" row

def initial (h : Call) (K : Nat) (u : BlockState) : BlockState :=
  ((u.setReg "col" .int [K] (columns K (u.pids 0))).setReg "mask" .bool [K] (masks h K (u.pids 0))).setReg
    "acc" .real [K] (Tile.vec fun _ => some 0)

theorem prefix_executes (h : Call) (K : Nat) (u : BlockState) :
    stepStmts (prefixCode h K) u=some (initial h K u) := by
  rw [(bindings h K).1]
  simp [initOps,colOp,stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.vec,
    NumericDType.add,NumericDType.mul,ComparableDType.lt,
    FloatDType.cast,FloatDType.ofWithBot,FloatDType.toWithBot,columns,masks,initial,
    -Nat.cast_mul,-Nat.cast_add,← Int.natCast_mul,← Int.natCast_add]

def nextAcc (h : Call) (K pid row : Nat) (u : BlockState) (acc : Fin K → ℝ) : Fin K → ℝ :=
  fun j => acc j+raw h K pid u "dx" row j*(raw h K pid u "x" row j*saved u row)

def mid (h : Call) (K pid row : Nat) (u : BlockState) : BlockState :=
  let v:=u.setReg "dy" .real [K] (Tile.vec fun j => some (raw h K pid u "dx" row j))
  let v:=v.setReg "x" .real [K] (Tile.vec fun j => some (raw h K pid u "x" row j))
  let v:=v.setReg "inv" .real [] (Tile.scalar (some (saved u row)))
  v.setReg "normed" .real [K] (Tile.vec fun j => some (raw h K pid u "x" row j*saved u row))
def beforeAcc (h : Call) (K pid row : Nat) (u : BlockState) : BlockState :=
  if h.gemma then mid h K pid row u else
    (mid h K pid row u).setReg "normed" .real [K]
      (Tile.vec fun j => some (raw h K pid u "x" row j*saved u row))
def result (h : Call) (K pid row : Nat) (u : BlockState) (acc : Fin K → ℝ) : BlockState :=
  (beforeAcc h K pid row u).setReg "acc" .real [K] (Tile.vec fun j => some (nextAcc h K pid row u acc j))

theorem iteration_run (h : Call) (K pid row : Nat) (u : BlockState) (acc : Fin K → ℝ)
    (hcol : u.regs .int [K] "col"=some (columns K pid))
    (hmask : u.regs .bool [K] "mask"=some (masks h K pid))
    (hrow : u.regs .int [] "row"=some (Tile.scalar (row : Int)))
    (hacc : u.regs .real [K] "acc"=some (Tile.vec fun j => some (acc j))) :
    stepStmts (loopBody h K) u=some (result h K pid row u acc) := by
  rw [(bindings h K).2.1]
  unfold body
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (RealWeightEval.load h K pid row "dx" u hcol hmask hrow))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (RealWeightEval.load h K pid row "x" _ (by simp [hcol,columns]) (by simp [hmask,masks]) (by simp [hrow])))]
  simp only [BlockState.setReg_readMem]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (RealWeightEval.inverse row _ (by simp [hrow])))]
  simp only [BlockState.setReg_readMem]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (RealWeightEval.normed K
    (raw h K pid u "x" row) (saved u row) _ (by simp [raw]) (by simp [saved])))]
  have branch : stepStmt (.ifThen (.boolNot (.constBool h.gemma)) [.assign .real [K] "normed" (refOp K "normed")])
      (mid h K pid row u)=some (beforeAcc h K pid row u) := by
    cases mode : h.gemma <;> simp [stepStmt,stepStmts,evalOp,evalOp_ref,refOp,beforeAcc,mid,mode]
  change stepStmts [.ifThen _ _,.assign _ _ _ _] (mid h K pid row u)=_
  rw [stepStmts.cons_some branch]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (RealWeightEval.accumulation K acc
    (raw h K pid u "dx" row) (fun j => raw h K pid u "x" row j*saved u row) _
    (by cases mode : h.gemma <;> simp [beforeAcc,mid,mode,hacc])
    (by cases mode : h.gemma <;> simp [beforeAcc,mid,mode])
    (by cases mode : h.gemma <;> simp [beforeAcc,mid,mode])))]
  simp [stepStmts,result,nextAcc]

theorem iteration (h : Call) (K pid row : Nat) (u : BlockState) (acc : Fin K → ℝ)
    (hcol : u.regs .int [K] "col"=some (columns K pid))
    (hmask : u.regs .bool [K] "mask"=some (masks h K pid))
    (hacc : u.regs .real [K] "acc"=some (Tile.vec fun j => some (acc j))) :
    ∃ v,stepStmts (loopBody h K) (u.setReg "row" .int [] (Tile.scalar (row : Int)))=some v ∧
      v.mem=u.mem ∧ v.pids=u.pids ∧
      v.regs .int [K] "col"=some (columns K pid) ∧
      v.regs .bool [K] "mask"=some (masks h K pid) ∧
      v.regs .real [K] "acc"=some (Tile.vec fun j => some (nextAcc h K pid row u acc j)) := by
  let bound:=u.setReg "row" .int [] (Tile.scalar (row : Int))
  refine ⟨result h K pid row bound acc,
    iteration_run h K pid row bound acc (by simp [bound,hcol]) (by simp [bound,hmask])
      (by simp [bound]) (by simp [bound,hacc]),?_,?_,?_,?_,?_⟩
  all_goals cases mode : h.gemma <;> simp [result,beforeAcc,mid,mode,bound,hcol,hmask,nextAcc,raw,saved] <;> rfl

#axiomsClean prefix_executes
#axiomsClean iteration
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealWeightStep
