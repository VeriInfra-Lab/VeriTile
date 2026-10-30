import bench.tritonbench_g.fast_rms_layernorm.Proofs.RealWeightMemory
import VeriTile.Triton.Memory.TraceProofs

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealWeightSafety
open VeriTile.Triton RealWeightProgram RealWeightOps RealWeightStep
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

structure Bounds (h : Call) (K pid : Nat) (bounds : RegionBounds) : Prop where
  input : ∀ row,row<h.rows → ∀ j : Fin K,pid*K+j.val<h.cols → row*h.cols+(pid*K+j.val)<bounds "x"
  upstream : ∀ row,row<h.rows → ∀ j : Fin K,pid*K+j.val<h.cols → row*h.cols+(pid*K+j.val)<bounds "dx"
  saved : ∀ row,row<h.rows → row<bounds "r"
  output : ∀ j : Fin K,pid*K+j.val<h.cols → pid*K+j.val<bounds "dw"

structure Ready (h : Call) (K pid : Nat) (u : BlockState) : Prop where
  col : u.regs .nat [K] "col"=some (columns K pid)
  mask : u.regs .bool [K] "mask"=some (masks h K pid)
  acc : ∃ a : Fin K → ℝ,u.regs .real [K] "acc"=some (Tile.vec fun j => some (a j))

theorem load_safe (h : Call) (K pid row : Nat) (r : RegionName) (bounds : RegionBounds)
    (u : BlockState) (hc : u.regs .nat [K] "col"=some (columns K pid))
    (hm : u.regs .bool [K] "mask"=some (masks h K pid))
    (hr : u.regs .nat [] "row"=some (Tile.scalar row))
    (inside : ∀ j : Fin K,pid*K+j.val<h.cols → row*h.cols+(pid*K+j.val)<bounds r) :
    (loadOp h K r).SafeAt bounds u := by
  simp [loadOp,offsetOp,colOp,maskOp,Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
    MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,Op.PointerAddressesSafeOn,Op.MemorySafe,
    evalOp,evalOp.eq_def,hc,hm,hr,columns,masks,Option.bind,Tile.bop,Tile.cop,Tile.vec,
    NumericDType.add,NumericDType.mul]
  exact inside

theorem inverse_safe (row : Nat) (bounds : RegionBounds) (u : BlockState)
    (hr : u.regs .nat [] "row"=some (Tile.scalar row)) (inside : row<bounds "r") :
    (Op.load .real (.region "r" (.ref .nat [] "row")) .none).SafeAt bounds u := by
  simp [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe,MaskOpt.Active,Op.PointerAddressesSafeOn,Op.MemorySafe,
    evalOp,evalOp.eq_def,hr,Option.bind,inside]

theorem iteration_safe (h : Call) (K pid row : Nat) (bounds : RegionBounds) (u : BlockState)
    (ready : Ready h K pid u) (inside : Bounds h K pid bounds) (hr : row<h.rows) :
    Stmt.TraceSafeList bounds (loopBody h K) (u.setReg "row" .nat [] (Tile.scalar row)) := by
  rw [(bindings h K).2.1]
  unfold body
  apply TraceProofs.assign_cons (load_safe h K pid row "dx" bounds _
    (by simp [ready.col]) (by simp [ready.mask]) (by simp) (inside.upstream row hr))
  intro dy
  apply TraceProofs.assign_cons (load_safe h K pid row "x" bounds _
    (by simp [ready.col]) (by simp [ready.mask]) (by simp) (inside.input row hr))
  intro x
  apply TraceProofs.assign_cons (inverse_safe row bounds _ (by simp) (inside.saved row hr))
  intro inv
  obtain ⟨acc,ha⟩:=ready.acc
  cases mode : h.gemma
  all_goals
    simp [Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,refOp,mode,
      stepStmt,stepStmts,evalOp,evalOp.eq_def,Option.bind,Tile.bop,NumericDType.add,NumericDType.mul,ha]

theorem next_ready (h : Call) (K pid row : Nat) (u v : BlockState) (ready : Ready h K pid u)
    (run : stepStmts (loopBody h K) (u.setReg "row" .nat [] (Tile.scalar row))=some v) : Ready h K pid v := by
  obtain ⟨acc,ha⟩:=ready.acc
  obtain ⟨w,he,_,_,hc,hm,hacc⟩:=iteration h K pid row u acc ready.col ready.mask ha
  have same : w=v := Option.some.inj (he.symm.trans run)
  subst w
  exact ⟨hc,hm,⟨nextAcc h K pid row u acc,hacc⟩⟩

theorem loop_safe (h : Call) (K pid : Nat) (bounds : RegionBounds) (u : BlockState)
    (ready : Ready h K pid u) (inside : Bounds h K pid bounds) :
    Stmt.forRangeTraceSafe bounds "row" 0 h.rows 1 (loopBody h K) u := by
  apply TraceProofs.range_safe (P:=fun _ s => Ready h K pid s)
    (fun row s hr ready => iteration_safe h K pid row bounds s ready inside hr)
    (fun row s t _ ready he => next_ready h K pid row s t ready he) 0 u ready

#axiomsClean iteration_safe
#axiomsClean loop_safe
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealWeightSafety
