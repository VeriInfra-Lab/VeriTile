import bench.tritonbench_g.softmax_optimize.Proofs.IOSafetyBodies
import bench.tritonbench_g.softmax_optimize.Proofs.Reduction
import bench.tritonbench_g.softmax_optimize.Proofs.Mathematical_SafetyLoop
import bench.tritonbench_g.softmax_optimize.Proofs.Mathematical_SafetyGeometry

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.IOSafetyLoops
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SoftmaxOptimize
open SignedTransport
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

def Control (N B row : Nat) (s : BlockState) : Prop :=
  s.regs .int [] "pid_m" = some (Tile.scalar (row : Int)) ∧
  s.regs .int [] "cols" = some (Tile.scalar (N : Int)) ∧
  s.regs .int [] "prev_multiple" = some (Tile.scalar (Config.previous N B : Int))

def Bodies (Y X : RegionName) (N B : Nat) : List (List Stmt) :=
  [Source.inputFull X N B, Source.inputTail X N B,
    Source.outputFull Y X N B, Source.outputTail Y X N B, reductionStmts B]

theorem avoids (Y X : RegionName) (N B : Nat) (body : List Stmt) (member : body ∈ Bodies Y X N B)
    (name : RegName) (hn : name ∈ ["pid_m","cols","prev_multiple"]) :
    ∀ st ∈ body, Safety.AvoidsRegister name st := by
  simp only [Bodies, List.mem_cons, List.not_mem_nil, or_false] at member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hn
  rcases hn with rfl | rfl | rfl
  all_goals rcases member with rfl | rfl | rfl | rfl | rfl <;>
    simp [Source.inputFull,Source.inputTail,Source.outputFull,Source.outputTail,
      Source.indexPrefix,Source.maskStatement,Source.update,sourceMaximum,outputPointer,
      loopAbody,loopBbody,loopApass2body,loopBpass2body,reductionStmts,Safety.AvoidsRegister]

theorem preserves (Y X : RegionName) (N B row : Nat) (body : List Stmt)
    (member : body ∈ Bodies Y X N B) (s t : BlockState) (hc : Control N B row s)
    (run : stepStmts body s = some t) : Control N B row t := by
  have eq (name : RegName) (hn : name ∈ ["pid_m","cols","prev_multiple"]) :=
    Safety.list_preserves_register name .int [] body s t (avoids Y X N B body member name hn) run
  refine ⟨?_,?_,?_⟩
  · rw [eq "pid_m" (by simp)]; exact hc.1
  · rw [eq "cols" (by simp)]; exact hc.2.1
  · rw [eq "prev_multiple" (by simp)]; exact hc.2.2

theorem full_range (bounds : RegionBounds) (Y X : RegionName) (N B row : Nat)
    (hB : 0 < B) (body : List Stmt) (member : body ∈ Bodies Y X N B)
    (safe : ∀ start s, Control N B row s → (∀ i : Fin B, start+i.val < N) →
      Stmt.TraceSafeList bounds body (s.setReg "start_n" .int [] (Tile.scalar (start : Int))))
    (c : Nat) (s : BlockState) (aligned : B ∣ c) (hc : Control N B row s) :
    Stmt.forRangeIntTraceSafe bounds "start_n" (c : Int) (Config.previous N B : Int) B body s := by
  apply Safety.signed_loop_safe bounds "start_n" (Config.previous N B) B body
    (fun start t => B ∣ start ∧ Control N B row t) hB
  · intro start t active inv
    apply safe start t inv.2
    intro i
    exact Safety.full_lane N B start hB active inv.1 i
  · intro start t u active inv run
    refine ⟨inv.1.add (dvd_refl B), ?_⟩
    apply preserves Y X N B row body member _ u _ run
    simpa [Control] using inv.2
  · exact ⟨aligned,hc⟩

theorem tail_range (bounds : RegionBounds) (Y X : RegionName) (N B row : Nat)
    (hB : 0 < B) (body : List Stmt) (member : body ∈ Bodies Y X N B)
    (safe : ∀ (start : Nat) (s : BlockState), Control N B row s →
      Stmt.TraceSafeList bounds body (s.setReg "start_n" .int [] (Tile.scalar (start : Int))))
    (c : Nat) (s : BlockState) (hc : Control N B row s) :
    Stmt.forRangeIntTraceSafe bounds "start_n" (c : Int) (N : Int) B body s := by
  apply Safety.signed_loop_safe bounds "start_n" N B body (fun _ t => Control N B row t) hB
  · intro start t _ inv
    exact safe start t inv
  · intro start t u _ inv run
    apply preserves Y X N B row body member _ u _ run
    simpa [Control] using inv
  · exact hc

def fullLoop (B : Nat) (body : List Stmt) : Stmt :=
  .forRangeInt "start_n" (.constInt 0) (.ref .int [] "prev_multiple") (.constNat B) body
def tailLoop (B : Nat) (body : List Stmt) : Stmt :=
  .forRangeInt "start_n" (.ref .int [] "prev_multiple") (.ref .int [] "cols") (.constNat B) body

theorem full_safe (bounds : RegionBounds) (N B row : Nat) (body : List Stmt)
    (s : BlockState) (hc : Control N B row s)
    (hs : Stmt.forRangeIntTraceSafe bounds "start_n" 0 (Config.previous N B : Int) B body s) :
    Stmt.TraceSafe bounds (fullLoop B body) s := by
  simpa [fullLoop,Stmt.TraceSafe,Op.SafeAt.eq_def,evalOp.eq_def,hc.2.2] using hs

theorem tail_safe (bounds : RegionBounds) (N B row : Nat) (body : List Stmt)
    (s : BlockState) (hc : Control N B row s)
    (hs : Stmt.forRangeIntTraceSafe bounds "start_n" (Config.previous N B : Int) (N : Int) B body s) :
    Stmt.TraceSafe bounds (tailLoop B body) s := by
  simpa [tailLoop,Stmt.TraceSafe,Op.SafeAt.eq_def,evalOp.eq_def,hc.2.1,hc.2.2] using hs

theorem range_preserves (Y X : RegionName) (N B row stop c : Nat) (body : List Stmt)
    (member : body ∈ Bodies Y X N B) (s t : BlockState) (hc : Control N B row s)
    (run : stepForRangeIntAux "start_n" (c : Int) (stop : Int) B body s = some t) :
    Control N B row t := by
  apply Safety.signed_loop_preserves "start_n" stop B body (Control N B row) _ c s t hc run
  intro i u v _ inv executed
  apply preserves Y X N B row body member _ v _ executed
  simpa [Control] using inv

theorem full_preserves (Y X : RegionName) (N B row : Nat) (body : List Stmt)
    (member : body ∈ Bodies Y X N B) (s t : BlockState) (hc : Control N B row s)
    (run : stepStmt (fullLoop B body) s = some t) : Control N B row t := by
  simp only [fullLoop,stepStmt,evalOp.eq_def,hc.2.2,Option.bind_some,Tile.scalar_data] at run
  exact range_preserves Y X N B row (Config.previous N B) 0 body member s t hc run

theorem tail_preserves (Y X : RegionName) (N B row : Nat) (body : List Stmt)
    (member : body ∈ Bodies Y X N B) (s t : BlockState) (hc : Control N B row s)
    (run : stepStmt (tailLoop B body) s = some t) : Control N B row t := by
  simp only [tailLoop,stepStmt,evalOp.eq_def,hc.2.1,hc.2.2,Option.bind_some,Tile.scalar_data] at run
  exact range_preserves Y X N B row N (Config.previous N B) body member s t hc run

#axiomsClean full_range
#axiomsClean tail_range
#axiomsClean full_preserves
#axiomsClean tail_preserves
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.IOSafetyLoops
