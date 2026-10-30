import bench.tritonbench_g.pow_scalar_tensor.Proofs.TileSafety
import bench.tritonbench_g.pow_scalar_tensor.Proofs.Kernel

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.SourceSafety
open VeriTile Triton FP.Structural Safety
open VeriTile.Bench.TritonBenchG.PowScalarTensor WholeTile
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
noncomputable section
set_option maxHeartbeats 300000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def tileAssign : ComputeStmt := .assign .nat [] "tile_id" (.alg
  (.add .nat .nil (.ref .nat [] "pid") (.mul .nat .nil (.ref .nat [] "j") (.ref .nat [] "num_ctas"))))
def loopBody (c : Parameters) (candidate : Bool) := tileAssign::perTile c candidate

theorem loop_safe (M : Algebra α) (c : Parameters) (candidate : Bool) (bounds : RegionName → Nat)
    (hi : ∀ i,i<c.n → (c.input.base+(i:Int)*c.input.stride).toNat<bounds c.input.region)
    (ho : ∀ i,i<c.n → (c.output.base+(i:Int)*c.output.stride).toNat<bounds c.output.region)
    (P C : Nat) : ∀ remaining cur,cur+remaining=c.tiles → ∀ (s : State α),
    s.regs .nat [] "pid"=some (fun _ => P) → s.regs .nat [] "num_ctas"=some (fun _ => C) →
    rangeSafe M bounds "j" cur c.tiles 1 (loopBody c candidate) s := by
  intro remaining
  induction remaining with
  | zero =>
    intro cur ends s hp hc
    rw [rangeSafe_unfold,if_neg (by decide),if_neg (by omega)]
    trivial
  | succ remaining ih =>
    intro cur ends s hp hc
    let atJ := s.setReg "j" .nat [] (fun _ => cur)
    let atTile := atJ.setReg "tile_id" .nat [] (fun _ => P+cur*C)
    have assignRun : step M tileAssign atJ=some atTile := by
      simp only [tileAssign,step,evalExpr,evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,
        atJ,atTile,State.setReg,hp,hc,numeric,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
        ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,dite_true,dite_false,ite_true,ite_false]
      rfl
    have ready : atTile.regs .nat [] "tile_id"=some (fun _ => P+cur*C) := by simp [atTile]
    obtain ⟨next,ran,_,preserveP,preserveC⟩ := tile_implementation M c candidate (P+cur*C) atTile ready
    have bodyRun : run M (loopBody c candidate) atJ=some next := by
      simp only [loopBody,run,assignRun,Option.bind_eq_bind,Option.bind_some,ran]
    rw [rangeSafe_unfold,if_neg (by decide),if_pos (by omega)]
    refine ⟨?_,?_⟩
    · refine traceSafeList_cons_of_step ?_ assignRun ?_
      · simp only [tileAssign,stepSafe_unfold,safeExpr_unfold,safeOp_unfold,and_true]
      · exact tile_safe M c candidate atTile (P+cur*C) ready bounds
          (fun j active => hi _ active) (fun j active => ho _ active)
    · rw [bodyRun]
      apply ih (cur+1) (by omega) next
      · rw [preserveP];simp [atTile,atJ,hp]
      · rw [preserveC];simp [atTile,atJ,hc]

def sourcePrefix (c : Parameters) (candidate : Bool) := (sourceKernel c candidate).surfaceBody.take 3

def prepared (c : Parameters) (s : State α) : State α :=
  let t := s.setReg "pid" .nat [] (fun _ => s.pids 0)
  let t := t.setReg "num_tiles0" .nat [] (fun _ => (c.n+c.B-1)/c.B)
  if c.oneTile then t.setReg "tile_id" .nat [] (fun _ => s.pids 0)
  else t.setReg "num_ctas" .nat [] (fun _ => s.numPids 0)

theorem prefix_run (M : Algebra α) (c : Parameters) (candidate : Bool) (s : State α) :
    run M (sourcePrefix c candidate) s=some (prepared c s) := by
  cases candidate <;> cases branch : c.oneTile
  all_goals simp only [sourcePrefix,sourceKernel,kernel,commutedKernel,branch,gridStrideKernel,gridStrideCommuted,
    oneTileKernel,oneTileCommuted,ComputeKernel.surfaceBody,List.take,
    Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,
    numeric,FP.Structural.bop,prepared,Broadcast.leftIndex,Broadcast.rightIndex,
    State.setReg,ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
    dite_true,dite_false,ite_true,ite_false]
  all_goals rfl

theorem prefix_safe (M : Algebra α) (c : Parameters) (candidate : Bool)
    (bounds : RegionName → Nat) (s : State α) : traceSafeList M bounds (sourcePrefix c candidate) s := by
  cases candidate <;> cases branch : c.oneTile
  all_goals simp only [sourcePrefix,sourceKernel,kernel,commutedKernel,branch,gridStrideKernel,gridStrideCommuted,
    oneTileKernel,oneTileCommuted,ComputeKernel.surfaceBody,List.take,
    Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeOp_unfold,
    step,evalExpr,evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,
    State.setReg,numeric,FP.Structural.bop,and_true,true_and,
    ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
    dite_true,dite_false,ite_true,ite_false]

theorem source_safe (M : Algebra α) (c : Parameters) (candidate : Bool) (bounds : RegionName → Nat)
    (hi : ∀ i,i<c.n → (c.input.base+(i:Int)*c.input.stride).toNat<bounds c.input.region)
    (ho : ∀ i,i<c.n → (c.output.base+(i:Int)*c.output.stride).toNat<bounds c.output.region)
    (s : State α) : FP.Structural.TraceSafe M bounds (sourceKernel c candidate) s := by
  unfold FP.Structural.TraceSafe
  rw [←List.take_append_drop 3 (sourceKernel c candidate).surfaceBody,traceSafeList_append]
  refine ⟨prefix_safe M c candidate bounds s,?_⟩
  intro t ran
  have same : t=prepared c s := Option.some.inj (ran.symm.trans (prefix_run M c candidate s))
  subst t
  cases branch : c.oneTile
  · have code : (sourceKernel c candidate).surfaceBody.drop 3=
        [.forRangeDyn "j" (.constNat 0) (.constNat c.tiles) (.constNat 1) (loopBody c candidate)] := by
      cases candidate <;> simp only [sourceKernel,kernel,commutedKernel,branch,gridStrideKernel,gridStrideCommuted,
        oneTileKernel,oneTileCommuted,perTile,loopBody,tileAssign,ComputeKernel.surfaceBody,List.drop,
        Bool.false_eq_true,ite_false,ite_true]
    rw [code]
    apply traceSafeList_cons_intro
    · simp only [stepSafe_unfold,safeOp_unfold,evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,true_and]
      apply loop_safe M c candidate bounds hi ho (s.pids 0) (s.numPids 0) c.tiles 0 (by omega)
      · simp [prepared,branch]
      · simp [prepared,branch]
    · intro t _;exact traceSafeList_nil _ _ _
  · have code : (sourceKernel c candidate).surfaceBody.drop 3=perTile c candidate := by
      cases candidate <;> simp only [sourceKernel,kernel,commutedKernel,branch,perTile,Bool.false_eq_true,ite_false,ite_true]
    rw [code]
    exact tile_safe M c candidate (prepared c s) (s.pids 0) (by simp [prepared,branch]) bounds
      (fun j active => hi _ active) (fun j active => ho _ active)

#axiomsClean source_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.SourceSafety
