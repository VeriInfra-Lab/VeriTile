import bench.tritonbench_g.int4_matmul.Proofs.Kernel.GraphResults
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.Int4Matmul.SourceSafety
open VeriTile Triton FP.Structural Safety KernelSource KernelExecution SourceGraph
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Exactly the four masked source reads and active output addresses. -/
structure Bounds (cfg : Parameters) (s : RegisterState (Node α)) (bounds : RegionName → Nat) : Prop where
  input : ∀ k,k<steps cfg → ∀ i : AIndex cfg,
    innerIndex cfg s k i.2.1<cfg.inner → aOffset cfg s k i<bounds cfg.input
  weight : ∀ k,k<steps cfg → ∀ i : BIndex cfg,
    innerIndex cfg s k i.1<cfg.inner → bOffset cfg s k i<bounds cfg.weight
  scale : ∀ k,k<steps cfg → ∀ i : BIndex cfg,
    innerIndex cfg s k i.1<cfg.inner → scaleOffset cfg s k i<bounds cfg.scale
  zero : ∀ k,k<steps cfg → ∀ i : BIndex cfg,
    innerIndex cfg s k i.1<cfg.inner → zeroOffset cfg s k i<bounds cfg.zero
  output : ∀ i : CIndex cfg,active cfg s i → outputOffset cfg s i<bounds cfg.output

theorem prelude_safe (o : Ops α) (rest : Algebra α) (cfg : Parameters)
    (s : RegisterState (Node α)) (bounds : RegionName → Nat) :
    traceSafeList (engine o rest) bounds (prelude cfg) s := by
  simp (config:={maxSteps:=500000}) [prelude,kernel,ComputeKernel.surfaceBody,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,
    Option.bind,State.setReg,numeric,bop,natLt,numericLt,numericEq,integralFloorDiv,integralMod,
    engine,KernelExecution.algebra]

theorem body_safe (o : Ops α) (rest : Algebra α) (cfg : Parameters)
    (initial current : RegisterState (Node α)) (k : Nat)
    (inv : Invariant (ops o) cfg initial k current)
    (bounds : RegionName → Nat) (inside : Bounds cfg initial bounds) (live : k<steps cfg) :
    traceSafeList (engine o rest) bounds (body cfg)
      (current.setReg "k" .nat [] (fun _=>k)) := by
  simp (config:={maxSteps:=500000}) [body,kernel,ComputeKernel.surfaceBody,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,
    Option.bind,State.setReg,numeric,bop,natLt,numericLt,numericEq,integralFloorDiv,integralMod,
    ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,ofFloat,toFloat,
    engine,KernelExecution.algebra,SourceGraph.base,SourceGraph.ops,
    inv.memory,inv.pidm,inv.pidn,inv.offsk,inv.offsn,inv.aptrs,inv.bptrs,inv.acc]
  refine ⟨?_,?_,?_,?_⟩
  · intro i j u h
    exact inside.input k live (i,j,()) h
  · intro i j u h
    exact inside.weight k live (i,j,()) h
  · intro i j h
    exact inside.scale k live (i,j,()) h
  · intro i j h
    exact inside.zero k live (i,j,()) h

theorem loop_safe (o : Ops α) (rest : Algebra α) (cfg : Parameters)
    (initial current : RegisterState (Node α)) (k : Nat)
    (inv : Invariant (ops o) cfg initial k current)
    (bounds : RegionName → Nat) (inside : Bounds cfg initial bounds) :
    rangeSafe (engine o rest) bounds "k" k (steps cfg) 1 (body cfg) current := by
  rw [rangeSafe_unfold]
  simp only [show (1:Nat)≠0 by decide,if_false]
  by_cases live : k<steps cfg
  · simp only [live,if_true]
    refine ⟨body_safe o rest cfg initial current k inv bounds inside live,?_⟩
    have ran := (body_equal o rest cfg _).symm.trans
      (bodyExecution (ops o) (base o rest) cfg initial current k inv).property
    rw [ran]
    exact loop_safe o rest cfg initial _ (k+1) (bodyInvariant _ _ _ _ _ _ inv) bounds inside
  · simp [live]
termination_by steps cfg-k

theorem epilogue_safe (o : Ops α) (rest : Algebra α) (cfg : Parameters)
    (initial current : RegisterState (Node α))
    (inv : Invariant (ops o) cfg initial (steps cfg) current)
    (bounds : RegionName → Nat) (inside : Bounds cfg initial bounds) :
    traceSafeList (engine o rest) bounds (epilogue cfg) current := by
  by_cases split : cfg.splits=1
  all_goals
    simp (config:={maxSteps:=500000}) [epilogue,kernel,ComputeKernel.surfaceBody,
      traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
      safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,
      Option.bind,State.setReg,FP.Structural.store,FP.Structural.atomicAdd,atomicAddition,
      Region.cast,numeric,bop,natLt,numericLt,numericEq,integralFloorDiv,integralMod,
      ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,ofFloat,toFloat,
      engine,KernelExecution.algebra,SourceGraph.base,SourceGraph.ops,
      inv.memory,inv.pidm,inv.pidn,inv.acc,split]
  all_goals
    intro i j hr hc
    simpa only [outputOffset] using inside.output (i,j,()) ⟨hr,hc⟩

theorem complete (o : Ops α) (rest : Algebra α) (cfg : Parameters)
    (initial : RegisterState (Node α)) (bounds : RegionName → Nat) (inside : Bounds cfg initial bounds) :
    TraceSafe (engine o rest) bounds (KernelSource.kernel cfg) initial := by
  unfold TraceSafe
  rw [source_decomposition,List.append_assoc,traceSafeList_append]
  refine ⟨prelude_safe o rest cfg initial bounds,?_⟩
  intro current ran
  have executed := (prelude_equal o rest cfg initial).symm.trans
    (preludeExecution (ops o) (base o rest) cfg initial).property
  rw [executed] at ran
  injection ran with eq
  subst current
  have inv := preludeInvariant (ops o) (base o rest) cfg initial
  apply traceSafeList_cons_intro
  · simpa only [stepSafe_unfold] using loop_safe o rest cfg initial _ 0 inv bounds inside
  · intro final ran
    obtain ⟨next,executed,invFinal⟩ := loopExecution (ops o) (base o rest) cfg initial _ 0 (by omega) inv
    rw [range_equal] at executed
    simp only [FP.Structural.step] at ran
    rw [executed] at ran
    injection ran with eq
    subst final
    exact epilogue_safe o rest cfg initial next invFinal bounds inside

#axiomsClean complete
end
end VeriTile.Bench.Repairs.Int4Matmul.SourceSafety
