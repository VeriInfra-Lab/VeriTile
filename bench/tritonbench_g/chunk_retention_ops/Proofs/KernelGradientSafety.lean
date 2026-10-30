import bench.tritonbench_g.chunk_retention_ops.Sources.KernelGradientProgram
import bench.tritonbench_g.chunk_retention.Proofs.PureAccessSafety
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelGradient
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open KernelBackward (Parameters PID Matrix)
open VeriTile.Bench.Repairs.ChunkLinearAttn (valueBlocks)
open VeriTile.Bench.Repairs.ChunkRetention.Whole001
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false
attribute [local simp 2000] AccessSafety.two_offset_pointer

private theorem entry_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (scale : ℝ) (s : State α) (bounds : RegionName→Nat)
    (hq : ∀ e t,e<p.K → t<p.T → s.pids 2*(p.T*p.K)+e+t*p.K<bounds "q")
    (hk : ∀ t e,t<p.T → e<p.K → s.pids 2*(p.T*p.K)+t*p.K+e<bounds "k") :
    Safety.traceSafeList (engine (withDot M D)) bounds (gradientParts p scale).before s := by
  simp [gradientParts,kernel,Source.backwardQKV,ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,
      Triton.Source.prefixBefore,Option.map,Option.getD,AccessSafety.singleton_safe,AccessSafety.pair_safe,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      Broadcast.leftIndex,Broadcast.rightIndex]
  constructor
  · intro e t he ht; exact hq _ _ he ht
  · intro t e ht he; exact hk _ _ ht he

private theorem body_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (scale : ℝ) (pid : PID) (heads : Nat) (input : Memory α) (s : State α) (vb : Nat)
    (a : GradAcc p α) (ctx : GradientReady p pid heads input s)
    (fixed : GradientFixed M D p pid input scale s) (ha : HasGrad p a s) (bounds : RegionName→Nat)
    (hv : ∀ t j,t<p.T → j<p.V → pid.z*(p.T*p.V)+t*p.V+j<bounds "v")
    (hg : ∀ t j,t<p.T → j<p.V → pid.z*(p.T*p.V)+t*p.V+j<bounds "incoming")
    (hh : ∀ j e,j<p.V → e<p.K → pid.z*(p.NT*p.K*p.V)+pid.y*p.K*p.V+j+e*p.V<bounds "h")
    (hdh : ∀ e j,e<p.K → j<p.V → pid.z*(p.NT*p.K*p.V)+pid.y*p.K*p.V+e*p.V+j<bounds "dhistory")
    (hdv : ∀ i,gradValueActive p pid vb i → gradValueOffset p pid heads vb i<bounds "valuePartials") :
    Safety.traceSafeList (engine (withDot M D)) bounds (gradientParts p scale).body
      (s.setReg "i_v" .nat [] (fun _=>vb)) := by
  have hn : Int.toNat (((pid.x:Int)*heads+pid.z)*((p.T:Int)*p.V))=
      (pid.x*heads+pid.z)*(p.T*p.V) := by norm_cast
  simp [gradientParts,kernel,Source.backwardQKV,ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,
      Triton.Source.prefixBefore,Option.map,Option.getD,AccessSafety.singleton_safe,AccessSafety.pair_safe,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      Broadcast.leftIndex,Broadcast.rightIndex,ctx.ik,ctx.it,ctx.bh,ctx.nh,
      fixed.dq,fixed.dk,fixed.ds,fixed.query,fixed.key,fixed.score,
      ha.query,ha.key,ha.score,FP.Structural.store,hn]
  refine ⟨?_,?_,?_,?_,?_⟩
  · intro t j ht hj; exact hv _ _ ht hj
  · intro t j ht hj; exact hg _ _ ht hj
  · intro j e hj he; exact hh _ _ hj he
  · intro e j he hj; exact hdh _ _ he hj
  · intro t j ht hj; exact hdv (t,j,PUnit.unit) ⟨ht,hj⟩

private theorem finish_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (scale : ℝ) (pid : PID) (heads : Nat) (input : Memory α) (s : State α)
    (a : GradAcc p α) (ctx : GradientReady p pid heads input s)
    (fixed : GradientFixed M D p pid input scale s) (ha : HasGrad p a s) (bounds : RegionName→Nat)
    (hq : ∀ i,gradKeyActive p pid i → gradKeyOffset p pid i<bounds "queryGradient")
    (hk : ∀ i,gradKeyActive p pid i → gradKeyOffset p pid i<bounds "keyGradient") :
    Safety.traceSafeList (engine (withDot M D)) bounds (gradientParts p scale).after s := by
  simp [gradientParts,kernel,Source.backwardQKV,ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,
      Triton.Source.prefixBefore,Option.map,Option.getD,AccessSafety.singleton_safe,AccessSafety.pair_safe,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      Broadcast.leftIndex,Broadcast.rightIndex,ctx.ik,ctx.it,ctx.bh,
      fixed.dq,fixed.dk,fixed.ds,fixed.query,fixed.key,fixed.score,
      ha.query,ha.key,ha.score,FP.Structural.store]
  constructor
  · intro t e ht he; exact hq (t,e,PUnit.unit) ⟨ht,he⟩
  · intro t e ht he; exact hk (t,e,PUnit.unit) ⟨ht,he⟩

/-- Complete actual source safety for the gradient stage, including every
value-block partial store and both final Q/K gradients. No PID restriction is
introduced: capacities describe the actual source addresses at that PID. -/
theorem source_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (scale : ℝ) (s : State α) (hBT : p.BT≤2^31) (bounds : RegionName→Nat)
    (hq : ∀ e t,e<p.K → t<p.T → s.pids 2*(p.T*p.K)+e+t*p.K<bounds "q")
    (hk : ∀ t e,t<p.T → e<p.K → s.pids 2*(p.T*p.K)+t*p.K+e<bounds "k")
    (hv : ∀ t j,t<p.T → j<p.V → s.pids 2*(p.T*p.V)+t*p.V+j<bounds "v")
    (hg : ∀ t j,t<p.T → j<p.V → s.pids 2*(p.T*p.V)+t*p.V+j<bounds "incoming")
    (hh : ∀ j e,j<p.V → e<p.K → s.pids 2*(p.NT*p.K*p.V)+s.pids 1*p.K*p.V+j+e*p.V<bounds "h")
    (hdh : ∀ e j,e<p.K → j<p.V → s.pids 2*(p.NT*p.K*p.V)+s.pids 1*p.K*p.V+e*p.V+j<bounds "dhistory")
    (hdv : ∀ vb,vb<valueBlocks p → ∀ i,gradValueActive p ⟨s.pids 0,s.pids 1,s.pids 2⟩ vb i →
      gradValueOffset p ⟨s.pids 0,s.pids 1,s.pids 2⟩ (s.numPids 2) vb i<bounds "valuePartials")
    (hdq : ∀ i,gradKeyActive p ⟨s.pids 0,s.pids 1,s.pids 2⟩ i →
      gradKeyOffset p ⟨s.pids 0,s.pids 1,s.pids 2⟩ i<bounds "queryGradient")
    (hdk : ∀ i,gradKeyActive p ⟨s.pids 0,s.pids 1,s.pids 2⟩ i →
      gradKeyOffset p ⟨s.pids 0,s.pids 1,s.pids 2⟩ i<bounds "keyGradient") :
    TraceSafe (engine (withDot M D)) bounds (kernel p scale) s := by
  let pid : PID := ⟨s.pids 0,s.pids 1,s.pids 2⟩
  let start := gradientEntry M D p scale s
  have ready := gradientEntry_values M D p scale s hBT
  let I := fun (_ : Nat) (t : State α) => GradientReady p pid (s.numPids 2) s.mem t ∧
    GradientFixed M D p pid s.mem scale t ∧ ∃ a,HasGrad p a t
  have safeLoop : Safety.rangeSafe (engine (withDot M D)) bounds "i_v" 0 (valueBlocks p) 1
      (gradientParts p scale).body start.val := by
    apply Safety.rangeSafe_of_invariant _ bounds "i_v" (valueBlocks p) 1 _ I
    · intro n hn t inv
      obtain ⟨a,ha⟩ := inv.2.2
      exact body_safe M D p scale pid (s.numPids 2) s.mem t n a inv.1 inv.2.1 ha bounds hv hg hh hdh (hdv n hn)
    · intro n _ t inv u ran
      obtain ⟨a,ha⟩ := inv.2.2
      let next := gradientBody M D p scale pid (s.numPids 2) s.mem t n a inv.1 inv.2.1 ha
      have kept := gradientBody_values M D p scale pid (s.numPids 2) s.mem t n a inv.1 inv.2.1 ha
      have same := Option.some.inj (next.property.symm.trans ran)
      subst u
      exact ⟨kept.1,kept.2.1,_,kept.2.2⟩
    · exact ⟨ready.1,ready.2.1,_,ready.2.2.1⟩
  obtain ⟨mid,hm,ctx,fixed,ha,_⟩ := gradientLoop M D p scale pid (s.numPids 2) s.mem start.val
    ready.1 ready.2.1 ready.2.2.1
  have h0 : (gradientParts p scale).start=.constNat 0 := rfl
  have hn : (gradientParts p scale).stop=.div .nat .nil
    (.sub .nat .nil (.add .nat .nil (.constNat p.V) (.constNat p.BV)) (.constNat 1)) (.constNat p.BV) := rfl
  have h1 : (gradientParts p scale).stride=.constNat 1 := rfl
  have loopRun : step (engine (withDot M D))
      (.forRangeDyn "i_v" (gradientParts p scale).start (gradientParts p scale).stop
        (gradientParts p scale).stride (gradientParts p scale).body) start.val=some mid := by
    simpa [step,h0,hn,h1,evalOp_unfold,Option.bind,numeric,bop,valueBlocks] using hm
  unfold TraceSafe
  rw [gradient_parts,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨entry_safe M D p scale s bounds hq hk,?_⟩
  intro t ran
  rw [start.property] at ran
  cases ran
  apply Safety.traceSafeList_cons_of_step
    (by simpa [Safety.stepSafe_unfold,Safety.safeOp_unfold,h0,hn,h1,evalOp_unfold,
      numeric,bop,Broadcast.leftIndex,Broadcast.rightIndex,valueBlocks] using safeLoop) loopRun
  exact finish_safe M D p scale pid (s.numPids 2) s.mem mid _ ctx fixed ha bounds hdq hdk

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelGradient
