import bench.tritonbench_g.context_attn_mistral.Program
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.NativeSourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ContextAttnMistral Formula
open VeriTile.Bench.TritonBenchG.ContextAttnMistral.Program (Regions)
open VeriTile.Bench.Repairs.ContextAttnMistral KernelReference
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Actual source windows, retaining raw signed address conversion. -/
structure Bounds (p : Layout) (r : Regions) (batch : Nat) (a : KernelReference.Inputs α)
    (bounds : RegionName → Nat) : Prop where
  start : batch<bounds r.start
  sequence : batch<bounds r.sequence
  query : ∀ (i : Fin p.BM) (d : Fin p.D),(row p a i.val : Int)<a.length →
    ((a.start+row p a i.val)*p.qRow+(a.head*p.qHead : Nat)+(d.val*p.qDim : Nat)).toNat<bounds r.query
  key : ∀ cur : Int,0≤cur → cur<stop p a → ∀ (d : Fin p.D) (j : Fin p.BN),cur+j.val<a.length →
    j.val*p.kRow+a.kvHead*p.kHead+d.val*p.kDim+((a.start+cur)*p.kRow).toNat<bounds r.key
  value : ∀ cur : Int,0≤cur → cur<stop p a → ∀ (j : Fin p.BN) (d : Fin p.D),cur+j.val<a.length →
    j.val*p.vRow+a.kvHead*p.vHead+d.val*p.vDim+((a.start+cur)*p.vRow).toNat<bounds r.value
  output : ∀ (i : Fin p.BM) (d : Fin p.D),(row p a i.val : Int)<a.length →
    ((a.start+row p a i.val)*p.outRow+(a.head*p.outHead : Nat)+(d.val*p.outDim : Nat)).toNat<bounds r.out

theorem prelude (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α) (s : State α)
    (bounds : RegionName → Nat) (access : Bounds p r (s.pids 0) (KernelReference.inputs p r s) bounds) :
    Safety.traceSafeList (engine o.algebra) bounds (parts p r .fp32 scale).before s := by
  simp [parts,program,VeriTile.Bench.TritonBenchG.ContextAttnMistral.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnMistral.original,VeriTile.Bench.Repairs.ContextAttnMistralKernel.kernel,VeriTile.Bench.Repairs.ContextAttnMistralKernel.storageProfile,ComputeKernel.surfaceBody,Source.intRange?]
  simp (config := {maxSteps:=400000}) [Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,Safety.precisionContext,
    FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,FPBackend.Ops.algebra,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,numeric,numericLt,numericLe,natLt,natLe,
    bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,integralFloorDiv,integralMod,
    State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
    Broadcast.leftIndex,Broadcast.rightIndex]
  repeat' apply And.intro
  all_goals first
  | exact access.start
  | exact access.sequence
  | intro i d on; simpa [KernelReference.inputs,KernelReference.row] using access.query i d on

#axiomsClean prelude
end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.NativeSourceSafety
