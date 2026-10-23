import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.FPReferenceProgram
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeSourceSafety
open VeriTile Triton FP.Structural FPSuccess FPReference
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Source-free windows of one input-array tile. Signed request/page values
retain the evaluator's exact toNat address conversion. -/
structure Bounds (p : Layout) (r : Regions) (a : FPReference.Inputs α)
    (bounds : RegionName → Nat) : Prop where
  start : a.batch<bounds r.start
  sequence : a.batch<bounds r.sequence
  prompt : a.batch<bounds r.prompt
  request : a.batch<bounds r.request
  query : ∀ (i : Fin p.BM) (d : Fin p.D),(row p a i.val : Int)<a.length →
    ((a.start+row p a i.val)*p.qRow+(a.head*p.qHead : Nat)+(d.val*p.qDim : Nat)).toNat<bounds r.query
  pages : ∀ token : Nat,(token : Int)<keyEnd p a →
    ((p.reqRow : Int)*a.request+(p.reqToken : Int)*token).toNat<bounds r.pages
  key : ∀ token : Nat,(token : Int)<keyEnd p a → ∀ d : Fin p.D,
    ((a.pages (((p.reqRow : Int)*a.request+(p.reqToken : Int)*token).toNat))*p.kRow+
      (a.kvHead*p.kHead : Nat)+(d.val*p.kDim : Nat)).toNat<bounds r.key
  value : ∀ token : Nat,(token : Int)<keyEnd p a → ∀ d : Fin p.D,
    ((a.pages (((p.reqRow : Int)*a.request+(p.reqToken : Int)*token).toNat))*p.vRow+
      (a.kvHead*p.vHead : Nat)+(d.val*p.vDim : Nat)).toNat<bounds r.value
  output : ∀ (i : Fin p.BM) (d : Fin p.D),(row p a i.val : Int)<a.length →
    ((a.start+row p a i.val)*p.outRow+(a.head*p.outHead : Nat)+(d.val*p.outDim : Nat)).toNat<bounds r.out

theorem prelude (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α) (s : State α)
    (bounds : RegionName → Nat) (access : Bounds p r (inputs p r s) bounds) :
    Safety.traceSafeList (M.withDefaultPrecision .fp32) bounds (parts p r scale).before s := by
  simp [parts,kernel,VeriTile.Bench.TritonBenchG.ContextAttnLlama.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnLlama.original,ComputeKernel.surfaceBody,Source.intRange?]
  simp (config := {maxSteps:=400000}) [Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,Safety.precisionContext,
    FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,numeric,numericLt,natLt,
    bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,integralFloorDiv,integralMod,
    State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
    Broadcast.leftIndex,Broadcast.rightIndex]
  repeat' apply And.intro
  all_goals first
  | exact access.start
  | exact access.prompt
  | exact access.sequence
  | exact access.request
  | intro i d on; simpa [inputs,FPReference.row] using access.query i d on

#axiomsClean prelude
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeSourceSafety
