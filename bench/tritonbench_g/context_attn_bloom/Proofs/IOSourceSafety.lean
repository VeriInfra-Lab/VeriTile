import bench.tritonbench_g.context_attn_bloom.Proofs.FPReferenceProgram
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.IOSourceSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program
open FPSuccess FPReference
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Active source addresses computed from the loaded metadata. Shared IO
derives these bounds; they are never a numerical or result assumption. -/
structure Bounds (p : Layout) (r : Regions) (a : FPReference.Inputs α)
    (bounds : RegionName→Nat) : Prop where
  start : a.batch<bounds r.start
  prompt : a.batch<bounds r.prompt
  sequence : a.batch<bounds r.sequence
  request : a.batch<bounds r.request
  query : ∀ (i : Fin p.BM) (d : Fin p.D),(FPReference.row p a i.val : Int)<a.length → d.val<p.headDim →
    ((a.start+FPReference.row p a i.val)*p.qRow+(a.head*p.qHead : Nat)+(d.val*p.qDim : Nat)).toNat<bounds r.query
  tokens : ∀ (cur : Int),0≤cur → (∃ n : Nat,cur=(n*p.BN : Nat)) →
    cur<(FPReference.blockMask p a : Int)*FPReference.keyEnd p a → ∀ j : Fin p.BN,
    cur+j.val<FPReference.keyEnd p a →
    ((p.requestRow : Int)*a.request+(p.requestToken : Int)*(cur+j.val)).toNat<bounds r.tokens
  key : ∀ (cur : Int),0≤cur → (∃ n : Nat,cur=(n*p.BN : Nat)) →
    cur<(FPReference.blockMask p a : Int)*FPReference.keyEnd p a →
    ∀ (j : Fin p.BN) (d : Fin p.D),cur+j.val<FPReference.keyEnd p a → d.val<p.headDim →
    ((FPReference.page p a cur j.val)*p.kRow+(a.kvHead*p.kHead : Nat)+(d.val*p.kDim : Nat)).toNat<bounds r.key
  value : ∀ (cur : Int),0≤cur → (∃ n : Nat,cur=(n*p.BN : Nat)) →
    cur<(FPReference.blockMask p a : Int)*FPReference.keyEnd p a →
    ∀ (j : Fin p.BN) (d : Fin p.D),cur+j.val<FPReference.keyEnd p a → d.val<p.headDim →
    ((FPReference.page p a cur j.val)*p.vRow+(a.kvHead*p.vHead : Nat)+(d.val*p.vDim : Nat)).toNat<bounds r.value
  output : ∀ (i : Fin p.BM) (d : Fin p.D),(FPReference.row p a i.val : Int)<a.length → d.val<p.headDim →
    ((a.start+FPReference.row p a i.val)*p.outRow+(a.head*p.outHead : Nat)+(d.val*p.outDim : Nat)).toNat<bounds r.out

theorem prelude_safe (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α) (s : State α)
    (bounds : RegionName→Nat) (hb : Bounds p r (FPReference.inputs p r s) bounds) :
    Safety.traceSafeList (M.withDefaultPrecision .fp32) bounds (parts p r scale).before s := by
  simp (config := {maxSteps := 700000}) [parts,kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnBloom.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnBloom.original,ComputeKernel.surfaceBody,Source.intRange?,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,Algebra.withDefaultPrecision,
    resolvePrecision,numeric,numericLt,numericLe,natLt,natLe,integralFloorDiv,integralMod,
    ofFloat,toFloat,Region.cast,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  refine ⟨hb.start,hb.prompt,hb.sequence,hb.request,?_⟩
  intro i d active dim
  simpa [FPReference.inputs,FPReference.row,Nat.cast_mul] using hb.query i d active dim

theorem body_safe (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : Dot α) (hBN : 0<p.BN) (initialState s : State α) (old : Acc p α)
    (live : Matches p M (FPReference.inputs p r initialState) old s)
    (memory : s.mem=initialState.mem) (cur : Int) (nonnegative : 0≤cur)
    (aligned : ∃ n : Nat,cur=(n*p.BN : Nat))
    (active : cur<(FPReference.blockMask p (FPReference.inputs p r initialState) : Int)*
      FPReference.keyEnd p (FPReference.inputs p r initialState))
    (bounds : RegionName→Nat) (hb : Bounds p r (FPReference.inputs p r initialState) bounds) :
    Safety.traceSafeList ((withDot M dots).withDefaultPrecision .fp32) bounds (parts p r scale).body
      (s.setReg "start_n" .int [] (fun _=>cur)) := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14⟩
  simp (config := {maxSteps := 1600000}) [parts,kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnBloom.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnBloom.original,ComputeKernel.surfaceBody,Source.intRange?,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,Algebra.withDefaultPrecision,
    resolvePrecision,numeric,numericLt,numericLe,natLt,natLe,integralFloorDiv,integralMod,
    ofFloat,toFloat,Region.cast,h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,
    withDot,TileShape.axisDim,hBN,memory,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  refine ⟨fun j h=>hb.tokens cur nonnegative aligned active j h,?_,?_⟩
  · intro d j h dim
    simpa [FPReference.page,h,FPReference.inputs,Nat.cast_mul] using
      hb.key cur nonnegative aligned active j d h dim
  · intro j d h dim
    simpa [FPReference.page,h,FPReference.inputs,Nat.cast_mul] using
      hb.value cur nonnegative aligned active j d h dim

theorem epilogue_safe (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (initialState s : State α) (old : Acc p α)
    (live : Matches p M (FPReference.inputs p r initialState) old s)
    (bounds : RegionName→Nat) (hb : Bounds p r (FPReference.inputs p r initialState) bounds) :
    Safety.traceSafeList (M.withDefaultPrecision .fp32) bounds (parts p r scale).after s := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14⟩
  simp (config := {maxSteps := 700000}) [parts,kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnBloom.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnBloom.original,ComputeKernel.surfaceBody,Source.intRange?,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,Algebra.withDefaultPrecision,
    resolvePrecision,numeric,numericLt,natLt,ofFloat,toFloat,Region.cast,
    h9,h10,h11,h12,h5,h4,h13,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  constructor
  · intro i d active dim
    simpa only [Nat.cast_mul] using hb.output i d active dim
  · split <;> trivial

end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.IOSourceSafety
