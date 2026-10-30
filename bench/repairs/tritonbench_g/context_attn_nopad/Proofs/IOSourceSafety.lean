import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.FPReferenceProgram
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOSourceSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Formula Program
open FP.ProfiledRewrite (engine)
open FPReference
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Exact enabled source accesses. These bounds are derived from shared IO,
not assumed as a numerical property or a program-result equality. -/
structure Bounds (p : Layout) (r : Regions) (batch : Nat) (a : FPReference.Inputs α)
    (bounds : RegionName → Nat) : Prop where
  metadataStart : batch<bounds r.start
  metadataLength : batch<bounds r.sequence
  query : ∀ (i : Fin p.BM) (d : Fin p.D),(row p a i.val : Int)<a.length →
    ((a.start+row p a i.val)*p.qRow+(a.head*p.qHead : Nat)+(d.val*p.qDim : Nat)).toNat<bounds r.query
  key : ∀ (cur : Int),0 ≤ cur → (∃ n : Nat,cur=(n*p.BN : Nat)) → cur < ((blockMask p a*(a.block+1)*p.BM : Nat) : Int) → ∀ (j : Fin p.BN) (d : Fin p.D),cur+j.val<a.length →
    j.val*p.kRow+a.head*p.kHead+d.val*p.kDim+((a.start+cur)*p.kRow).toNat<bounds r.key
  value : ∀ (cur : Int),0 ≤ cur → (∃ n : Nat,cur=(n*p.BN : Nat)) → cur < ((blockMask p a*(a.block+1)*p.BM : Nat) : Int) → ∀ (j : Fin p.BN) (d : Fin p.D),cur+j.val<a.length →
    j.val*p.vRow+a.head*p.vHead+d.val*p.vDim+((a.start+cur)*p.vRow).toNat<bounds r.value
  output : ∀ (i : Fin p.BM) (d : Fin p.D),(row p a i.val : Int)<a.length →
    ((a.start+row p a i.val)*p.outRow+(a.head*p.outHead : Nat)+(d.val*p.outDim : Nat)).toNat<bounds r.out

theorem prelude_safe (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α) (s : State α)
    (bounds : RegionName → Nat) (hb : Bounds p r (s.pids 0) (FPReference.inputs r s) bounds) :
    Safety.traceSafeList (engine o.algebra) bounds (FPSymbolic.parts p r scale).before s := by
  simp (config := {maxSteps := 700000}) [FPSymbolic.parts,Program.parts,Program.mathematical,Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnNopad.original,ComputeKernel.surfaceBody,Source.intRange?,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,Algebra.withDefaultPrecision,engine,
    resolvePrecision,numeric,numericLt,numericLe,natLt,natLe,integralFloorDiv,integralMod,
    ofFloat,toFloat,Region.cast,Matches,FPReference.inputs,initial,blockMask,queryTile,row,bop,
    Broadcast.leftIndex,Broadcast.rightIndex]
  exact ⟨hb.metadataLength,hb.metadataStart,fun i d h => hb.query i d h⟩

theorem body_safe (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α)
    (hBN : 0<p.BN) (initialState s : State α) (old : Acc p α)
    (live : Matches p r o.algebra (FPReference.inputs r initialState) old s) (cur : Int) (hc : 0 ≤ cur)
    (aligned : ∃ n : Nat,cur=(n*p.BN : Nat))
    (ht : cur < ((blockMask p (FPReference.inputs r initialState)*((FPReference.inputs r initialState).block+1)*p.BM : Nat) : Int))
    (bounds : RegionName → Nat) (hb : Bounds p r (initialState.pids 0) (FPReference.inputs r initialState) bounds) :
    Safety.traceSafeList (engine o.algebra) bounds (FPSymbolic.parts p r scale).body
      (s.setReg "start_n" .int [] (fun _ => cur)) := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13⟩
  simp (config := {maxSteps := 1600000}) [FPSymbolic.parts,Program.parts,Program.mathematical,Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnNopad.original,ComputeKernel.surfaceBody,Source.intRange?,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,Algebra.withDefaultPrecision,engine,FPBackend.Ops.algebra,
    resolvePrecision,numeric,numericLt,numericLe,natLt,natLe,integralFloorDiv,integralMod,
    ofFloat,toFloat,Region.cast,h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,
    TileShape.axisDim,hBN,Matches,advance,keyTile,valueTile,FPReference.inputs,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  exact ⟨fun d j h => hb.key cur hc aligned ht j d h,fun j d h => hb.value cur hc aligned ht j d h⟩

theorem epilogue_safe (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α)
    (initialState s : State α) (acc : Acc p α) (live : Matches p r o.algebra (FPReference.inputs r initialState) acc s)
    (bounds : RegionName → Nat) (hb : Bounds p r (initialState.pids 0) (FPReference.inputs r initialState) bounds) :
    Safety.traceSafeList (engine o.algebra) bounds (FPSymbolic.parts p r scale).after s := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13⟩
  simp (config := {maxSteps := 700000}) [FPSymbolic.parts,Program.parts,Program.mathematical,Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnNopad.original,ComputeKernel.surfaceBody,Source.intRange?,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,Algebra.withDefaultPrecision,engine,FPBackend.Ops.algebra,
    resolvePrecision,numeric,numericLt,natLt,ofFloat,toFloat,Region.cast,
    h0,h2,h3,h4,h6,h12,store,scatter,active,offset,FPReference.inputs,row,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  exact fun i d h => hb.output i d h

#axiomsClean prelude_safe
#axiomsClean body_safe
#axiomsClean epilogue_safe
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOSourceSafety
