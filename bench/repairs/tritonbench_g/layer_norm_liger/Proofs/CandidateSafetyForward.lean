import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.CandidateForward
import VeriTile.Triton.Float.StructuralSafety
namespace VeriTile.Bench.Repairs.LayerNormLiger.CandidateSafety
open VeriTile.Triton FP.Structural Safety
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive withSum)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem assign_any (M : Algebra α) (bounds : RegionName → Nat)
    (s : State α) (dtype : TileDType) (shape : TileShape) (name : RegName)
    (e : ComputeExpr dtype shape) (rest : List ComputeStmt)
    (safe : safeExpr M bounds s e)
    (next : ∀ v, traceSafeList M bounds rest (s.setReg name dtype shape v)) :
    traceSafeList M bounds (.assign dtype shape name e :: rest) s := by
  apply traceSafeList_cons_intro (by simpa only [stepSafe_unfold] using safe)
  intro t ht
  cases he : evalExpr M e s with
  | none => simp [step,he] at ht
  | some v =>
    have eq : s.setReg name dtype shape v=t := by simpa [step,he] using ht
    rw [← eq]
    exact next v

/-- Exact forward source accesses. Mean and rstd are scalar stores even when
no vector lane is active; all three input vectors share the column mask. -/
theorem forward (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters)
    (s : State α) (bounds : RegionName → Nat)
    (hx : ∀ i : Fin p.block,i.val<p.columns → s.pids 0*p.xStride+i.val<bounds p.x)
    (hw : ∀ i : Fin p.block,i.val<p.columns → i.val<bounds p.weight)
    (hb : ∀ i : Fin p.block,i.val<p.columns → i.val<bounds p.bias)
    (hm : s.pids 0*p.meanStride<bounds p.mean)
    (hr : s.pids 0*p.rstdStride<bounds p.rstd)
    (hy : ∀ i : Fin p.block,i.val<p.columns → s.pids 0*p.yStride+i.val<bounds p.y) :
    TraceSafe (withSum M R) bounds (Forward.candidateKernel p) s := by
  change traceSafeList (withSum M R) bounds (Forward.candidateKernel p).surfaceBody s
  simp only [Forward.candidateKernel,Forward.candidateSource,ComputeKernel.surfaceBody]
  rw [traceSafeList_unfold]
  constructor
  · simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
      safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
      State.setReg,ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,
      ofFloat,toFloat,numeric,bop,numericLt,natLt,Region.cast,withSum,hm,hr]
  · simp [step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,bind,pure,
      State.setReg,numeric,bop,numericLt,natLt,Region.cast]
    rw [traceSafeList_unfold]
    constructor
    · simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
        safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
        step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
        State.setReg,ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,
        ofFloat,toFloat,numeric,bop,numericLt,natLt,Region.cast,withSum,hm,hr]
    · simp [step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,bind,pure,
        State.setReg,numeric,bop,numericLt,natLt,Region.cast]
      rw [traceSafeList_unfold]
      constructor
      · simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
          safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
          step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
          State.setReg,ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,
          ofFloat,toFloat,numeric,bop,numericLt,natLt,Region.cast,withSum,hm,hr]
      · simp [step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,bind,pure,
          State.setReg,numeric,bop,numericLt,natLt,Region.cast]
        rw [traceSafeList_unfold]
        constructor
        · simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
            safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
            step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
            State.setReg,ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,
            ofFloat,toFloat,numeric,bop,numericLt,natLt,Region.cast,withSum,hm,hr]
        · simp [step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,bind,pure,
            State.setReg,numeric,bop,numericLt,natLt,Region.cast]
          rw [traceSafeList_unfold]
          constructor
          · simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
              safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
              step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
              State.setReg,ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,
              ofFloat,toFloat,numeric,bop,numericLt,natLt,Region.cast,withSum,hm,hr]
          · simp [step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,bind,pure,
              State.setReg,numeric,bop,numericLt,natLt,Region.cast]
            rw [traceSafeList_unfold]
            constructor
            · simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
                safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
                step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
                State.setReg,ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,
                ofFloat,toFloat,numeric,bop,numericLt,natLt,Region.cast,withSum,hm,hr]
            · simp [step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,bind,pure,
                State.setReg,numeric,bop,numericLt,natLt,Region.cast]
              rw [traceSafeList_unfold]
              constructor
              · simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
                  safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
                  step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
                  State.setReg,ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,
                  ofFloat,toFloat,numeric,bop,numericLt,natLt,Region.cast,withSum,hm,hr]
              · simp [step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,bind,pure,
                  State.setReg,numeric,bop,numericLt,natLt,Region.cast]
                apply assign_any
                · simpa [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
                    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
                    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
                    State.setReg,ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,
                    ofFloat,toFloat,numeric,bop,numericLt,natLt,Region.cast,withSum,hm,hr,hx,hw,hb] using hx
                · intro value0
                  apply assign_any
                  · simpa [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
                      safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
                      step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
                      State.setReg,ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,
                      ofFloat,toFloat,numeric,bop,numericLt,natLt,Region.cast,withSum,hm,hr,hx,hw,hb] using hw
                  · intro value1
                    apply assign_any
                    · simpa [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
                        safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
                        step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
                        State.setReg,ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,
                        ofFloat,toFloat,numeric,bop,numericLt,natLt,Region.cast,withSum,hm,hr,hx,hw,hb] using hb
                    · intro value2
                      apply assign_any
                      · simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
                          safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
                          step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
                          State.setReg,ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,
                          ofFloat,toFloat,numeric,bop,numericLt,natLt,Region.cast,withSum,hm,hr,hx,hw,hb]
                      · intro value3
                        apply assign_any
                        · simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
                            safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
                            step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
                            State.setReg,ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,
                            ofFloat,toFloat,numeric,bop,numericLt,natLt,Region.cast,withSum,hm,hr,hx,hw,hb]
                        · intro value4
                          apply assign_any
                          · simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
                              safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
                              step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
                              State.setReg,ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,
                              ofFloat,toFloat,numeric,bop,numericLt,natLt,Region.cast,withSum,hm,hr,hx,hw,hb]
                          · intro value5
                            apply assign_any
                            · simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
                                safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
                                step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
                                State.setReg,ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,
                                ofFloat,toFloat,numeric,bop,numericLt,natLt,Region.cast,withSum,hm,hr,hx,hw,hb]
                            · intro value6
                              rw [traceSafeList_unfold]
                              constructor
                              · simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
                                  safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
                                  step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
                                  State.setReg,ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,
                                  ofFloat,toFloat,numeric,bop,numericLt,natLt,Region.cast,withSum,hm,hr]
                              · simp [step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,bind,pure,store,
                                  State.setReg,ofFloat,toFloat,Region.cast]
                                rw [traceSafeList_unfold]
                                constructor
                                · simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
                                    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
                                    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
                                    State.setReg,ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,
                                    ofFloat,toFloat,numeric,bop,numericLt,natLt,Region.cast,withSum,hm,hr]
                                · simp [step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,bind,pure,store,
                                    State.setReg,ofFloat,toFloat,Region.cast]
                                  apply assign_any
                                  · simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
                                      safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
                                      step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
                                      State.setReg,ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,
                                      ofFloat,toFloat,numeric,bop,numericLt,natLt,Region.cast,withSum,hm,hr]
                                  · intro outputValue
                                    simpa [traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
                                        safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
                                        step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
                                        State.setReg,ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,
                                        ofFloat,toFloat,numeric,bop,numericLt,natLt,Region.cast,withSum,hm,hr,hy] using hy

#axiomsClean forward
end
end VeriTile.Bench.Repairs.LayerNormLiger.CandidateSafety
