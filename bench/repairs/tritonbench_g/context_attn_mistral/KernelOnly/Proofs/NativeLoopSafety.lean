import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Proofs.NativeSourceSafety

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

theorem body (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α)
    (hBN : 0<p.BN) (initialState s : State α)
    (old : Acc p α) (live : Matches p r o (KernelReference.inputs p r initialState) old s)
    (memory : s.mem=initialState.mem) (cur : Int) (nonnegative : 0≤cur)
    (active : cur<stop p (KernelReference.inputs p r initialState))
    (bounds : RegionName → Nat) (access : Bounds p r (initialState.pids 0) (KernelReference.inputs p r initialState) bounds) :
    Safety.traceSafeList (engine o.algebra) bounds (parts p r .fp32 scale).body
      (s.setReg "start_n" .int [] (fun _=>cur)) := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13⟩
  simp [parts,program,VeriTile.Bench.TritonBenchG.ContextAttnMistral.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnMistral.original,ComputeKernel.surfaceBody,Source.intRange?]
  simp (config := {maxSteps:=500000}) [Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,Safety.precisionContext,
    FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,FPBackend.Ops.algebra,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,numericLt,numericLe,numericEq,natLt,natLe,
    bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,integralFloorDiv,integralMod,
    State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
    Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,
    h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,memory,hBN,keyPointer,valuePointer]
  constructor
  · intro d j on
    exact access.key cur nonnegative active d j on
  · intro j d on
    exact access.value cur nonnegative active j d on

#axiomsClean body
end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.NativeSourceSafety
