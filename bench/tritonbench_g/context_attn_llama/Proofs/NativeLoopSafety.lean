import bench.tritonbench_g.context_attn_llama.Proofs.NativeSourceSafety

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeSourceSafety
open VeriTile Triton FP.Structural FPSuccess FPReference
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem body (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : DotSupport p M) (hBN : 0<p.BN) (initialState s : State α)
    (old : Acc p α) (live : Matches p M (inputs p r initialState) old s)
    (memory : s.mem=initialState.mem) (cur : Int) (nonnegative : 0≤cur)
    (bounds : RegionName → Nat) (access : Bounds p r (inputs p r initialState) bounds) :
    Safety.traceSafeList (M.withDefaultPrecision .fp32) bounds (parts p r scale).body
      (s.setReg "start_n" .int [] (fun _=>cur)) := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14⟩
  simp [parts,kernel,originalFP32,VeriTile.Bench.TritonBenchG.ContextAttnLlama.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnLlama.original,ComputeKernel.surfaceBody,Source.intRange?]
  simp (config := {maxSteps:=500000}) [Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,Safety.precisionContext,
    FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,numeric,numericLt,numericLe,natLt,
    bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,integralFloorDiv,integralMod,
    State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
    Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,
    h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,memory,
    dots.qk_bound,dots.pv_bound,hBN]
  refine ⟨?_,?_,?_⟩
  · intro j on
    have token : ((cur+j.val).toNat : Int)=cur+j.val := Int.toNat_of_nonneg (by omega)
    simpa only [token] using access.pages (cur+j.val).toNat (by simpa only [token] using on)
  · intro d j on
    have token : ((cur+j.val).toNat : Int)=cur+j.val := Int.toNat_of_nonneg (by omega)
    simp only [if_pos on]
    simpa only [token,inputs,Nat.cast_mul] using access.key (cur+j.val).toNat
      (by simpa only [token] using on) d
  · intro j d on
    have token : ((cur+j.val).toNat : Int)=cur+j.val := Int.toNat_of_nonneg (by omega)
    simp only [if_pos on]
    simpa only [token,inputs,Nat.cast_mul] using access.value (cur+j.val).toNat
      (by simpa only [token] using on) d

#axiomsClean body
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeSourceSafety
