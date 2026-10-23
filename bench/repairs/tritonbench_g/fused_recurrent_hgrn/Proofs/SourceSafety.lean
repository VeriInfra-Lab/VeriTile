import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.RawKernel
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.SourceSafety
open VeriTile.Triton FP.Structural Execution
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem forward_prologue_safe (p : Ops α) (s : State α) (T D BD : Nat)
    (U : Bool) (bounds : RegionName → Nat)
    (hh : U=Bool.true → ∀ j, lane s BD j<D → slot s D BD j<bounds "h0") :
    Safety.traceSafeList (algebra p) bounds (forwardPrologue T D BD U) s := by
  cases U <;>
    simp [forwardPrologue,forward,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      numeric,bop,natLt,numericLt,Region.cast,State.setReg,algebra,ofFloat,toFloat]
  exact fun j => hh rfl (j,())

theorem forward_body_safe (p : Ops α) (s : State α) (T D BD t : Nat)
    (carry : TileIndex [BD] → α)
    (hm : s.regs .bool [BD] "mask"=some (fun j => decide (lane s BD j<D)))
    (hx : s.regs .ptr [BD] "p_x"=some (fun j => (("x" : RegionName),row s t T D BD j)))
    (hg : s.regs .ptr [BD] "p_g"=some (fun j => (("g" : RegionName),row s t T D BD j)))
    (ho : s.regs .ptr [BD] "p_o"=some (fun j => (("o" : RegionName),row s t T D BD j)))
    (hc : s.regs .real [BD] "b_h"=some carry)
    (bounds : RegionName → Nat)
    (bx : ∀ j, lane s BD j<D → row s t T D BD j<bounds "x")
    (bg : ∀ j, lane s BD j<D → row s t T D BD j<bounds "g")
    (bo : ∀ j, lane s BD j<D → row s t T D BD j<bounds "o") :
    Safety.traceSafeList (algebra p) bounds (forwardBody D BD) s := by
  simp [forwardBody,forward,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
    numeric,bop,natLt,numericLt,Region.cast,State.setReg,algebra,store,ofFloat,toFloat,
    hm,hx,hg,ho,hc,masked_regs]
  exact ⟨fun a b => bx (a,b),fun a b => bg (a,b),fun a b => bo (a,b)⟩

theorem forward_epilogue_safe (p : Ops α) (s : State α) (T D BD : Nat)
    (S : Bool) (carry : TileIndex [BD] → α)
    (hm : s.regs .bool [BD] "mask"=some (fun j => decide (lane s BD j<D)))
    (hb : s.regs .nat [] "i_bh"=some (fun _ => s.pids 1))
    (hd : s.regs .nat [BD] "o_d"=some (lane s BD))
    (hc : s.regs .real [BD] "b_h"=some carry)
    (bounds : RegionName → Nat)
    (bh : S=Bool.true → ∀ j, lane s BD j<D → slot s D BD j<bounds "ht") :
    Safety.traceSafeList (algebra p) bounds (forwardEpilogue T D BD S) s := by
  cases S <;>
    simp [forwardEpilogue,forward,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      numeric,bop,natLt,numericLt,Region.cast,State.setReg,algebra,store,ofFloat,toFloat,
      hm,hb,hd,hc]
  exact fun a b => bh rfl (a,b)

theorem backward_prologue_safe (p : Ops α) (s : State α) (T D BD : Nat)
    (V : Bool) (bounds : RegionName → Nat)
    (hh : V=Bool.true → ∀ j, lane s BD j<D → slot s D BD j<bounds "dht") :
    Safety.traceSafeList (algebra p) bounds (backwardPrologue T D BD V) s := by
  cases V <;>
    simp [backwardPrologue,backward,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      numeric,bop,natLt,numericLt,Region.cast,State.setReg,algebra,ofFloat,toFloat]
  exact fun j => hh rfl (j,())

theorem backward_body_safe (p : Ops α) (s : State α) (T D BD n : Nat) (U : Bool)
    (carry : TileIndex [BD] → α)
    (hn : s.regs .nat [] "step"=some (fun _ => n))
    (hm : s.regs .bool [BD] "mask"=some (fun j => decide (lane s BD j<D)))
    (hb : s.regs .nat [] "i_bh"=some (fun _ => s.pids 1))
    (hd : s.regs .nat [BD] "o_d"=some (lane s BD))
    (hc : s.regs .real [BD] "b_dh"=some carry)
    (bounds : RegionName → Nat)
    (bg : ∀ j, lane s BD j<D → row s (T-1-n) T D BD j<bounds "g")
    (bdo : ∀ j, lane s BD j<D → row s (T-1-n) T D BD j<bounds "do")
    (bx : ∀ j, lane s BD j<D → row s (T-1-n) T D BD j<bounds "dx")
    (bdg : ∀ j, lane s BD j<D → row s (T-1-n) T D BD j<bounds "dg")
    (bo : 0<T-1-n → ∀ j, lane s BD j<D → row s (T-1-n-1) T D BD j<bounds "o")
    (bh : U=Bool.true → ∀ j, lane s BD j<D → slot s D BD j<bounds "h0") :
    Safety.traceSafeList (algebra p) bounds (backwardBody T D BD U) s := by
  have hrow : 0<T-1-n → s.pids 1*T+(T-1-n)-1=s.pids 1*T+(T-1-n-1) := by omega
  cases U <;> by_cases hi : 0<T-1-n <;>
    simp [backwardBody,backward,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      numeric,bop,natLt,numericLt,Region.cast,State.setReg,algebra,store,ofFloat,toFloat,
      hn,hm,hb,hd,hc,hi,masked_regs]
  · exact ⟨fun a b => bg (a,b),fun a b => bdo (a,b),by
      intro a b
      simpa only [row,VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.rowOffset,hrow hi]
        using bo hi (a,b),fun a b => bx (a,b),fun a b => bdg (a,b)⟩
  · exact ⟨fun a b => bg (a,b),fun a b => bdo (a,b),fun a b => bx (a,b),fun a b => bdg (a,b)⟩
  · exact ⟨fun a b => bg (a,b),fun a b => bdo (a,b),by
      intro a b
      simpa only [row,VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.rowOffset,hrow hi]
        using bo hi (a,b),fun a b => bx (a,b),fun a b => bdg (a,b)⟩
  · exact ⟨fun a b => bg (a,b),fun a b => bdo (a,b),fun a b => bh rfl (a,b),
      fun a b => bx (a,b),fun a b => bdg (a,b)⟩

end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.SourceSafety
