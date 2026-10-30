import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.IO

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.SourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite Safety FusedRwkv6Kernel Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure Bounds (c : Shape) (s : State α) (bounds : RegionName → Nat) : Prop where
  bias : ∀ j : Fin c.BK, s.pids 1*c.BK+j.val<c.K →
    (s.pids 2%c.H)*c.K+j.val+s.pids 1*c.BK < bounds "u"
  seed : c.initial=Bool.true → ∀ (l : Fin c.BV) (j : Fin c.BK),
    s.pids 0*c.BV+l.val<c.V → s.pids 1*c.BK+j.val<c.K →
    terminalAddress c s (l,j,()) < bounds "h0"
  key : ∀ n, n<c.T → ∀ j : Fin c.BK, s.pids 1*c.BK+j.val<c.K →
    ∀ r∈["q","k","w"], pointer c (keyBase c s j) c.K n < bounds r
  value : ∀ n, n<c.T → ∀ l : Fin c.BV, s.pids 0*c.BV+l.val<c.V →
    pointer c (valueBase c s l) c.V n < bounds "v"
  partialOutput : ∀ n, n<c.T → ∀ l : Fin c.BV, s.pids 0*c.BV+l.val<c.V →
    pointer c (outputBase c s l) c.V n < bounds "o"
  terminal : c.final=Bool.true → ∀ (l : Fin c.BV) (j : Fin c.BK),
    s.pids 0*c.BV+l.val<c.V → s.pids 1*c.BK+j.val<c.K →
    terminalAddress c s (l,j,()) < bounds "ht"

theorem entry_safe (M : Algebra α) (c : Shape) (s : State α)
    (bounds : RegionName → Nat) (hb : Bounds c s bounds) :
    traceSafeList (engine M) bounds (parts c).before s := by
  by_cases hi : c.initial=Bool.true
  all_goals
    simp [parts,kernel,ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,
      Option.map,Option.getD,hi,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,
      safeCompute_unfold,safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,
      Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralMod,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg]
  · constructor
    · intro l j hl hj
      have h := hb.seed hi l j hl hj
      unfold terminalAddress at h
      ring_nf at h ⊢
      exact h
    · intro j hj
      exact hb.bias j hj
  · intro j hj
    exact hb.bias j hj

theorem body_tail (M : Algebra α) (c : Shape) (s : State α) (bounds : RegionName → Nat) :
    traceSafeList (engine M) bounds ((parts c).body.drop 11) s := by
  simp [parts,kernel,ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,
    Option.map,Option.getD,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,
    safeCompute_unfold,safeOp_unfold,evalOp_unfold]
  all_goals repeat' first | exact True.intro | constructor | split

theorem body_safe (M : Algebra α) (c : Shape) (initial current : State α) (scale : α)
    (n : Nat) (hn : n<c.T) (ctx : Context M c initial scale n current)
    (bounds : RegionName → Nat) (hb : Bounds c initial bounds) :
    traceSafeList (engine M) bounds (parts c).body
      (current.setReg "_i" .nat [] (fun _ => n)) := by
  rw [←List.take_append_drop 11 (parts c).body,traceSafeList_append]
  refine ⟨?_,fun next _ => body_tail M c next bounds⟩
  simp [parts,kernel,ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,
    Option.map,Option.getD,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,
    safeCompute_unfold,safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,
    Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralMod,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    ctx.km,ctx.vm,ctx.kvm,ctx.pk,ctx.pv,ctx.pq,ctx.pw,ctx.po,ctx.bias,ctx.state,ctx.scalar,
    keyMask,valueMask]
  refine ⟨?_,?_,?_,?_,?_,?_⟩
  · intro j hj
    exact hb.key n hn j hj "k" (by simp)
  · intro l hl
    exact hb.value n hn l hl
  · intro j hj
    exact hb.key n hn j hj "q" (by simp)
  · intro j hj
    exact hb.key n hn j hj "w" (by simp)
  · intro l hl
    exact hb.partialOutput n hn l hl
  · split <;> trivial

theorem finish_safe (M : Algebra α) (c : Shape) (initial current : State α) (scale : α)
    (ctx : FinalContext M c initial scale current) (bounds : RegionName → Nat)
    (hb : Bounds c initial bounds) : traceSafeList (engine M) bounds (parts c).after current := by
  by_cases hf : c.final=Bool.true
  all_goals
    simp [parts,kernel,ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,
      Option.map,Option.getD,hf,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,
      safeCompute_unfold,safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,
      Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralMod,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      ctx.mem,ctx.iv,ctx.ik,ctx.iz,ctx.kvm,ctx.state,keyMask,valueMask,FP.Structural.store]
  intro l j hl hj
  have h := hb.terminal hf l j hl hj
  unfold terminalAddress at h
  ring_nf at h ⊢
  exact h

end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.SourceSafety
