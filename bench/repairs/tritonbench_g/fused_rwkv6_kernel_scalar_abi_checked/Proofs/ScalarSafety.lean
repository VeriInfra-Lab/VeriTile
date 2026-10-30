import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.ScalarTransport
import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.SafetyLoop

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite Safety FusedRwkv6Kernel
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem i32_body_safe (M : Algebra α) (c : Shape) (initial current : State α)
    (value : Int) (n : Nat)
    (ctx : Context M c initial (M.fromInt (some .fp32) value) n current)
    (bounds : RegionName → Nat) :
    traceSafeList (engine M) bounds (i32Parts c value).body
      (current.setReg "_" .int [] (fun _ => (n : Int))) ↔
    traceSafeList (engine M) bounds (parts c).body
      (current.setReg "_" .int [] (fun _ => (n : Int))) := by
  by_cases hr : c.reverse=Bool.true
  all_goals
    simp [i32Parts,parts,kernelI32,kernel,ComputeKernel.surfaceBody,Triton.Source.intRange?,
      Option.map,Option.getD,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,
      safeCompute_unfold,safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,boundaryZero,Option.bind,engine,Algebra.withDefaultPrecision,
      resolvePrecision,ComputeDType.eraseDType,numeric,integralMod,numericLt,natLt,
      bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,hr,
      ctx.mem,ctx.iv,ctx.ik,ctx.iz,ctx.km,ctx.vm,ctx.kvm,ctx.pq,ctx.pk,
      ctx.pw,ctx.pv,ctx.po,ctx.bias,ctx.state,ctx.scalar,FP.Structural.store]

#axiomsClean i32_body_safe

theorem i64_body_safe (M : Algebra α) (c : Shape) (initial current : State α)
    (value : Int) (n : Nat)
    (ctx : Context M c initial (M.fromInt (some .fp32) value) n current)
    (bounds : RegionName → Nat) :
    traceSafeList (engine M) bounds (i64Parts c value).body
      (current.setReg "_" .int [] (fun _ => (n : Int))) ↔
    traceSafeList (engine M) bounds (parts c).body
      (current.setReg "_" .int [] (fun _ => (n : Int))) := by
  by_cases hr : c.reverse=Bool.true
  all_goals
    simp [i64Parts,parts,kernelI64,kernel,ComputeKernel.surfaceBody,Triton.Source.intRange?,
      Option.map,Option.getD,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,
      safeCompute_unfold,safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,boundaryZero,Option.bind,engine,Algebra.withDefaultPrecision,
      resolvePrecision,ComputeDType.eraseDType,numeric,integralMod,numericLt,natLt,
      bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,hr,
      ctx.mem,ctx.iv,ctx.ik,ctx.iz,ctx.km,ctx.vm,ctx.kvm,ctx.pq,ctx.pk,
      ctx.pw,ctx.pv,ctx.po,ctx.bias,ctx.state,ctx.scalar,FP.Structural.store]

#axiomsClean i64_body_safe

theorem u64_body_safe (M : Algebra α) (c : Shape) (initial current : State α)
    (value : Nat) (n : Nat)
    (ctx : Context M c initial (M.fromNat (some .fp32) value) n current)
    (bounds : RegionName → Nat) :
    traceSafeList (engine M) bounds (u64Parts c value).body
      (current.setReg "_" .int [] (fun _ => (n : Int))) ↔
    traceSafeList (engine M) bounds (parts c).body
      (current.setReg "_" .int [] (fun _ => (n : Int))) := by
  by_cases hr : c.reverse=Bool.true
  all_goals
    simp [u64Parts,parts,kernelU64,kernel,ComputeKernel.surfaceBody,Triton.Source.intRange?,
      Option.map,Option.getD,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,
      safeCompute_unfold,safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,boundaryZero,Option.bind,engine,Algebra.withDefaultPrecision,
      resolvePrecision,ComputeDType.eraseDType,numeric,integralMod,numericLt,natLt,
      bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,hr,
      ctx.mem,ctx.iv,ctx.ik,ctx.iz,ctx.km,ctx.vm,ctx.kvm,ctx.pq,ctx.pk,
      ctx.pw,ctx.pv,ctx.po,ctx.bias,ctx.state,ctx.scalar,FP.Structural.store]

#axiomsClean u64_body_safe

private theorem scalar_loop_safe_eq (M : Algebra α) (c : Shape) (initial : State α) (scale : α)
    (body : List ComputeStmt)
    (body_eq : ∀ (current : State α) (n : Nat) (ctx : Context M c initial scale n current),
      run (engine M) body (current.setReg "_" .int [] (fun _ => (n : Int)))=
        some (bodyExec M c initial current scale n ctx).val)
    (body_safe : ∀ (current : State α) (n : Nat) (ctx : Context M c initial scale n current)
      (bounds : RegionName → Nat),
      traceSafeList (engine M) bounds body (current.setReg "_" .int [] (fun _ => (n : Int))) ↔
        traceSafeList (engine M) bounds (parts c).body (current.setReg "_" .int [] (fun _ => (n : Int))))
    (current : State α) (n : Nat) (hn : n≤c.T) (ctx : Context M c initial scale n current)
    (bounds : RegionName → Nat) :
    rangeIntSafe (engine M) bounds "_" n c.T 1 body current ↔
      rangeIntSafe (engine M) bounds "_" n c.T 1 (parts c).body current := by
  by_cases live : n<c.T
  · have live' : (n : Int)<(c.T : Int) := Int.ofNat_lt.mpr live
    rw [rangeIntSafe_unfold,rangeIntSafe_unfold]
    simp only [show (1 : Nat)≠0 by decide,if_false,if_pos live',
      body_eq current n ctx,(bodyExec M c initial current scale n ctx).property]
    apply and_congr (body_safe current n ctx bounds)
    simpa only [Nat.cast_add,Nat.cast_one] using
      scalar_loop_safe_eq M c initial scale body body_eq body_safe
        (bodyExec M c initial current scale n ctx).val (n+1) (by omega)
        (body_context M c initial current scale n ctx live) bounds
  · have live' : ¬(n : Int)<(c.T : Int) := fun h => live (Int.ofNat_lt.mp h)
    rw [rangeIntSafe_unfold,rangeIntSafe_unfold]
    simp only [show (1 : Nat)≠0 by decide,if_false,if_neg live']
termination_by c.T-n
decreasing_by omega

private theorem scalar_source_safe_eq (M : Algebra α) (c : Shape) (s : State α) (scale : α)
    (hs : s.regs .real [] "scale"=some (fun _ => scale))
    (code : ComputeKernel) (body : List ComputeStmt)
    (source : code.surfaceBody=(parts c).before++
      [.forRangeInt "_" (.constInt 0) (.constInt (c.T : Int)) (.constNat 1) body]++(parts c).after)
    (body_eq : ∀ (current : State α) (n : Nat) (ctx : Context M c s scale n current),
      run (engine M) body (current.setReg "_" .int [] (fun _ => (n : Int)))=
        some (bodyExec M c s current scale n ctx).val)
    (body_safe : ∀ (current : State α) (n : Nat) (ctx : Context M c s scale n current)
      (bounds : RegionName → Nat),
      traceSafeList (engine M) bounds body (current.setReg "_" .int [] (fun _ => (n : Int))) ↔
        traceSafeList (engine M) bounds (parts c).body (current.setReg "_" .int [] (fun _ => (n : Int))))
    (bounds : RegionName → Nat) :
    TraceSafe (engine M) bounds code s ↔ TraceSafe (engine M) bounds (kernel c) s := by
  let start := entry M c s
  have ctx := entry_context M c s scale hs
  have loops := scalar_loop_eq M c s scale body body_eq start.val 0 (by omega) ctx
  have safe := scalar_loop_safe_eq M c s scale body body_eq body_safe start.val 0 (by omega) ctx bounds
  unfold TraceSafe
  rw [source,source_body,List.append_assoc,List.append_assoc,
    traceSafeList_append,traceSafeList_append]
  apply and_congr_right
  intro _
  apply forall_congr'
  intro next
  apply imp_congr_right
  intro ran
  have heq : next=start.val := Option.some.inj (ran.symm.trans start.property)
  subst next
  simp only [List.singleton_append,traceSafeList_cons,stepSafe_unfold,safeOp_unfold,
    FP.Structural.step,evalOp_unfold,Option.bind,true_and]
  change (rangeIntSafe (engine M) bounds "_" 0 c.T 1 body start.val ∧
      ∀ t, rangeInt (engine M) "_" 0 c.T 1 body start.val=some t →
        traceSafeList (engine M) bounds (parts c).after t) ↔
    (rangeIntSafe (engine M) bounds "_" 0 c.T 1 (parts c).body start.val ∧
      ∀ t, rangeInt (engine M) "_" 0 c.T 1 (parts c).body start.val=some t →
        traceSafeList (engine M) bounds (parts c).after t)
  simp only [Nat.cast_zero] at loops safe
  rw [loops,safe]

theorem i32_source_safe (M : Algebra α) (c : Shape) (s : State α) (value : Int)
    (hs : s.regs .real [] "scale"=some (fun _ => M.fromInt (some .fp32) value))
    (bounds : RegionName → Nat) :
    TraceSafe (engine M) bounds (kernelI32 c value) s ↔
      TraceSafe (engine M) bounds (kernel c) s :=
  scalar_source_safe_eq M c s (M.fromInt (some .fp32) value) hs
    (kernelI32 c value) (i32Parts c value).body (i32_body c value)
    (fun current n ctx => i32_body_exec M c s current value n ctx)
    (fun current n ctx bounds => i32_body_safe M c s current value n ctx bounds) bounds

#axiomsClean i32_source_safe

theorem i64_source_safe (M : Algebra α) (c : Shape) (s : State α) (value : Int)
    (hs : s.regs .real [] "scale"=some (fun _ => M.fromInt (some .fp32) value))
    (bounds : RegionName → Nat) :
    TraceSafe (engine M) bounds (kernelI64 c value) s ↔
      TraceSafe (engine M) bounds (kernel c) s :=
  scalar_source_safe_eq M c s (M.fromInt (some .fp32) value) hs
    (kernelI64 c value) (i64Parts c value).body (i64_body c value)
    (fun current n ctx => i64_body_exec M c s current value n ctx)
    (fun current n ctx bounds => i64_body_safe M c s current value n ctx bounds) bounds

#axiomsClean i64_source_safe

theorem u64_source_safe (M : Algebra α) (c : Shape) (s : State α) (value : Nat)
    (hs : s.regs .real [] "scale"=some (fun _ => M.fromNat (some .fp32) value))
    (bounds : RegionName → Nat) :
    TraceSafe (engine M) bounds (kernelU64 c value) s ↔
      TraceSafe (engine M) bounds (kernel c) s :=
  scalar_source_safe_eq M c s (M.fromNat (some .fp32) value) hs
    (kernelU64 c value) (u64Parts c value).body (u64_body c value)
    (fun current n ctx => u64_body_exec M c s current value n ctx)
    (fun current n ctx bounds => u64_body_safe M c s current value n ctx bounds) bounds

#axiomsClean u64_source_safe
theorem one_body_safe (M : Algebra α) (c : Shape) (initial current : State α)
    (n : Nat)
    (ctx : Context M c initial (M.literal (some .fp32) .real 1) n current)
    (bounds : RegionName → Nat) :
    traceSafeList (engine M) bounds (oneParts c).body
      (current.setReg "_" .int [] (fun _ => (n : Int))) ↔
    traceSafeList (engine M) bounds (parts c).body
      (current.setReg "_" .int [] (fun _ => (n : Int))) := by
  by_cases hr : c.reverse=Bool.true
  all_goals
    simp [oneParts,parts,kernelOne,kernel,ComputeKernel.surfaceBody,Triton.Source.intRange?,
      Option.map,Option.getD,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,
      safeCompute_unfold,safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,boundaryZero,Option.bind,engine,Algebra.withDefaultPrecision,
      resolvePrecision,ComputeDType.eraseDType,numeric,integralMod,numericLt,natLt,
      bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,hr,
      ctx.mem,ctx.iv,ctx.ik,ctx.iz,ctx.km,ctx.vm,ctx.kvm,ctx.pq,ctx.pk,
      ctx.pw,ctx.pv,ctx.po,ctx.bias,ctx.state,ctx.scalar,FP.Structural.store]

#axiomsClean one_body_safe

theorem one_source_safe (M : Algebra α) (c : Shape) (s : State α)
    (hs : s.regs .real [] "scale"=some (fun _ => M.literal (some .fp32) .real 1))
    (bounds : RegionName → Nat) :
    TraceSafe (engine M) bounds (kernelOne c) s ↔
      TraceSafe (engine M) bounds (kernel c) s :=
  scalar_source_safe_eq M c s (M.literal (some .fp32) .real 1) hs
    (kernelOne c) (oneParts c).body (one_body c)
    (fun current n ctx => one_body_exec M c s current n ctx)
    (fun current n ctx bounds => one_body_safe M c s current n ctx bounds) bounds

#axiomsClean one_source_safe


end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
