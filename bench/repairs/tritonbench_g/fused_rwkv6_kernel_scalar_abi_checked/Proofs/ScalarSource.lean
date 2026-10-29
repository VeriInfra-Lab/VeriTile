import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.Body

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite FusedRwkv6Kernel
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/- The integer source arguments retain their source ABI and the actual
integer-to-float primitive. These lemmas connect the complete raw loop body
to the recurrence at that converted operand; no integer input is restricted
and no numerical conversion is assumed to be an identity. -/

def i32Parts (c : Shape) (value : Int) : Triton.Source.IntRange :=
  (Triton.Source.intRange? (kernelI32 c value).surfaceBody "_").getD
    ⟨[],.constInt 0,.constInt 0,.constNat 0,[],[]⟩

theorem i32_before (c : Shape) (value : Int) :
    (i32Parts c value).before=(parts c).before := rfl

theorem i32_after (c : Shape) (value : Int) :
    (i32Parts c value).after=(parts c).after := rfl

theorem i32_body (c : Shape) (value : Int) : (kernelI32 c value).surfaceBody=
    (parts c).before++[.forRangeInt "_" (.constInt 0) (.constInt (c.T : Int)) (.constNat 1)
      (i32Parts c value).body]++(parts c).after := rfl

theorem i32_body_exec (M : Algebra α) (c : Shape) (initial current : State α)
    (value : Int) (n : Nat)
    (ctx : Context M c initial (M.fromInt (some .fp32) value) n current) :
    run (engine M) (i32Parts c value).body
      (current.setReg "_" .int [] (fun _ => (n : Int)))=
      some (bodyExec M c initial current (M.fromInt (some .fp32) value) n ctx).val := by
  by_cases hr : c.reverse=Bool.true
  all_goals
    simp [bodyExec,i32Parts,parts,kernelI32,kernel,ComputeKernel.surfaceBody,Triton.Source.intRange?,
      Option.map,Option.getD,run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,boundaryZero,Option.bind,engine,Algebra.withDefaultPrecision,
      resolvePrecision,ComputeDType.eraseDType,numeric,integralMod,numericLt,natLt,
      bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,hr,
      ctx.mem,ctx.iv,ctx.ik,ctx.iz,ctx.km,ctx.vm,ctx.kvm,ctx.pq,ctx.pk,
      ctx.pw,ctx.pv,ctx.po,ctx.bias,ctx.state,ctx.scalar,FP.Structural.store,
      AttentionForwardTriton.Scatter.masked_scatter_regs]

#axiomsClean i32_body_exec

def i64Parts (c : Shape) (value : Int) : Triton.Source.IntRange :=
  (Triton.Source.intRange? (kernelI64 c value).surfaceBody "_").getD
    ⟨[],.constInt 0,.constInt 0,.constNat 0,[],[]⟩

theorem i64_before (c : Shape) (value : Int) :
    (i64Parts c value).before=(parts c).before := rfl

theorem i64_after (c : Shape) (value : Int) :
    (i64Parts c value).after=(parts c).after := rfl

theorem i64_body (c : Shape) (value : Int) : (kernelI64 c value).surfaceBody=
    (parts c).before++[.forRangeInt "_" (.constInt 0) (.constInt (c.T : Int)) (.constNat 1)
      (i64Parts c value).body]++(parts c).after := rfl

theorem i64_body_exec (M : Algebra α) (c : Shape) (initial current : State α)
    (value : Int) (n : Nat)
    (ctx : Context M c initial (M.fromInt (some .fp32) value) n current) :
    run (engine M) (i64Parts c value).body
      (current.setReg "_" .int [] (fun _ => (n : Int)))=
      some (bodyExec M c initial current (M.fromInt (some .fp32) value) n ctx).val := by
  by_cases hr : c.reverse=Bool.true
  all_goals
    simp [bodyExec,i64Parts,parts,kernelI64,kernel,ComputeKernel.surfaceBody,Triton.Source.intRange?,
      Option.map,Option.getD,run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,boundaryZero,Option.bind,engine,Algebra.withDefaultPrecision,
      resolvePrecision,ComputeDType.eraseDType,numeric,integralMod,numericLt,natLt,
      bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,hr,
      ctx.mem,ctx.iv,ctx.ik,ctx.iz,ctx.km,ctx.vm,ctx.kvm,ctx.pq,ctx.pk,
      ctx.pw,ctx.pv,ctx.po,ctx.bias,ctx.state,ctx.scalar,FP.Structural.store,
      AttentionForwardTriton.Scatter.masked_scatter_regs]

#axiomsClean i64_body_exec

def u64Parts (c : Shape) (value : Nat) : Triton.Source.IntRange :=
  (Triton.Source.intRange? (kernelU64 c value).surfaceBody "_").getD
    ⟨[],.constInt 0,.constInt 0,.constNat 0,[],[]⟩

theorem u64_before (c : Shape) (value : Nat) :
    (u64Parts c value).before=(parts c).before := rfl

theorem u64_after (c : Shape) (value : Nat) :
    (u64Parts c value).after=(parts c).after := rfl

theorem u64_body (c : Shape) (value : Nat) : (kernelU64 c value).surfaceBody=
    (parts c).before++[.forRangeInt "_" (.constInt 0) (.constInt (c.T : Int)) (.constNat 1)
      (u64Parts c value).body]++(parts c).after := rfl

theorem u64_body_exec (M : Algebra α) (c : Shape) (initial current : State α)
    (value : Nat) (n : Nat)
    (ctx : Context M c initial (M.fromNat (some .fp32) value) n current) :
    run (engine M) (u64Parts c value).body
      (current.setReg "_" .int [] (fun _ => (n : Int)))=
      some (bodyExec M c initial current (M.fromNat (some .fp32) value) n ctx).val := by
  by_cases hr : c.reverse=Bool.true
  all_goals
    simp [bodyExec,u64Parts,parts,kernelU64,kernel,ComputeKernel.surfaceBody,Triton.Source.intRange?,
      Option.map,Option.getD,run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,boundaryZero,Option.bind,engine,Algebra.withDefaultPrecision,
      resolvePrecision,ComputeDType.eraseDType,numeric,integralMod,numericLt,natLt,
      bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,hr,
      ctx.mem,ctx.iv,ctx.ik,ctx.iz,ctx.km,ctx.vm,ctx.kvm,ctx.pq,ctx.pk,
      ctx.pw,ctx.pv,ctx.po,ctx.bias,ctx.state,ctx.scalar,FP.Structural.store,
      AttentionForwardTriton.Scatter.masked_scatter_regs]

#axiomsClean u64_body_exec
def oneParts (c : Shape) : Triton.Source.IntRange :=
  (Triton.Source.intRange? (kernelOne c).surfaceBody "_").getD
    ⟨[],.constInt 0,.constInt 0,.constNat 0,[],[]⟩

theorem one_before (c : Shape) :
    (oneParts c).before=(parts c).before := rfl

theorem one_after (c : Shape) :
    (oneParts c).after=(parts c).after := rfl

theorem one_body (c : Shape) : (kernelOne c).surfaceBody=
    (parts c).before++[.forRangeInt "_" (.constInt 0) (.constInt (c.T : Int)) (.constNat 1)
      (oneParts c).body]++(parts c).after := rfl

theorem one_body_exec (M : Algebra α) (c : Shape) (initial current : State α)
    (n : Nat)
    (ctx : Context M c initial (M.literal (some .fp32) .real 1) n current) :
    run (engine M) (oneParts c).body
      (current.setReg "_" .int [] (fun _ => (n : Int)))=
      some (bodyExec M c initial current (M.literal (some .fp32) .real 1) n ctx).val := by
  by_cases hr : c.reverse=Bool.true
  all_goals
    simp [bodyExec,oneParts,parts,kernelOne,kernel,ComputeKernel.surfaceBody,Triton.Source.intRange?,
      Option.map,Option.getD,run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,boundaryZero,Option.bind,engine,Algebra.withDefaultPrecision,
      resolvePrecision,ComputeDType.eraseDType,numeric,integralMod,numericLt,natLt,
      bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,hr,
      ctx.mem,ctx.iv,ctx.ik,ctx.iz,ctx.km,ctx.vm,ctx.kvm,ctx.pq,ctx.pk,
      ctx.pw,ctx.pv,ctx.po,ctx.bias,ctx.state,ctx.scalar,FP.Structural.store,
      AttentionForwardTriton.Scatter.masked_scatter_regs]

#axiomsClean one_body_exec


end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
