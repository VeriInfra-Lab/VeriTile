import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.RuntimeIntegerScalarBody
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def signatureU32 : CallSignature where
  scalars := [("x",.fp32),("scale",.u32)]

def sourceU32 (scale : Nat) : ComputeKernel :=
  with_triton_call signatureU32 call_args [scale := scale] in
  triton {
    value = x * scale
    commuted = scale * x
  }

theorem executionU32 {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (x : α) (scale : Nat)
    (hx : s.regs .real [] "x"=some (fun _ => x)) :
    FP.Structural.exec (engine M) (sourceU32 scale) s=
      some ((s.setReg "value" .real []
        (fun _ => M.binary (some .fp32) .real .mul x (M.fromNat (some .fp32) scale))).setReg "commuted" .real []
        (fun _ => M.binary (some .fp32) .real .mul (M.fromNat (some .fp32) scale) x)) := by
  simp [FP.Structural.exec,sourceU32,ComputeKernel.surfaceBody,run,FP.Structural.step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,Algebra.withDefaultPrecision,
    resolvePrecision,ComputeDType.eraseDType,numeric,ofFloat,toFloat,
    State.setReg,hx,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

#axiomsClean executionU32

def signatureU64 : CallSignature where
  scalars := [("x",.fp32),("scale",.u64)]

def sourceU64 (scale : Nat) : ComputeKernel :=
  with_triton_call signatureU64 call_args [scale := scale] in
  triton {
    value = x * scale
    commuted = scale * x
  }

theorem executionU64 {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (x : α) (scale : Nat)
    (hx : s.regs .real [] "x"=some (fun _ => x)) :
    FP.Structural.exec (engine M) (sourceU64 scale) s=
      some ((s.setReg "value" .real []
        (fun _ => M.binary (some .fp32) .real .mul x (M.fromNat (some .fp32) scale))).setReg "commuted" .real []
        (fun _ => M.binary (some .fp32) .real .mul (M.fromNat (some .fp32) scale) x)) := by
  simp [FP.Structural.exec,sourceU64,ComputeKernel.surfaceBody,run,FP.Structural.step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,Algebra.withDefaultPrecision,
    resolvePrecision,ComputeDType.eraseDType,numeric,ofFloat,toFloat,
    State.setReg,hx,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

#axiomsClean executionU64

def signatureI32 : CallSignature where
  scalars := [("x",.fp32),("scale",.i32)]

def sourceI32 (scale : Int) : ComputeKernel :=
  with_triton_call signatureI32 call_args [scale := scale] in
  triton {
    value = x * scale
    commuted = scale * x
  }

theorem executionI32 {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (x : α) (scale : Int)
    (hx : s.regs .real [] "x"=some (fun _ => x)) :
    FP.Structural.exec (engine M) (sourceI32 scale) s=
      some ((s.setReg "value" .real []
        (fun _ => M.binary (some .fp32) .real .mul x (M.fromInt (some .fp32) scale))).setReg "commuted" .real []
        (fun _ => M.binary (some .fp32) .real .mul (M.fromInt (some .fp32) scale) x)) := by
  simp [FP.Structural.exec,sourceI32,ComputeKernel.surfaceBody,run,FP.Structural.step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,Algebra.withDefaultPrecision,
    resolvePrecision,ComputeDType.eraseDType,numeric,ofFloat,toFloat,
    State.setReg,hx,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

#axiomsClean executionI32

def signatureI64 : CallSignature where
  scalars := [("x",.fp32),("scale",.i64)]

def sourceI64 (scale : Int) : ComputeKernel :=
  with_triton_call signatureI64 call_args [scale := scale] in
  triton {
    value = x * scale
    commuted = scale * x
  }

theorem executionI64 {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (x : α) (scale : Int)
    (hx : s.regs .real [] "x"=some (fun _ => x)) :
    FP.Structural.exec (engine M) (sourceI64 scale) s=
      some ((s.setReg "value" .real []
        (fun _ => M.binary (some .fp32) .real .mul x (M.fromInt (some .fp32) scale))).setReg "commuted" .real []
        (fun _ => M.binary (some .fp32) .real .mul (M.fromInt (some .fp32) scale) x)) := by
  simp [FP.Structural.exec,sourceI64,ComputeKernel.surfaceBody,run,FP.Structural.step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,Algebra.withDefaultPrecision,
    resolvePrecision,ComputeDType.eraseDType,numeric,ofFloat,toFloat,
    State.setReg,hx,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

#axiomsClean executionI64

def constexprSignature : CallSignature where
  scalars := [("x",.fp32)]
  constexpr := ["scale"]

def constexprSource (scale : Nat) : ComputeKernel :=
  with_triton_call constexprSignature call_args [] in
  triton {
    value = x * scale
  }

/-- A constexpr value remains a literal; the runtime-ABI fix must not insert
an integer conversion into compile-time source constants. -/
theorem constexpr_execution {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (x : α) (scale : Nat)
    (hx : s.regs .real [] "x"=some (fun _ => x)) :
    FP.Structural.exec (engine M) (constexprSource scale) s=
      some (s.setReg "value" .real []
        (fun _ => M.binary (some .fp32) .real .mul x (M.literal (some .fp32) .real scale))) := by
  simp [FP.Structural.exec,constexprSource,ComputeKernel.surfaceBody,run,FP.Structural.step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,Algebra.withDefaultPrecision,
    resolvePrecision,ComputeDType.eraseDType,numeric,ofFloat,toFloat,
    State.setReg,hx,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

#axiomsClean constexpr_execution

end
end VeriTile.Tests.RuntimeIntegerScalarBody
