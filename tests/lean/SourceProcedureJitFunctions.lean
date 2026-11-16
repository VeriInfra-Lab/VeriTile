import VeriTile.Triton.DSL.SourceProcedures
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Structural

namespace VeriTile.Tests.SourceProcedureJitFunctions
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000

def duplicateSource : StatementJitFunction :=
  with_triton_params (x) constexpr_args [] in triton {
    return tl.interleave(x, x)
  }

def source : SourceProcedure :=
  with_triton_params (out_ptr, B) constexpr_args [B]
    jit_functions [duplicate := duplicateSource] in triton {
    x = tl.arange(0, B)
    output = duplicate(x)
    scales = tl.arange(0, B // 16)
    output = tl.reshape(output, [(B * 2) // 32, 32])
    scales = tl.reshape(scales, [(B // 16) // 1, 1])
    output = output + scales
    output = tl.reshape(output, [B * 2])
    offsets = tl.arange(0, B * 2)
    tl.store(out_ptr + offsets, output)
  }

def signature : CallSignature where
  pointers := [("out_ptr",.u8)]
  constexpr := ["B"]

def isolated (out : RegionName) (b : Nat) : ComputeKernel :=
  instantiate_source_procedure source signature signature
    procedure_args [out_ptr := out, B := b]

def callerSignature : CallSignature where
  pointers := [("q",.u8),("k",.u8),("v",.u8)]
  constexpr := ["B"]
  writePorts := ["q","k","v"]

def triple (q k v : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call callerSignature call_args [] source_procedures [convert := source] in triton {
    convert(q, B)
    convert(k, B)
    convert(v, B)
  }

/-- One generic source, with a returned JIT helper and symbolic reshape
arithmetic, expands identically in isolated and three-call forms. -/
theorem exact_instances (q k v : RegionName) (b : Nat) :
    (triple q k v b).surfaceBody=(isolated q b).surfaceBody++
      (isolated k b).surfaceBody++(isolated v b).surfaceBody := by rfl

theorem normalized_rows (b : Nat) : (b*2)/32=b/16 := by omega

def bitcastHelper : StatementJitFunction :=
  with_triton_params (x) constexpr_args [] in triton {
    value = x.to(tl.float32, bitcast=True)
    value = value.to(tl.bfloat16)
    return value
  }

def bitcastSource : SourceProcedure :=
  with_triton_params (out_ptr) constexpr_args []
    jit_functions [decode := bitcastHelper] in triton {
    bits = tl.cast(0x21, tl.int32)
    value = decode(bits)
    tl.store(out_ptr + 0, value)
  }

def bitcastKernel (out : RegionName) : ComputeKernel :=
  instantiate_source_procedure bitcastSource
    signature ({pointers := [("out_ptr",.bf16)]} : CallSignature)
    procedure_args [out_ptr := out]

/-- Returned bitcasts retain the exact payload and BF16 conversion, including
when the surrounding source is elaborated through a procedure call. -/
theorem bitcast_value {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (s : FP.Structural.State α) (out : RegionName) :
    ∃ t,FP.Structural.exec M (bitcastKernel out) s=some t ∧
      (t.mem out 0).read .bf16=M.cast none .real .bf16 (M.fp32Bits ⟨0x21⟩) := by
  cases out
  simp [bitcastKernel,FP.Structural.exec,ComputeKernel.surfaceBody,
    FP.Structural.run,FP.Structural.step,FP.Structural.evalExpr,FP.Structural.evalComputeOp,
    FP.Structural.evalOp_unfold,FP.Structural.store,FP.Structural.numeric,
    FP.Structural.bop,FP.Structural.natLt,FP.Structural.numericLt,
    FP.Structural.integralMod,FP.Structural.integralFloorDiv,FP.Structural.numericEq,
    FP.Structural.ofFloat,FP.Structural.toFloat,ComputeDType.eraseDType,
    FP.Structural.State.setReg,FP.Structural.State.write,
    TileShape.allIndices,Region.cast,Option.bind,Broadcast.leftIndex,Broadcast.rightIndex]

theorem reject_rebound_helper : True := by
  fail_if_success
    let invalid : SourceProcedure :=
      with_triton_params (out_ptr) constexpr_args [] jit_functions [duplicate := duplicateSource] in triton {
        duplicate = tl.arange(0, 32)
        tl.store(out_ptr + 0, 1)
      }
  trivial

theorem reject_parameter_collision : True := by
  fail_if_success
    let invalid : SourceProcedure :=
      with_triton_params (duplicate) constexpr_args [] jit_functions [duplicate := duplicateSource] in triton {
        x = duplicate(1)
      }
  trivial

#axiomsClean bitcast_value
#axiomsClean exact_instances
#axiomsClean normalized_rows
end
end VeriTile.Tests.SourceProcedureJitFunctions
