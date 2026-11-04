import VeriTile.Triton.DSL.SourceProcedures
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
open VeriTile.Triton
namespace VeriTile.Tests.SourceProcedures
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000
def helper : SourceProcedure :=
  triton_source_procedure (x_ptr,out_ptr,N,ROT) constexpr_args [ROT] fp32_constants [] {
    N = tl.cast(N, tl.int64)
    x = tl.load(x_ptr + 0).to(tl.float32)
    if ROT:
        x = x + 1.0
    tl.store(out_ptr + 0, x)
  }
def abi : CallSignature where
  pointers := [("x_ptr",.fp32),("q_ptr",.fp16),("k_ptr",.fp32)]
  scalars := [("N",.integer)]
def caller (x_ptr q_ptr k_ptr : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_call abi call_args [N := N] fp32_constants []
    source_procedures [helper := helper] in triton {
    helper(x_ptr=x_ptr, out_ptr=q_ptr, N=N, ROT=True)
    helper(x_ptr=x_ptr, out_ptr=k_ptr, N=N, ROT=False)
  }

def isolated (half : Bool) (x_ptr out_ptr : RegionName) (N : Nat) (ROT : Bool) : ComputeKernel :=
  instantiate_source_procedure helper signature
    ({pointers := [("x_ptr",.fp32),("out_ptr",if half then .fp16 else .fp32)],scalars := [("N",.integer)],constexpr := ["ROT"]} : CallSignature)
    procedure_args [x_ptr := x_ptr,out_ptr := out_ptr,N := N,ROT := ROT]
theorem exact_call_instances (x q k : RegionName) (n : Nat) : (caller x q k n).surfaceBody =
  (isolated Bool.true x q n Bool.true).surfaceBody ++ (isolated Bool.false x k n Bool.false).surfaceBody := by
  rfl

theorem reject_constexpr_write : True := by
  fail_if_success
    let invalid : SourceProcedure := triton_source_procedure (N) constexpr_args [N] fp32_constants [] {
      N = tl.cast(N, tl.int64)
    }
  trivial

theorem reject_hidden_caller_capture : True := by
  fail_if_success
    let invalid : SourceProcedure := triton_source_procedure (X) constexpr_args [] fp32_constants [] {
      value = tl.load(hidden + 0)
    }
  trivial

theorem reject_missing_argument (x_ptr q_ptr k_ptr : RegionName) (N : Nat) : True := by
  fail_if_success
    let invalid : ComputeKernel :=
      with_triton_call abi call_args [N := N] fp32_constants [] source_procedures [helper := helper] in triton {
        helper(x_ptr=x_ptr, out_ptr=q_ptr, ROT=True)
      }
  trivial

theorem reject_runtime_as_constexpr (x_ptr q_ptr k_ptr : RegionName) (N : Nat) : True := by
  fail_if_success
    let invalid : ComputeKernel :=
      with_triton_call abi call_args [N := N] fp32_constants [] source_procedures [helper := helper] in triton {
        helper(x_ptr=x_ptr, out_ptr=q_ptr, N=N, ROT=N)
      }
  trivial

#axiomsClean exact_call_instances
end
end VeriTile.Tests.SourceProcedures
