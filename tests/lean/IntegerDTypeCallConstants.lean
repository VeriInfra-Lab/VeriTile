import VeriTile.Triton.DSL.IntegerDTypeCallConstants
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.IntegerDTypeCallConstants
open VeriTile.Triton
noncomputable section

def profile : CallSignature where
  pointers := [("output",.i32)]
  constexpr := ["COUNT","ACC_TYPE"]
  writePorts := ["output"]
def directProfile : CallSignature := {profile with constexpr := ["COUNT"]}
def copied (output : RegionName) (COUNT : Nat) : ComputeKernel :=
  let ACC_TYPE : SourceDType := .i32
  with_triton_call profile call_args [] in triton {
    value = tl.zeros([COUNT], dtype=ACC_TYPE)
    tl.store(output + tl.arange(0, COUNT), value)
  }
def direct (output : RegionName) (COUNT : Nat) : ComputeKernel :=
  with_triton_call directProfile call_args [] in triton {
    value = tl.zeros([COUNT], dtype=tl.int32)
    tl.store(output + tl.arange(0, COUNT), value)
  }
theorem integer_type_exact (output : RegionName) (COUNT : Nat) : copied output COUNT=direct output COUNT := by rfl
#axiomsClean integer_type_exact

/-- error: integer dtype constexpr may occur only in dtype positions -/
#guard_msgs in
example (output : RegionName) (COUNT : Nat) : ComputeKernel :=
  let ACC_TYPE : SourceDType := .i32
  with_triton_call profile call_args [] in triton {
    ACC_TYPE = 0
    tl.store(output, 0)
  }
/-- error: integer dtype constexpr requires a known integer SourceDType constructor -/
#guard_msgs in
example (output : RegionName) (COUNT : Nat) : ComputeKernel :=
  let ACC_TYPE : SourceDType := .fp32
  with_triton_call profile call_args [] in triton {
    value = tl.zeros([COUNT], dtype=ACC_TYPE)
    tl.store(output + tl.arange(0, COUNT), value)
  }
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.IntegerDTypeCallConstants
