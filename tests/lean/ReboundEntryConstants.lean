import VeriTile.Triton.DSL.ReboundEntryConstants
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.ReboundEntryConstants
open VeriTile.Triton
noncomputable section

def profile : CallSignature where
  pointers := [("output",.i32)]
  constexpr := ["GROUP_M"]
  writePorts := ["output"]
def copied (output : RegionName) (GROUP_M : Nat) : ComputeKernel :=
  with_triton_call profile call_args [] in triton {
    width = GROUP_M * 2
    GROUP_M = tl.program_id(0) + GROUP_M
    value = GROUP_M + width
    tl.store(output, value)
  }
def directProfile : CallSignature where
  pointers := [("output",.i32)]
  constexpr := ["entry"]
  writePorts := ["output"]
def direct (output : RegionName) (entry : Nat) : ComputeKernel :=
  with_triton_call directProfile call_args [] in triton {
    width = entry * 2
    GROUP_M = tl.program_id(0) + entry
    value = GROUP_M + width
    tl.store(output, value)
  }
/-- Every initial use reads the immutable call value; later uses read the
actual runtime assignment. There is no added executable binding statement. -/
theorem entry_and_runtime_exact (output : RegionName) (GROUP_M : Nat) :
    copied output GROUP_M=direct output GROUP_M := by rfl
#axiomsClean entry_and_runtime_exact

/-- error: entry constexpr rebinding requires a top-level plain assignment -/
#guard_msgs in
example (output : RegionName) (GROUP_M : Nat) : ComputeKernel :=
  with_triton_call profile call_args [] in triton {
    if tl.program_id(0) == 0:
        GROUP_M = 1
    GROUP_M = GROUP_M + 2
    tl.store(output, GROUP_M)
  }
/-- error: entry constexpr rebinding requires a top-level plain assignment -/
#guard_msgs in
example (output : RegionName) (GROUP_M : Nat) : ComputeKernel :=
  with_triton_call profile call_args [] in triton {
    for GROUP_M in range(0, 3, 1):
        tl.store(output, GROUP_M)
    GROUP_M = GROUP_M + 1
  }
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.ReboundEntryConstants
