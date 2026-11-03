import VeriTile.Triton.DSL.ModuleFloatConstants
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.ModuleFloatConstants
open VeriTile.Triton
noncomputable section

def profile : CallSignature where
  pointers := [("x",.fp32),("y",.fp32)]
  sourceConstants := [("factor","1.44269504")]
  readPorts := ["x"]
  writePorts := ["y"]

def bound (x y : RegionName) : ComputeKernel :=
  with_triton_signature profile in triton {
    value = tl.load(x)
    value = value * factor
    tl.store(y, value)
  }

def directProfile : CallSignature where
  pointers := [("x",.fp32),("y",.fp32)]
  readPorts := ["x"]
  writePorts := ["y"]

def direct (x y : RegionName) : ComputeKernel :=
  with_triton_signature directProfile in triton {
    value = tl.load(x)
    value = value * 1.44269504
    tl.store(y, value)
  }

theorem literal_program_exact (x y : RegionName) : bound x y=direct x y := by rfl
#axiomsClean literal_program_exact

/-- error: module floating constants cannot be reassigned -/
#guard_msgs in
example (x y : RegionName) : ComputeKernel :=
  with_triton_signature profile in triton {
    factor = 0.0
    tl.store(y, tl.load(x))
  }

/-- error: a loop cannot shadow a module constant -/
#guard_msgs in
example (x y : RegionName) : ComputeKernel :=
  with_triton_signature profile in triton {
    for factor in range(2):
        tl.store(y, tl.load(x))
  }

def colliding : CallSignature where
  pointers := [("x",.fp32),("y",.fp32)]
  sourceConstants := [("x","1.44269504")]

/-- error: module constants must be distinct from parameters and each other -/
#guard_msgs in
example (x y : RegionName) : ComputeKernel :=
  with_triton_signature colliding in triton {
    tl.store(y, tl.load(x))
  }

def duplicate : CallSignature where
  pointers := [("x",.fp32),("y",.fp32)]
  sourceConstants := [("factor","1.44269504"),("factor","2.0")]

/-- error: module constants must be distinct from parameters and each other -/
#guard_msgs in
example (x y : RegionName) : ComputeKernel :=
  with_triton_signature duplicate in triton {
    tl.store(y, tl.load(x) * factor)
  }

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.ModuleFloatConstants
