import VeriTile.Triton.DSL.PythonOptionalConditionals
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.PythonOptionalConditionals
open VeriTile.Triton
noncomputable section

def signature (dtype : FloatDType) : CallSignature where
  floatingPointers := [("X",dtype),("Y",dtype)]
  scalars := []
  constexpr := ["USE","BIAS"]
  readPorts := ["X"]
  writePorts := ["Y"]

def provided (dtype : FloatDType) (X Y : RegionName) (USE BIAS : Bool) : ComputeKernel :=
  with_triton_call (signature dtype) call_args [] optional_pointers [X := Bool.true] in triton {
    if USE and BIAS:
        value = tl.load(X) if X is not None else 0.0
        tl.store(Y, value)
  }

/-- The selected load retains its storage dtype; the nullable fallback adds
no cast or runtime load. All four constexpr flag combinations are checked. -/
theorem complete_provided (dtype : FloatDType) (X Y : RegionName) (USE BIAS : Bool) :
    provided dtype X Y USE BIAS =
      (with_triton_call (signature dtype) call_args [] in triton {
        if USE:
            if BIAS:
                value = tl.load(X)
                tl.store(Y, value)
      }) := by cases USE <;> cases BIAS <;> rfl

def absent (dtype : FloatDType) (X Y : RegionName) (USE BIAS : Bool) : ComputeKernel :=
  with_triton_call (signature dtype) call_args [] optional_pointers [X := Bool.false] in triton {
    value = tl.load(X) if X is not None else 0.0
    tl.store(Y, value)
  }

theorem complete_absent (dtype : FloatDType) (X Y : RegionName) (USE BIAS : Bool) :
    absent dtype X Y USE BIAS =
      (with_triton_call (signature dtype) call_args [] in triton {
        value = 0.0
        tl.store(Y, value)
      }) := by rfl

def reverse (dtype : FloatDType) (X Y : RegionName) (USE BIAS : Bool) : ComputeKernel :=
  with_triton_call (signature dtype) call_args [] optional_pointers [X := Bool.true] in triton {
    value = 0.0 if X is None else tl.load(X)
    tl.store(Y, value)
  }

theorem reverse_provided (dtype : FloatDType) (X Y : RegionName) (USE BIAS : Bool) :
    reverse dtype X Y USE BIAS =
      (with_triton_call (signature dtype) call_args [] in triton {
        value = tl.load(X)
        tl.store(Y, value)
      }) := by rfl

/-- error: optional pointer use requires a dominating presence guard -/
#guard_msgs in
def wrong_branch (dtype : FloatDType) (X Y : RegionName) (USE BIAS : Bool) : ComputeKernel :=
  with_triton_call (signature dtype) call_args [] optional_pointers [X := Bool.false] in triton {
    value = tl.load(X) if X is None else 0.0
    tl.store(Y, value)
  }

/-- error: optional pointer use requires a dominating presence guard -/
#guard_msgs in
def unguarded (dtype : FloatDType) (X Y : RegionName) (USE BIAS : Bool) : ComputeKernel :=
  with_triton_call (signature dtype) call_args [] optional_pointers [X := Bool.true] in triton {
    value = tl.load(X)
    tl.store(Y, value)
  }

#axiomsClean complete_provided
#axiomsClean complete_absent
#axiomsClean reverse_provided
end
end VeriTile.Tests.PythonOptionalConditionals
