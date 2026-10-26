import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.SourceFunction
open VeriTile.Triton
noncomputable section

def relu : SourceFunction := triton_function relu_forward(x) in triton {
  return tl.where(x > 0, x, 0)
}
def add : SourceFunction := triton_function add_values(x, y) in triton {
  return x.to(tl.float32) + y.to(tl.float32)
}
def composed : SourceFunction := triton_function composed_value(x, y) in triton {
  return add_values(x, y)
}
def signature : CallSignature where
  pointers := [("X",.fp16),("Y",.fp32)]
  functions := [relu,add,composed]

def pasted (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    y = relu_forward(x)
    z = composed_value(y, y)
    tl.store(Y, z)
  }

def direct (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({signature with functions := []} : CallSignature) in triton {
    x = tl.load(X)
    y = tl.where(x > 0, x, 0)
    z = y.to(tl.float32) + y.to(tl.float32)
    tl.store(Y, z)
  }

/-- Whole executable equality includes FP16 input, both explicit FP32 casts,
the original FP comparison/select and destination conversion. No helper port. -/
theorem complete_expansion (X Y : RegionName) : pasted X Y = direct X Y := rfl

/-- error: source helper arguments must be named source values -/
#guard_msgs in
def duplicateLoad (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    y = relu_forward(tl.load(X))
    tl.store(Y, y)
  }

/-- error: source helper argument count differs from its declaration -/
#guard_msgs in
def wrongArity (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    y = add_values(x)
    tl.store(Y, y)
  }

/-- error: source helpers may only occur in direct calls -/
#guard_msgs in
def helperAsValue (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = relu_forward
    tl.store(Y, x)
  }

def accidental : SourceFunction := triton_function accidental(x) in triton {
  return x + outer
}
def recursiveHelper : SourceFunction := triton_function recurse(x) in triton {
  return recurse(x)
}

/-- error: unbound name in source helper: outer -/
#guard_msgs in
def capturedLean (X Y : RegionName) (outer : ℝ) : ComputeKernel :=
  with_triton_signature ({signature with functions := [accidental]} : CallSignature) in triton {
    x = tl.load(X)
    y = accidental(x)
    tl.store(Y, y)
  }

/-- error: recursive source helpers are unsupported -/
#guard_msgs in
def recursive (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({signature with functions := [recursiveHelper]} : CallSignature) in triton {
    x = tl.load(X)
    y = recurse(x)
    tl.store(Y, y)
  }

/-- error: multi-result source helper calls are unsupported -/
#guard_msgs in
def unknownTupleHelper (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({signature with functions := []} : CallSignature) in triton {
    x = tl.load(X)
    a, b, c = missing_helper(x)
    tl.store(Y, x)
  }

/-- error: multi-result source helper calls are unsupported -/
#guard_msgs in
def unknownPairHelper (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({signature with functions := []} : CallSignature) in triton {
    x = tl.load(X)
    a, b = missing_helper(x)
    tl.store(Y, x)
  }

def identitySilu : SourceFunction := triton_function silu (x) in triton {
  return x
}
def shorthandSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  functions := [identitySilu]

def registeredShorthand (X Y : RegionName) : ComputeKernel :=
  with_triton_signature shorthandSignature in triton {
    x = tl.load(X)
    y = silu(x)
    tl.store(Y, y)
  }
def explicitIdentity (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({shorthandSignature with functions := []} : CallSignature) in triton {
    x = tl.load(X)
    y = x
    tl.store(Y, y)
  }

/-- A same-named legacy shorthand cannot override the checked source helper. -/
theorem registered_definition_owns_callee (X Y : RegionName) :
    registeredShorthand X Y = explicitIdentity X Y := rfl

def spliceHelper : SourceFunction := triton_function splice(x) in triton {
  return $(x)
}
/-- error: Lean antiquotes do not belong in source helpers -/
#guard_msgs in
def hiddenSplice (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({signature with functions := [spliceHelper]} : CallSignature) in triton {
    x = tl.load(X)
    y = splice(x)
    tl.store(Y, y)
  }

def namespaceCapture : SourceFunction := triton_function namespace_capture(x) in triton {
  return tl.hidden
}
/-- error: unbound name in source helper: tl.hidden -/
#guard_msgs in
def namespaceCannotSupplyValue (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({signature with functions := [namespaceCapture]} : CallSignature) in triton {
    x = tl.load(X)
    y = namespace_capture(x)
    tl.store(Y, y)
  }

#axiomsClean registered_definition_owns_callee
#axiomsClean complete_expansion
end
end VeriTile.Tests.SourceFunction
