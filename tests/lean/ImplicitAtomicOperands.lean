import VeriTile.Triton.DSL.ImplicitAtomicOperands
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.ImplicitAtomicOperands
open VeriTile.Triton
noncomputable section
set_option linter.unusedVariables false

def signature (d : FloatDType) : CallSignature where
  floatingPointers := [("p",d)]
  scalars := [("value",.fp32)]
  constexpr := ["FLAG"]
  readPorts := ["p"]
  writePorts := ["p"]

def source (d : FloatDType) (p : RegionName) (FLAG : Bool) : ComputeKernel :=
  with_triton_call (signature d) call_args [] in triton {
    offset = tl.program_id(0)
    q = p + offset
    if FLAG:
        tl.atomic_add(q, value)
    else:
        r = q + offset
        tl.atomic_add(r, value, mask=True)
  }

/-- The complete caller body equals a separately typed body with explicit
operand conversions, for every retained floating storage type and branch. -/
theorem complete_program (d : FloatDType) (p : RegionName) (FLAG : Bool) :
    source d p FLAG = (with_triton_signature (signature d) in triton {
      offset = tl.program_id(0)
      q = p + offset
      if FLAG:
          tl.atomic_add(q, (value).to(p.dtype.element_ty))
      else:
          r = q + offset
          tl.atomic_add(r, (value).to(p.dtype.element_ty), mask=True)
    }) := by rfl

/-- error: cannot determine floating atomic destination dtype -/
#guard_msgs in
def unknown_pointer (d : FloatDType) (p : RegionName) (FLAG : Bool) : ComputeKernel :=
  with_triton_call (signature d) call_args [] in triton {
    tl.atomic_add(unknown, value)
  }

/-- error: cannot determine floating atomic destination dtype -/
#guard_msgs in
def escaping_alias (d : FloatDType) (p : RegionName) (FLAG : Bool) : ComputeKernel :=
  with_triton_call (signature d) call_args [] in triton {
    if FLAG:
        q = p + 0
    tl.atomic_add(q, value)
  }

/-- error: atomic pointer parameters must remain immutable -/
#guard_msgs in
def parameter_rebinding (d : FloatDType) (p : RegionName) (FLAG : Bool) : ComputeKernel :=
  with_triton_call (signature d) call_args [] in triton {
    p = 1
    tl.atomic_add(p, value)
  }

/-- error: atomic pointer aliases cannot be rebound under control flow -/
#guard_msgs in
def alias_rebinding (d : FloatDType) (p : RegionName) (FLAG : Bool) : ComputeKernel :=
  with_triton_call (signature d) call_args [] in triton {
    q = p + 0
    if FLAG:
        q = 1
    tl.atomic_add(q, value)
  }

/-- error: atomic pointer aliases cannot be rebound under control flow -/
#guard_msgs in
def loop_index_rebinding (d : FloatDType) (p : RegionName) (FLAG : Bool) : ComputeKernel :=
  with_triton_call (signature d) call_args [] in triton {
    q = p + 0
    for q in range(0, 2, 1):
        untouched = q
    tl.atomic_add(q, value)
  }

def integerSignature : CallSignature where
  pointers := [("p",.i64)]
  scalars := [("value",.fp32)]

/-- error: cannot determine floating atomic destination dtype -/
#guard_msgs in
def integer_pointer (p : RegionName) : ComputeKernel :=
  with_triton_call integerSignature call_args [] in triton {
    tl.atomic_add(p, value)
  }

/-- error: atomic pointer aliases cannot be rebound under control flow -/
#guard_msgs in
def tuple_rebinding (d : FloatDType) (p : RegionName) (FLAG : Bool) : ComputeKernel :=
  with_triton_call (signature d) call_args [] in triton {
    q = p
    if FLAG:
        q, other = 1, 2
    tl.atomic_add(q, value)
  }

-- Atomic return-value expressions are rejected by the source grammar.
/-- error: <input>:2:11: expected 'tl.max(' or tritonExpr -/
#guard_msgs in
def atomic_result (d : FloatDType) (p : RegionName) (FLAG : Bool) : ComputeKernel :=
  with_triton_call (signature d) call_args [] in triton {
    previous = tl.atomic_add(p, value)
  }

theorem loop_program (d : FloatDType) (p : RegionName) (FLAG : Bool) :
    (with_triton_call (signature d) call_args [] in triton {
      for k in range(tl.cast(0, tl.int64), tl.cast(2, tl.int64), 1):
          tl.atomic_add(p, value)
    }) = (with_triton_signature (signature d) in triton {
      for k in range(tl.cast(0, tl.int64), tl.cast(2, tl.int64), 1):
          tl.atomic_add(p, (value).to(p.dtype.element_ty))
    }) := by rfl

#axiomsClean loop_program
#axiomsClean complete_program
end
end VeriTile.Tests.ImplicitAtomicOperands
