import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Structural

namespace VeriTile.Tests.CallArguments
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("N",.i32)]
  constexpr := ["B"]

def copy (X Y : RegionName) (n B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [N := n] in triton {
    """A function docstring with a } delimiter and 'quotes'.

    It has no kernel instruction.
    """
    offsets = tl.arange(0, B)
    value = tl.load(X + offsets, mask=offsets < N, other=0.)
    tl.store(Y + offsets, value, mask=offsets < N)
  }

/-- A runtime size is bound by the invocation, while the quoted body remains
ordinary Python. Its masked loads and stores remain in the executable kernel. -/
theorem source_binding (X Y : RegionName) (n B : Nat) : copy X Y n B = (triton {
    tl.region X = tl.float32, Y = tl.float32
    offsets = tl.arange(0, B)
    value = tl.load(X + offsets, mask=offsets < $(n), other=0.0)
    tl.store(Y + offsets, value, mask=offsets < $(n))
  }) := rfl

def signedSignature : CallSignature where
  pointers := [("Y",.i64)]
  scalars := [("N",.i64)]

def signed (Y : RegionName) (n : Int) : ComputeKernel :=
  with_triton_call signedSignature call_args [N := n] in triton {
    value = N
    tl.store(Y, value)
  }

/-- Negative scalar values retain an Int carrier rather than silently taking
an absolute value or becoming an unsigned natural index. -/
theorem signed_binding (Y : RegionName) (n : Int) : signed Y n = (triton {
    tl.region Y = tl.int64
    value = $((n : Int))
    tl.store(Y, value.to(tl.int64))
  }) := rfl

/-- error: per-call specialization does not yet support reassigned scalar arguments -/
#guard_msgs in
def reassigned (X Y : RegionName) (n B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [N := n] in triton {
    N += 1
    value = tl.load(X)
    tl.store(Y, value)
  }

/-- error: every integer runtime scalar needs an explicit call binding -/
#guard_msgs in
def missing (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    value = tl.load(X)
    tl.store(Y, value)
  }

/-- error: per-call specialization does not yet support reassigned scalar arguments -/
#guard_msgs in
def shadowedLoop (X Y : RegionName) (n B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [N := n] in triton {
    for N in range(B):
      value = tl.load(X)
      tl.store(Y, value)
  }

noncomputable section

def castSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]

def castCopy (X Y : RegionName) : ComputeKernel :=
  with_triton_signature castSignature in triton {
    value = tl.load(X).to(tl.float32)
    tl.store(Y, value)
  }

/-- A cast cannot erase the source storage interpretation. These two opaque
callbacks remain distinct even when a concrete native model makes them identities. -/
theorem cast_keeps_fp32_load {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (X Y : RegionName) (s : FP.Structural.State α) :
    ∃ t, FP.Structural.exec M (castCopy X Y) s = some t ∧
      t.mem Y 0 = FP.Structural.Cell.mk .real
        (M.cast (some .fp32) .real .real (M.fp32Load ((s.mem X 0).read .real))) := by
  simp [castCopy,FP.Structural.exec,FP.Structural.run,FP.Structural.step,
    FP.Structural.evalExpr,FP.Structural.evalComputeOp,FP.Structural.evalOp_unfold,
    FP.Structural.store,FP.Structural.evalOffsets,ComputeKernel.surfaceBody,
    ComputeDType.eraseDType,Region.cast,Option.bind,FP.Structural.ofFloat,FP.Structural.toFloat]

#axiomsClean cast_keeps_fp32_load
end

#axiomsClean source_binding
#axiomsClean signed_binding
end VeriTile.Tests.CallArguments
