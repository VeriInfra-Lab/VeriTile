import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! A direct load cast to another register's dtype must preserve both the
source storage channel and the numerical conversion. Distinct memory tags
and an arbitrary backend make type erasure observable in these theorems. -/
namespace VeriTile.Tests.RegisterDTypeLoad
open VeriTile.Triton FP.Structural

def direct (X Y : RegionName) : ComputeKernel := triton {
  tl.region X = tl.bfloat16, Y = tl.float16
  target = tl.load(Y + $(0))
  value = tl.load(X + $(0)).to(target.dtype)
}

def explicit (X Y : RegionName) : ComputeKernel := triton {
  tl.region X = tl.bfloat16, Y = tl.float16
  target = tl.load(Y + $(0))
  value = tl.load(X + $(0)).to(tl.float16)
}

theorem direct_preserves_cast (X Y : RegionName) : direct X Y = explicit X Y := by
  rfl

theorem direct_exec {α : Type} [Inhabited α] (M : Algebra α)
    (X Y : RegionName) (s : State α) :
    FP.Structural.exec M (direct X Y) s = some
      ((s.setReg "target" .fp16 [] (fun _ => (s.mem Y 0).read .fp16)).setReg
        "value" .fp16 [] (fun _ => M.cast none .bf16 .fp16 ((s.mem X 0).read .bf16))) := by
  simp [direct, FP.Structural.exec, run, step, evalExpr, FP.Structural.evalOp,
    State.setReg, ofFloat, toFloat]


/-- Both storage parameters must occur in the load-conversion graph. -/
def parametric (dtype out_dtype : FloatDType) (X Y : RegionName) : ComputeKernel := triton {
  tl.region X = DTYPE, Y = OUT_DTYPE
  target = tl.load(Y + $(0))
  value = tl.load(X + $(0)).to(target.dtype)
}

def parametricExplicit (dtype out_dtype : FloatDType) (X Y : RegionName) : ComputeKernel := triton {
  tl.region X = DTYPE, Y = OUT_DTYPE
  target = tl.load(Y + $(0))
  value = tl.load(X + $(0)).to(OUT_DTYPE)
}

theorem parametric_preserves_both_formats (dtype out_dtype : FloatDType) (X Y : RegionName) :
    parametric dtype out_dtype X Y = parametricExplicit dtype out_dtype X Y := by rfl

def masked (X Y : RegionName) (n B : Nat) : ComputeKernel := triton {
  tl.region X = tl.bfloat16, Y = tl.float16
  i = tl.arange(0, $(B))
  target = tl.load(Y + i, mask=i < $(n), other=0)
  value = tl.load(X + i, mask=i < $(n), other=0).to(target.dtype)
}

def maskedExplicit (X Y : RegionName) (n B : Nat) : ComputeKernel := triton {
  tl.region X = tl.bfloat16, Y = tl.float16
  i = tl.arange(0, $(B))
  target = tl.load(Y + i, mask=i < $(n), other=0)
  value = tl.load(X + i, mask=i < $(n), other=0).to(tl.float16)
}

theorem masked_preserves_storage_and_target (X Y : RegionName) (n B : Nat) :
    masked X Y n B = maskedExplicit X Y n B := by rfl

#axiomsClean parametric_preserves_both_formats
#axiomsClean masked_preserves_storage_and_target

#axiomsClean direct_preserves_cast
#axiomsClean direct_exec
end VeriTile.Tests.RegisterDTypeLoad
