import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Formulas
import VeriTile.Triton.Float.ProgramEquivalence
import VeriTile.Spec

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PairIO
open VeriTile Triton FP.Structural FlashAttn FlashAttn.ActualSource
noncomputable section

/-- The existing head-offset normalization is applied to every repaired
independent head pitch. Both values select complete actual source programs. -/
structure Program (x : Layout) where
  swapped : Bool

def commutedInvocation (M : Algebra A) (dot : Dot A) (x : Layout) (scale : A)
    (job : ActualSource.Program) : Invocation A :=
  { PublicIO.invocation M dot x scale job with kernel := ActualSource.indexCommuted (parameters x) }

def Program.source {x : Layout} (p : Program x) : ComputeKernel :=
  if p.swapped then ActualSource.indexCommuted (parameters x) else ActualSource.kernel (parameters x)

def Program.io {x : Layout} (p : Program x) (I : KernelNative.Instructions)
    (scale : Checked.ScalarABI) : ProgramIO Word KernelNative.Expr :=
  { PublicIO.nativeIO I x scale with
    schedule := fun _ =>
      (programs (parameters x)).map (fun job =>
        if p.swapped then commutedInvocation KernelNative.algebra KernelNative.dot x (KernelNative.scalar scale) job
        else PublicIO.invocation KernelNative.algebra KernelNative.dot x (KernelNative.scalar scale) job) }

def variant {x : Layout} (I : KernelNative.Instructions) (scale : Checked.ScalarABI)
    (left right : Program x) : ProgramIO.Variant (left.io I scale) where
  schedule := (right.io I scale).schedule
  observe := (left.io I scale).observe

/-- Both full grids safely succeed on all original raw words and scalar ABI
values, agree on every observed typed O/L cell (including actual last writers),
and preserve both complete exterior frames. The native instruction model is
fixed explicitly, rather than replaced by a whole-output numerical premise. -/
def Equivalent {x : Layout} (left right : Program x) : Prop :=
  ∀ (I : KernelNative.Instructions) (scale : Checked.ScalarABI),
    (left.io I scale).Equivalent (variant I scale left right)

instance {x : Layout} : Spec.ProgramSyntax (Program x) where
  Statement := ComputeStmt
  Signature := List IOChannel × List IOChannel
  signature _ := (PublicIO.inputs x,PublicIO.outputs x)
  body p := (programs (parameters x)).flatMap (fun _ => p.source.surfaceBody)
  structural := some Equivalent
  sameContext := fun left right => left=right

def original (x : Layout) : Program x := ⟨Bool.false⟩
def candidate (x : Layout) : Program x := ⟨Bool.true⟩
def assumptions : Spec.Assumptions ComputeStmt := []

end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PairIO
