import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Targets
import bench.repairs.tritonbench_g.flash_attn.KernelOnly.RealModel

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked
open VeriTile Triton FP.Structural FlashAttn FlashAttn.ActualSource
noncomputable section
namespace Logical

/-- The Real source model also represents its internal -∞ sentinel. Public
Real inputs are exactly ordinary reals, as in the previous theorem. This is a
mathematical carrier condition, not a numerical experiment or FP restriction. -/
def RealInputs (x : Layout) (xs : PublicIO.Inputs KernelReal.R x) : Prop :=
  ∀ port i, ∃ value : ℝ, PublicIO.array xs port i=some value

def realAt {x : Layout} (xs : PublicIO.Inputs KernelReal.R x) (port : Fin 3)
    (b h row col : Nat) : ℝ := (PublicIO.inputAt xs port b h row col).getD 0

def score (x : Layout) (scale : ℝ) (xs : PublicIO.Inputs KernelReal.R x)
    (b h row key : Nat) : ℝ :=
  ((List.range x.shape.dim).map (fun d =>
    (scale*1.44269504*realAt xs 0 b h row d)*realAt xs 1 b h key d)).sum

def denominator (x : Layout) (scale : ℝ) (xs : PublicIO.Inputs KernelReal.R x)
    (b h row : Nat) : ℝ :=
  ((Spec.keys x.shape row).map (fun key => pow2 (score x scale xs b h row key))).sum

def numerator (x : Layout) (scale : ℝ) (xs : PublicIO.Inputs KernelReal.R x)
    (b h row col : Nat) : ℝ :=
  ((Spec.keys x.shape row).map (fun key =>
    pow2 (score x scale xs b h row key)*realAt xs 2 b h key col)).sum

/-- Independent normalized attention and base-2 log-sum-exp formulas. Both
results are explicitly `some`; observing zero cannot conceal an undefined result. -/
def mathematical (x : Layout) (scale : ℝ) (xs : PublicIO.Inputs KernelReal.R x) :
    PublicIO.Outputs KernelReal.R x :=
  fun port => match port with
  | ⟨0,_⟩ => fun i => some (numerator x scale xs i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val /
      denominator x scale xs i.1.val i.2.1.val i.2.2.1.val)
  | ⟨1,_⟩ => fun i => some (Real.log (denominator x scale xs i.1.val i.2.1.val i.2.2.1.val)/Real.log 2)
  | ⟨n+2,h⟩ => False.elim (by simp [PublicIO.outputs] at h)

def wordLeaf (w : Word) : KernelNative.Expr := fun _ => .word w

def outputPayload (M : Algebra B) (dot : Dot B) (encode : A → B) [Inhabited A]
    (x : Layout) (scale : B) (xs : PublicIO.Inputs A x)
    (i : TileIndex [x.shape.batches,x.shape.heads,x.shape.length,x.shape.dim]) : B :=
  match lastWriter x (FlashAttn.address x.o i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val) with
  | some w => output M (parameters x) (rowAccumulator M dot encode x scale xs w.1) w.2
  | none => M.literal (some .fp32) .real 0

def logPayload (M : Algebra B) (dot : Dot B) (encode : A → B) [Inhabited A]
    (x : Layout) (scale : B) (xs : PublicIO.Inputs A x)
    (i : TileIndex [x.shape.batches,x.shape.heads,x.shape.length]) : B :=
  log M (parameters x)
    (rowAccumulator M dot encode x scale xs (i.2.2.1.val/128,i.1.val*x.shape.heads+i.2.1.val))
    ⟨i.2.2.1.val%128,Nat.mod_lt _ (by decide)⟩

/-- The previous replica/owner observation, including ambiguous physical O
addresses under colliding strides, is retained exactly. -/
def nativeObservation (I : KernelNative.Instructions) (x : Layout) :
    (port : Fin (PublicIO.outputs x).length) → TileIndex (PublicIO.outputs x)[port].shape → KernelNative.Expr → Word :=
  fun port => match port with
  | ⟨0,_⟩ => fun i value => KernelNative.eval I (value (KernelNative.readOwner x "O"
      (FlashAttn.address x.o i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val)))
  | ⟨1,_⟩ => fun i value => KernelNative.eval I (value (KernelNative.readOwner x "L"
      (FlashAttn.lAddress x.shape i.1.val i.2.1.val i.2.2.1.val)))
  | ⟨n+2,h⟩ => False.elim (by simp [PublicIO.outputs] at h)

def nativeFormula (I : KernelNative.Instructions) (x : Layout) (scale : Checked.ScalarABI)
    (xs : PublicIO.Inputs Word x) : PublicIO.Outputs Word x :=
  fun port => match port with
  | ⟨0,_⟩ => fun i => nativeObservation I x (0 : Fin 2) i
      (outputPayload KernelNative.algebra KernelNative.dot wordLeaf x (KernelNative.scalar scale) xs i)
  | ⟨1,_⟩ => fun i => nativeObservation I x (1 : Fin 2) i
      (logPayload KernelNative.algebra KernelNative.dot wordLeaf x (KernelNative.scalar scale) xs i)
  | ⟨n+2,h⟩ => False.elim (by simp [PublicIO.outputs] at h)

end Logical
namespace PublicIO

def realIO (x : Layout) (scale : ℝ) : ProgramIO KernelReal.R KernelReal.R :=
  sourceIO KernelReal.algebra KernelReal.dot id x (some scale) (fun _ _ => id)

def nativeIO (I : KernelNative.Instructions) (x : Layout) (scale : Checked.ScalarABI) :
    ProgramIO Word KernelNative.Expr :=
  sourceIO KernelNative.algebra KernelNative.dot Logical.wordLeaf x (KernelNative.scalar scale)
    (Logical.nativeObservation I x)

end PublicIO
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked
