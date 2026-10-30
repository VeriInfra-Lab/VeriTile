import bench.repairs.tritonbench_g.fast_layernorm.Proofs.IOAffine
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.RawNative

namespace VeriTile.Bench.Repairs.FastLayernorm.Interface
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelNative
noncomputable section

def forwardObserver (I : Instructions) (S : OutputSchedule) (N B pid : Nat) :
    (o : Fin (forwardOutputs B).length) → TileIndex (forwardOutputs B)[o].shape → Term → Word
  | ⟨0,_⟩ => fun j => eval I (S "y" (pid*N+j.1.val))
  | ⟨1,_⟩ => fun _ => eval I (S "r" pid)
  | ⟨2,_⟩ => fun _ => eval I (S "mu" pid)
  | ⟨o+3,h⟩ => False.elim (by simp [forwardOutputs] at h)
def backwardObserver (I : Instructions) (S : OutputSchedule) (N B pid : Nat) :
    (o : Fin (backwardOutputs B).length) → TileIndex (backwardOutputs B)[o].shape → Term → Word
  | ⟨0,_⟩ => fun j => eval I (S "dx" (pid*N+j.1.val))
  | ⟨o+1,h⟩ => False.elim (by simp [backwardOutputs] at h)
def affineObserver (I : Instructions) (S : OutputSchedule) (pid : Nat) :
    (o : Fin affineOutputs.length) → TileIndex affineOutputs[o].shape → Term → Word
  | ⟨0,_⟩ => fun j => eval I (S "dw" (pid*128+j.1.val))
  | ⟨1,_⟩ => fun j => eval I (S "db" (pid*128+j.1.val))
  | ⟨o+2,h⟩ => False.elim (by simp [affineOutputs] at h)

/-- Native formulas evaluate the independent ordered scalar expressions.
The compiler observation may select distinct schedules for distinct outputs. -/
def nativeForward (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (N B pid : Nat) (eps : ℝ) (bias : Bool) (v : ForwardValues Term B) :
    ChannelValues Word (forwardOutputs B) :=
  fun o j => mapValue (forwardObserver I S N B pid o j) (forwardOutputs B)[o].dtype
    (forwardGraph (algebra R) N B eps bias v o j)
def nativeBackward (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (N B pid : Nat) (v : BackwardValues Term B) : ChannelValues Word (backwardOutputs B) :=
  fun o j => mapValue (backwardObserver I S N B pid o j) (backwardOutputs B)[o].dtype
    (backwardGraph (algebra R) N B v o j)
def nativeAffine (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (rows N wc bc pid : Nat) (bias : Bool) (v : AffineValues Term rows) : ChannelValues Word affineOutputs :=
  fun o j => mapValue (affineObserver I S pid o j) affineOutputs[o].dtype
    (affineGraph (algebra R) rows N pid v o j)

theorem forward_native (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (N B pid : Nat) (eps : ℝ) (bias : Bool) :
    (forwardIO (algebra R) N B pid eps bias).Observes (forwardObserver I S N B pid)
      (nativeForward R I S N B pid eps bias) :=
  ProgramIO.Observes.of_implements (forward_implements (algebra R) N B pid eps bias) _
theorem backward_native (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (N B pid : Nat) :
    (backwardIO (algebra R) N B pid).Observes (backwardObserver I S N B pid)
      (nativeBackward R I S N B pid) :=
  ProgramIO.Observes.of_implements (backward_implements (algebra R) N B pid) _
theorem affine_native (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (rows N wc bc pid : Nat) (bias : Bool) :
    (affineIO (algebra R) rows N wc bc pid bias).Observes (affineObserver I S pid)
      (nativeAffine R I S rows N wc bc pid bias) :=
  ProgramIO.Observes.of_implements (affine_implements (algebra R) rows N wc bc pid bias) _
end
end VeriTile.Bench.Repairs.FastLayernorm.Interface
