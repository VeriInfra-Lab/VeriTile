import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.WideAddress

/-! Independent finite stage boundary. Partials/locks expose precisely the
prefix addressed by this source; allocation padding remains in the frame.
Output retains its complete allocated window because untouched cells are part
of the returned tensor formula, including overlapping output strides. -/
noncomputable section
namespace VeriTile.Bench.Repairs.SpinningLock.Formula
open VeriTile.Bench.TritonBenchG.SpinningLock
open Classical

def partialCount (h : HostArgs) : Nat := h.numSms * h.bm * h.bn

structure Values (V : Type) (h : HostArgs) (outputCount : Nat) where
  partials : Fin (partialCount h) → V
  output : Fin outputCount → V
  locks : Fin h.numSms → I32

def role (h : HostArgs) (stage : Phased.Stage) (p : Nat) : Prop :=
  match stage with
  | .producers => p % h.groupK ≠ 0
  | .consumers => p % h.groupK = 0

def writes (h : HostArgs) (stage : Phased.Stage) (address : Nat) : Prop :=
  ∃ p < h.numSms, role h stage p ∧ ∃ row < h.bm, ∃ col < h.bn,
    row < h.m ∧ p * h.bn + col < h.n ∧
      row * h.strideM + (p * h.bn + col) * h.strideN = address

def stage (zero : V) (h : HostArgs) (which : Phased.Stage)
    (input : Values V h outputCount) : Values V h outputCount where
  partials := fun i =>
    if which = .producers ∧ (i.val / (h.bm * h.bn)) % h.groupK ≠ 0 then zero else input.partials i
  output := fun i => if writes h which i.val then zero else input.output i
  locks := fun i => if which = .producers ∧ i.val % h.groupK ≠ 0 then 1 else input.locks i

/-- Publication is a logical input fact about finite arrays; it is not an
execution, output, safety, or numerical-approximation assumption. -/
def Published (zero : V) (h : HostArgs) (input : Values V h outputCount) : Prop :=
  ∀ p : Fin h.numSms, input.locks p = 1 → ∀ lane (hl : lane < h.bm * h.bn),
    input.partials ⟨p.val * h.bm * h.bn + lane, by
      have hp := Nat.mul_le_mul_right (h.bm*h.bn) (Nat.succ_le_of_lt p.isLt)
      simp only [Nat.succ_mul,Nat.mul_assoc,partialCount] at hp ⊢
      exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left hl _) hp⟩ = zero

def Ready (h : HostArgs) (input : Values V h outputCount) : Prop :=
  ∀ p : Fin h.numSms, p.val % h.groupK ≠ 0 → input.locks p = 1

def Domain (zero : V) (h : HostArgs) (which : Phased.Stage)
    (input : Values V h outputCount) : Prop :=
  Published zero h input ∧ (which = .consumers → Ready h input)

theorem producers_ready (zero : V) (h : HostArgs) (input : Values V h outputCount) :
    Ready h (stage zero h .producers input) := by
  intro p hp
  simp [stage,hp]

end VeriTile.Bench.Repairs.SpinningLock.Formula
