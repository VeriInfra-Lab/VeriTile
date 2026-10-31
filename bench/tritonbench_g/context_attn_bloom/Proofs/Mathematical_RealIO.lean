import bench.tritonbench_g.context_attn_bloom.Proofs.Mathematical_Flat
import VeriTile.Meta.Specification
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.RealIO
open VeriTile.Triton Math Program RealSetup
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 4000

abbrev Lane (p : Layout) := TileIndex [p.BM,p.D]
def active (p : Layout) (l : Launch) (a : Inputs) (i : Lane p) : Prop :=
  (queryRow p l i.1.val : Int) < sequence a l ∧ i.2.1.val < p.headDim

def write (p : Layout) (l : Launch) (a : Inputs) (i : Lane p) : Nat :=
  (AddressBounds.outputAddress p l a i.1.val i.2.1.val).toNat

def valid (p : Layout) (l : Launch) (a : Inputs) : Prop :=
  0 ≤ a.prompt l.batch ∧ ∀ i j, active p l a i → active p l a j → write p l a i = write p l a j → i = j

structure IO (p : Layout) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  read : BlockState → Inputs
  valid : Launch → Inputs → Prop
  bounds : Launch → Inputs → RegionBounds → Prop
  out : RegionName
  write : Launch → Inputs → Lane p → Nat
  mask : Launch → Inputs → Lane p → Prop

def IO.regionWrites {p : Layout} (io : IO p) (l : Launch) (a : Inputs) : WriteFootprint := fun cell =>
  ∃ i, io.mask l a i ∧ io.out = cell.1 ∧ io.write l a i = cell.2

def IO.writes {p : Layout} (io : IO p) (A : FlatAlloc) (l : Launch) (a : Inputs) : WriteFootprint := fun cell =>
  cell.1 = A.flat ∧ ∃ rg off, io.regionWrites l a (rg,off) ∧ A.addr rg off = cell.2

theorem projection (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool) (scale : ℝ) :
    (kernel p r storage swapped scale).eraseDType.toAlgorithm? =
      .ok (kernel p r storage swapped scale).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType,Program.projects,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

def expected (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : Lane p) : ℝ :=
  Formula.expected p l a scale i.1.val i.2.1.val (count p l a)

/-- Any independently displayed recurrence with these prefix-sum equations has
the same result. The public entry uses this lemma to expose its own formula. -/
theorem recurrence_eq (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d : Nat)
    (f : Nat → ℝ) (h0 : f 0 = 0)
    (hs : ∀ c, f (c+1) = f c *
      (if ((c*p.BN : Nat) : Int) ≤ (queryRow p l i : Int)+a.prompt l.batch then
        Formula.denominator p l a scale i c / Formula.denominator p l a scale i (c+1) else 1) +
      (Formula.numerator p l a scale i d (c+1)-Formula.numerator p l a scale i d c) /
        Formula.denominator p l a scale i (c+1)) :
    ∀ c, f c = Formula.expected p l a scale i d c := by
  intro c
  induction c with
  | zero => exact h0
  | succ c ih => rw [hs,Formula.expected,ih]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.RealIO
