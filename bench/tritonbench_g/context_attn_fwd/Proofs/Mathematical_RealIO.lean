import bench.tritonbench_g.context_attn_fwd.Proofs.Mathematical_Flat
import VeriTile.Meta.Specification
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.RealIO
open VeriTile.Triton Formula Program
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 4000

abbrev Lane (p : Layout) := TileIndex [p.BM,p.D]
def active (p : Layout) (l : Launch) (a : Inputs) (i : Lane p) : Prop :=
  (queryRow p l i.1.val : Int) < Formula.sequence p l a

def write (p : Layout) (l : Launch) (a : Inputs) (i : Lane p) : Nat :=
  (AddressBounds.outputAddress p l a i.1.val i.2.1.val).toNat

def valid (p : Layout) (l : Launch) (a : Inputs) : Prop :=
  ∀ i j, active p l a i → active p l a j → write p l a i = write p l a j → i = j

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
  Formula.expected p l a scale i.1.val i.2.1.val

/-- The complete visited window has no remaining prefix filter. -/
def closedExpected (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : Lane p) : ℝ :=
  (∑ j : Fin (count p l a*p.BN), pow2 (Formula.score p l a scale i.1.val j.val) * Formula.value p l a j.val i.2.1.val) /
    ∑ j : Fin (count p l a*p.BN), pow2 (Formula.score p l a scale i.1.val j.val)

theorem expected_closed (p : Layout) (scale : ℝ) : expected p scale = closedExpected p scale := by
  funext l a i
  simp only [expected,Formula.expected,Formula.numerator,Formula.denominator,closedExpected,
    Fin.is_lt,↓reduceIte]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.RealIO
