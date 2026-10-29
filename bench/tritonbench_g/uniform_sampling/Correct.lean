import bench.tritonbench_g.uniform_sampling.Proofs.RealSurface
import bench.tritonbench_g.uniform_sampling.Proofs.Witness
import VeriTile.Meta.Specification

/-! Complete per-program real correctness: standard tagged-cell realization
and frame, together with successful finite flat-memory execution. The scalar
formula retains canonical Philox and its rounded fp32 uniform decoder; only
the outer arithmetic and destination conversions erase precision here.
The signed64 input ABI, launched index domain and active allocation bounds
are explicit. Host generator mutation and device rounding are separate.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.Correctness
open VeriTile VeriTile.Triton TypedSource
noncomputable section

def original (storage : Storage) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (hB : SourceDomain.LegalBlock B) : ComputeKernel :=
  (source storage Bool.false y N seed offset lower upper B hB).eraseDType

def indexCommuted (storage : Storage) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (hB : SourceDomain.LegalBlock B) : ComputeKernel :=
  (source storage Bool.true y N seed offset lower upper B hB).eraseDType

specification uniform_sampling_correctness (storage : Storage) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    Spec.Real (
      (∀ u : BlockState, N < 2^31 → u.pid < SourceDomain.grid N B →
        Random.IntegerType.i64.inRange seed = Bool.true →
        Random.IntegerType.i64.inRange offset = Bool.true →
        ComputeCorrect.RealizesFrame_without_Rounding (α := MemCell)
          (original storage y N seed offset lower upper B hB) u
          (RealSurface.writes y N B u.pid) (RealSurface.expected seed offset lower upper B u)) ∧
      RealIO.Implements (original storage y N seed offset lower upper B hB).toAlgKernel
        y N B seed offset lower upper) := by
  constructor
  · exact RealSurface.realizes_frame storage Bool.false y N seed offset lower upper B hB
  · exact RealIO.source_correct storage Bool.false y N seed offset lower upper B hB

specification index_commuted_correctness (storage : Storage) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    Spec.Real (
      (∀ u : BlockState, N < 2^31 → u.pid < SourceDomain.grid N B →
        Random.IntegerType.i64.inRange seed = Bool.true →
        Random.IntegerType.i64.inRange offset = Bool.true →
        ComputeCorrect.RealizesFrame_without_Rounding (α := MemCell)
          (indexCommuted storage y N seed offset lower upper B hB) u
          (RealSurface.writes y N B u.pid) (RealSurface.expected seed offset lower upper B u)) ∧
      RealIO.Implements (indexCommuted storage y N seed offset lower upper B hB).toAlgKernel
        y N B seed offset lower upper) := by
  constructor
  · exact RealSurface.realizes_frame storage Bool.true y N seed offset lower upper B hB
  · exact RealIO.source_correct storage Bool.true y N seed offset lower upper B hB

#axiomsClean uniform_sampling_correctness
#axiomsClean index_commuted_correctness

/-- Both public real programs execute on a finite output allocation with
all four streams active and a masked final lane. The full values and frame
of these same witnesses are supplied by Witness.real_nonempty. -/
theorem nonempty_instances (storage : Storage) :
    (∃ v, exec (Witness.allocation.flattenKernel
      (original storage "out" 4095 (-1) (-1) 0 1 1024 TypedExecution.block_legal).toAlgKernel)
      (Witness.allocation.flattenState Witness.initial) = some v) ∧
    (∃ v, exec (Witness.allocation.flattenKernel
      (indexCommuted storage "out" 4095 (-1) (-1) 0 1 1024 TypedExecution.block_legal).toAlgKernel)
      (Witness.allocation.flattenState Witness.initial) = some v) := by
  obtain ⟨u, hu, _, _⟩ := Witness.real_nonempty storage Bool.false
  obtain ⟨v, hv, _, _⟩ := Witness.real_nonempty storage Bool.true
  exact ⟨⟨u, hu⟩, ⟨v, hv⟩⟩

#axiomsClean nonempty_instances

end
end VeriTile.Bench.TritonBenchG.UniformSampling.Correctness
