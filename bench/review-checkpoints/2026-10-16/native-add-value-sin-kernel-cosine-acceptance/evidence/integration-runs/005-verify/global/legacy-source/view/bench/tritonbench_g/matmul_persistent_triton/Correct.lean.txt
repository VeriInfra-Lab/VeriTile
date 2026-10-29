import bench.tritonbench_g.matmul_persistent_triton.Kernels
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealFull
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealIOCoverage
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealFiniteWitness
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealZeroK
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealSurface
import VeriTile.Spec

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
open VeriTile.Triton
noncomputable section

/-- Empty K leaves all memory unchanged, including preexisting nonzero output.
The standard region surface uses an empty write map, so its frame preserves
every cell. The separate universal finite-flat contract also proves successful
execution under explicit allocation and undef conditions. Both storage profiles
and variants are covered; no zero-matrix write is asserted. -/
specification zero_K_preserves_memory (p : Parameters) (storage : TypedSource.Storage) (v : Bool)
    (hK : p.K = 0) (hBK : 0 < p.BK) :
    VeriTile.Spec.Real
      (RealSurface.FlatPreservesMemory (kernel p storage v).eraseDType.toAlgKernel ∧
        ∀ s, ComputeCorrect.RealizesFrame_without_Rounding (α := MemCell)
          (kernel p storage v).eraseDType s RealSurface.emptyWrites (fun _ => .real 0)) :=
  ⟨RealSurface.zero_flat_preserves p storage v hK hBK,
    fun s => RealSurface.zero_realizes_frame p storage v s hK hBK⟩

/-- Complete source real-erasure correctness for both storage profiles and
variants. Independent original A/B matrices determine every assigned active
GEMM output. The finite-flat contract includes actual host-grid pins, original
tagged input cells, successful execution, allocation safety, input preservation
and the full raw memory frame. The standard region realization additionally
exposes every assigned active output through exact MemCell equality and the
same complete frame; it is derived from actual successful source execution.
Positive dimensions/block sizes, output/input region separation and logical
output-stride injectivity are explicit. Nat/Int arithmetic is unbounded.
Native legal tile/dtype/launch admission, fixed-width integer intermediate
correspondence, GPU execution and IEEE numerical correctness remain unproved. -/
specification real_correct (p : Parameters) (storage : TypedSource.Storage) (v : Bool)
    (h : RealIO.Admissible p) :
    VeriTile.Spec.Real
      (RealIO.Implements p (kernel p storage v).eraseDType.toAlgKernel ∧
        ∀ (s : BlockState) (xs : RealMath.Inputs),
          RealIO.Launch p s → RealInput.Represents p xs s →
          ComputeCorrect.RealizesFrame_without_Rounding (α := MemCell)
            (kernel p storage v).eraseDType s (RealSurface.writes p (s.pids 0))
            (RealSurface.expected p xs (s.pids 0))) :=
  ⟨RealFull.storage_correct p storage v h,
    fun s xs hl hi => RealSurface.realizes_frame p storage v h s xs hl hi⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
