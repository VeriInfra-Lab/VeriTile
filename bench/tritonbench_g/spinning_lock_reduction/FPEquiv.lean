import bench.tritonbench_g.spinning_lock_reduction.Proofs.AreaEquivalence
import bench.tritonbench_g.spinning_lock_reduction.Proofs.Mathematical_FP
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.SpinningLock.FPEquiv
open VeriTile Triton VeriTile.Bench.TritonBenchG.SpinningLock
open scoped VeriTile.Spec
noncomputable section

def original (h : HostArgs) (count : Nat) (stage : Phased.Stage) :
    ProgramView.Implementation fp32 h count stage := ⟨kernel⟩

def tileArea (h : HostArgs) (count : Nat) (stage : Phased.Stage) :
    ProgramView.Implementation fp32 h count stage := ⟨areaKernel⟩

/-- Both actual complete concurrent source stages. Fold constexpr tile area
before the live program-index multiply; CAS old words, retries, barriers and
all source interleavings are retained. The fixed FP32 primitives are unchanged. -/
specification tile_area_equiv (h : HostArgs) (buf : Wide.Buffers h) (stage : Phased.Stage) :
    original h buf.cElements stage ≡[([] : Spec.Assumptions _)] tileArea h buf.cElements stage :=
  Spec.FloatingPoint.ofStructural rfl rfl
    (AreaEquivalence.equivalent fp32 fp32_positive_zero_add h buf stage)

#axiomsClean tile_area_equiv
end
end VeriTile.Bench.Repairs.SpinningLock.FPEquiv
