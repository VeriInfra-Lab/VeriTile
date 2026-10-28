import bench.repairs.tritonbench_g.decay_cumsum.Proofs.RealExecutionBridge
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.Geometry
import bench.tritonbench_g.decay_cumsum.Proofs.ForwardCorrect
import bench.tritonbench_g.decay_cumsum.Proofs.PrepareCorrect
import bench.tritonbench_g.decay_cumsum.Proofs.BackwardCorrect

/-! Reuse the original complete per-program Real execution and frame proofs.
Their size/separation premises follow from independent packed allocations and
the complete/short launch geometry; no original proof is copied or weakened.
-/
namespace VeriTile.Bench.Repairs.DecayCumsum.RealTiles
open VeriTile VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DecayCumsum
noncomputable section

def legacyForwardIO (h : Call) (p : Phase) (scale : ℝ) :=
  ForwardCorrect.originalIO .fp32 "g" "go"
    h.headStride h.width 1 h.B h.H h.T p.span h.BK h.width scale
def legacyPrepareIO (h : Call) (p : Phase) (scale : ℝ) :=
  PrepareCorrect.originalIO .fp32 .fp32 "q" "k" "g" "qg" "kg"
    h.headStride h.width 1 h.B h.H h.T h.width p.span h.BK scale
def legacyBackwardIO (h : Call) (p : Phase) (scale : ℝ) :=
  BackwardCorrect.originalIO .fp32 .fp32 "dqi" "dqo" "dki" "dko" "q" "k" "g" "dg"
    h.headStride h.width 1 h.B h.H h.T h.width p.span h.BK scale

theorem legacy_forward_complete (h : Call) (p : Phase) (scale : ℝ) (positive : 0<h.DK) :
    StreamEmitMasked3DKernelIO₁.Implements (legacyForwardIO h p scale)
      (fun _ _ _ xs t j => ForwardCorrect.scan p.span h.BK xs t j) := by
  exact ForwardCorrect.original_correct .fp32 "g" "go"
    h.headStride h.width 1 h.B h.H h.T p.span h.BK h.width scale
    (by decide) (Geometry.tile_le_width h positive)

theorem legacy_prepare_complete (h : Call) (p : Phase) (hp : p∈phases h)
    (scale : ℝ) (positive : 0<h.DK) :
    StreamGroupedEmitMasked3DKernelIO.Implements (legacyPrepareIO h p scale)
      (fun _ _ _ xs o t j => PrepareCorrect.values p.span h.BK scale xs o t j) := by
  exact PrepareCorrect.original_correct .fp32 .fp32 "q" "k" "g" "qg" "kg"
    h.headStride h.width 1 h.B h.H h.T h.width p.span h.BK scale
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (Geometry.tile_le_width h positive) (Geometry.phase_positive h p hp).2.1

theorem legacy_backward_complete (h : Call) (p : Phase) (hp : p∈phases h)
    (scale : ℝ) (positive : 0<h.DK) :
    StreamGroupedEmitMasked3DKernelIO.Implements (legacyBackwardIO h p scale)
      (fun _ _ _ xs o t j => BackwardCorrect.values p.span h.BK xs o t j) := by
  exact BackwardCorrect.original_correct .fp32 .fp32 "dqi" "dqo" "dki" "dko" "q" "k" "g" "dg"
    h.headStride h.width 1 h.B h.H h.T h.width p.span h.BK scale
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (Geometry.tile_le_width h positive) (Geometry.phase_positive h p hp).2.1


def forwardIO (h : Call) (p : Phase) (scale : ℝ) :=
  { legacyForwardIO h p scale with
    kernel := (Kernels.source .forward h p scale).eraseDType
    projection := by
      simp only [ComputeKernel.eraseDType,Kernels.source_projects,
        ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody] }

theorem forward_source (h : Call) (p : Phase) (scale : ℝ) :
    (forwardIO h p scale).kernel=(Kernels.source .forward h p scale).eraseDType := rfl

def prepareIO (h : Call) (p : Phase) (scale : ℝ) :=
  { legacyPrepareIO h p scale with
    kernel := (Kernels.source .prepare h p scale).eraseDType
    projection := by
      simp only [ComputeKernel.eraseDType,Kernels.source_projects,
        ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody] }

theorem prepare_source (h : Call) (p : Phase) (scale : ℝ) :
    (prepareIO h p scale).kernel=(Kernels.source .prepare h p scale).eraseDType := rfl

def backwardIO (h : Call) (p : Phase) (scale : ℝ) :=
  { legacyBackwardIO h p scale with
    kernel := (Kernels.source .backward h p scale).eraseDType
    projection := by
      simp only [ComputeKernel.eraseDType,Kernels.source_projects,
        ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody] }

theorem backward_source (h : Call) (p : Phase) (scale : ℝ) :
    (backwardIO h p scale).kernel=(Kernels.source .backward h p scale).eraseDType := rfl

theorem forward_complete (h : Call) (p : Phase) (scale : ℝ) (positive : 0<h.DK) :
    StreamEmitMasked3DKernelIO₁.Implements (forwardIO h p scale) (fun _ _ _ xs t j => ForwardCorrect.scan p.span h.BK xs t j) := by
  intro A hd hr hc p₀ p₁ p₂ xs u hp₀ hp₁ hp₂ hu hread hwrite hin
  obtain ⟨v,he,hv,hf⟩ := legacy_forward_complete h p scale positive
    A hd hr hc p₀ p₁ p₂ xs u hp₀ hp₁ hp₂ hu hread hwrite hin
  change exec (A.flattenKernel (LegacySurface.source .forward h p scale).eraseDType.toAlgKernel)
    (A.flattenState u)=some v at he
  have same := RealSourceBridge.forward_flat A h p scale (A.flattenState u)
  rw [he] at same
  cases ran : exec (A.flattenKernel (Kernels.source .forward h p scale).eraseDType.toAlgKernel) (A.flattenState u) with
  | none => simp [ran] at same
  | some w =>
    have states : RealBodyBridge.norm v=RealBodyBridge.norm w := by simpa [ran] using same
    have memory : v.mem=w.mem := by simpa only [RealBodyBridge.norm] using congrArg BlockState.mem states
    refine ⟨w,ran,?_,?_⟩
    · intro t j active
      simpa only [BlockState.readMem,memory] using hv t j active
    · intro r o outside
      simpa only [memory] using hf r o outside

theorem prepare_complete (h : Call) (p : Phase) (hp : p∈phases h) (scale : ℝ) (positive : 0<h.DK) :
    StreamGroupedEmitMasked3DKernelIO.Implements (prepareIO h p scale) (fun _ _ _ xs o t j => PrepareCorrect.values p.span h.BK scale xs o t j) := by
  intro A hd hr hc p₀ p₁ p₂ xs u hp₀ hp₁ hp₂ hu hread hwrite hin
  obtain ⟨v,he,hv,hf⟩ := legacy_prepare_complete h p hp scale positive
    A hd hr hc p₀ p₁ p₂ xs u hp₀ hp₁ hp₂ hu hread hwrite hin
  change exec (A.flattenKernel (LegacySurface.source .prepare h p scale).eraseDType.toAlgKernel)
    (A.flattenState u)=some v at he
  have same := RealSourceBridge.flat .prepare A h p scale (Geometry.phase_positive h p hp).2.1 (A.flattenState u)
  rw [he] at same
  cases ran : exec (A.flattenKernel (Kernels.source .prepare h p scale).eraseDType.toAlgKernel) (A.flattenState u) with
  | none => simp [ran] at same
  | some w =>
    have states : RealBodyBridge.norm v=RealBodyBridge.norm w := by simpa [ran] using same
    have memory : v.mem=w.mem := by simpa only [RealBodyBridge.norm] using congrArg BlockState.mem states
    refine ⟨w,ran,?_,?_⟩
    · intro o t j active
      simpa only [BlockState.readMem,memory] using hv o t j active
    · intro r o outside
      simpa only [memory] using hf r o outside

theorem backward_complete (h : Call) (p : Phase) (hp : p∈phases h) (scale : ℝ) (positive : 0<h.DK) :
    StreamGroupedEmitMasked3DKernelIO.Implements (backwardIO h p scale) (fun _ _ _ xs o t j => BackwardCorrect.values p.span h.BK xs o t j) := by
  intro A hd hr hc p₀ p₁ p₂ xs u hp₀ hp₁ hp₂ hu hread hwrite hin
  obtain ⟨v,he,hv,hf⟩ := legacy_backward_complete h p hp scale positive
    A hd hr hc p₀ p₁ p₂ xs u hp₀ hp₁ hp₂ hu hread hwrite hin
  change exec (A.flattenKernel (LegacySurface.source .backward h p scale).eraseDType.toAlgKernel)
    (A.flattenState u)=some v at he
  have same := RealSourceBridge.flat .backward A h p scale (Geometry.phase_positive h p hp).2.1 (A.flattenState u)
  rw [he] at same
  cases ran : exec (A.flattenKernel (Kernels.source .backward h p scale).eraseDType.toAlgKernel) (A.flattenState u) with
  | none => simp [ran] at same
  | some w =>
    have states : RealBodyBridge.norm v=RealBodyBridge.norm w := by simpa [ran] using same
    have memory : v.mem=w.mem := by simpa only [RealBodyBridge.norm] using congrArg BlockState.mem states
    refine ⟨w,ran,?_,?_⟩
    · intro o t j active
      simpa only [BlockState.readMem,memory] using hv o t j active
    · intro r o outside
      simpa only [memory] using hf r o outside

#axiomsClean forward_complete
#axiomsClean prepare_complete
#axiomsClean backward_complete
end
end VeriTile.Bench.Repairs.DecayCumsum.RealTiles
