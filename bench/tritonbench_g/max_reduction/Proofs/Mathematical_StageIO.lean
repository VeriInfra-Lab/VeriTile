import bench.tritonbench_g.max_reduction.Proofs.Mathematical_Real
import bench.tritonbench_g.max_reduction.Proofs.Mathematical_Stage2Real
import bench.tritonbench_g.max_reduction.Proofs.Mathematical_Candidate

/-! Shared exact stage proofs. Public readers apply these results; required
proof modules can reuse them without importing a public Correct reader. -/
namespace VeriTile.Bench.TritonBenchG.MaxReduction
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedTileKernelIO₁

/-- **The headline on the IO surface** for `max_reduction.py`'s `max_kernel_1`:
for every disjoint flat placement of the two buffers, every program id whose
active lanes are in bounds, and every launch state whose input window holds `xs`
at the active lanes, the translated pointer kernel terminates, `mid[pid]` holds
the genuine block max — `Tile.reduceMax` of `xs` over the active lanes, with the
`other=-float("inf")` padding modeled as `⊥` — and every other memory cell is
unchanged.

This is the first *reduction* on an `io ⊨ f` face, and it is what the spec `f`'s
program-id argument is for: the reduced index set is `{i | pid·BLOCK_SIZE + i < M}`,
so the value is irreducibly pid-dependent (the tail block reduces fewer lanes
than a full block) and a pid-independent spec would be falsifiable.

Dimension-general in `M` and `BLOCK_SIZE`. Honest side-condition:
`0 < BLOCK_SIZE` — with an empty tile `tl.max` has no axis to reduce and the
kernel faults, so termination genuinely fails there. -/
theorem max_kernel_1_io_correctness
    (inp mid : RegionName) (M BLOCK_SIZE : Nat) (hB : 0 < BLOCK_SIZE) :
    maxKernel1IO inp mid M BLOCK_SIZE
      ⊨ fun pid xs _ => maxTileSpecOf M BLOCK_SIZE pid xs := by
  refine MaskedTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact max_kernel_1_flattenOk inp mid M BLOCK_SIZE
  · intro bounds s h1 h2
    exact max_kernel_1_traceSafe inp mid M BLOCK_SIZE hB bounds s
      (fun i hi => h1 (i, PUnit.unit) hi)
      (h2 (⟨0, hB⟩, PUnit.unit) rfl)
  · intro s₀ xs hin
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      max_kernel_1_region_run inp mid M BLOCK_SIZE hB s₀ xs
        (fun idx hact => hin idx hact)
    refine ⟨s1, hexec, hval, ?_⟩
    intro r o hcond
    refine hframe r o ?_
    rcases hcond with h | h
    · exact Or.inl h
    -- a non-degenerate tile always has a write-active lane, which turns the
    -- skin's lane-wise disjunct into the single-cell one
    · exact Or.inr (h (⟨0, hB⟩, PUnit.unit) rfl)


def typedStage1IO (storage : Storage) (v : Bool) (inp out : RegionName) (M B : Nat) :
    MaskedTileKernelIO₁ :=
  { maxKernel1IO inp out M B with
    kernel := (stage1 storage v inp out M B).eraseDType
    projection := stage1_supported storage v inp out M B }

def typedStage2IO (storage : Storage) (v : Bool) (inp out : RegionName) (M B : Nat) :
    MaskedTileKernelIO₁ :=
  { stage2IO inp out M B with
    kernel := (stage2 storage v inp out M B).eraseDType
    projection := stage2_supported storage v inp out M B }

namespace StageIOProof

/-- Full second-stage IO correctness against a finite mathematical maximum. -/
theorem stage2_math_correct (mid out : RegionName) (M B : Nat) (hB : 0 < B) :
    stage2IO mid out M B ⊨ fun _ xs _ => blockMaximum M B 0 xs := by
  refine MaskedTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact stage2_flattenOk mid out M B
  · intro bounds s hi ho
    exact stage2_traceSafe mid out M B hB bounds s
      (fun i h => hi (i, PUnit.unit) h) (ho (⟨0,hB⟩, PUnit.unit) rfl)
  · intro s xs hx
    obtain ⟨t, he⟩ := stage2_terminates mid out M B hB s
    refine ⟨t, he, ?_, ?_⟩
    · intro i hi
      change t.readMem out 0 = blockMaximum M B 0 xs
      rw [max_kernel_2_correct mid out M B s t he]
      exact stage2_formula mid M B hB s xs (fun i h => hx (i, PUnit.unit) h)
    · intro r o hf
      apply stage2_frame mid out M B hB s t he r o
      rcases hf with hr | ho
      · exact Or.inl hr
      · exact Or.inr (ho (⟨0,hB⟩, PUnit.unit) rfl)

/-- Each source version independently implements the finite maximum of the
masked original input window, terminates, and preserves every other flat cell. -/
theorem stage1_real_correct (storage : Storage) (v : Bool)
    (inp out : RegionName) (M B : Nat) (hB : 0 < B) :
    VeriTile.Spec.Real (typedStage1IO storage v inp out M B ⊨
      fun pid xs _ => blockMaximum M B pid xs) := by
  intro A hd hr hc pid hin hout xs s hp hu hx
  obtain ⟨t, he, hv, hf⟩ := max_kernel_1_io_correctness inp out M B hB
    A hd hr hc pid hin hout xs s hp hu hx
  refine ⟨t, ?_, ?_, hf⟩
  · exact (stage1_flat_exec A storage v inp out M B (A.flattenState s)).trans he
  · intro i hi
    change t.readMem A.flat (A.addr out pid) = blockMaximum M B pid xs
    rw [blockMaximum_eq_tile M B pid hB]
    exact hv i hi

/-- Each second-stage source independently computes the finite maximum of the
masked intermediate window, with successful execution and a full flat frame. -/
theorem stage2_real_correct (storage : Storage) (v : Bool)
    (inp out : RegionName) (M B : Nat) (hB : 0 < B) :
    VeriTile.Spec.Real (typedStage2IO storage v inp out M B ⊨
      fun _ xs _ => blockMaximum M B 0 xs) := by
  intro A hd hr hc pid hin hout xs s hp hu hx
  obtain ⟨t, he, hv, hf⟩ := stage2_math_correct inp out M B hB
    A hd hr hc pid hin hout xs s hp hu hx
  refine ⟨t, ?_, hv, hf⟩
  exact (stage2_flat_exec A storage v inp out M B (A.flattenState s)).trans he

#axiomsClean stage2_math_correct
#axiomsClean stage1_real_correct
#axiomsClean stage2_real_correct
end StageIOProof
end VeriTile.Bench.TritonBenchG.MaxReduction
