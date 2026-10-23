import bench.repairs.tritonbench_g.masked_add_cuda.IO
import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.AccessSafety
import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.Variant

namespace VeriTile.Bench.Repairs.MaskedAddCuda.NativeIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs (offset)
noncomputable section

private def leafWord : Graph.Expr → Graph.Word
  | .leaf w => w
  | _ => 0

theorem invocation_initial (rest : Algebra Graph.Expr) (alphaWord : Graph.Word)
    (storage : MaskStorage) (n pid : Nat) (token : ℝ) (flag : Bool) (s : State Graph.Expr) :
    (invocation rest alphaWord storage n pid token flag).initial s =
      {s with pids := fun ax => if ax = 0 then pid else 0} := rfl

theorem schedule_safe (rest : Algebra Graph.Expr) (alphaWord : Graph.Word)
    (storage : MaskStorage) (n pid : Nat) (token : ℝ) (flag : Bool)
    (values : Inputs) (bounds : RegionName → Nat)
    (hb : (originalIO rest alphaWord storage n pid token).InBounds bounds values)
    (s : State Graph.Expr) :
    ScheduleSafe bounds [invocation rest alphaWord storage n pid token flag] s := by
  rw [scheduleSafe_singleton]
  change TraceSafe (Graph.nativeAlgebra rest alphaWord) bounds
    (if flag then commutedKernel storage "grad" "data" "mask" n token
      else kernel storage "grad" "data" "mask" n token)
    ((invocation rest alphaWord storage n pid token flag).initial s)
  rw [invocation_initial]
  cases flag
  · apply AccessSafety.source_safe _ storage (parameters "grad" "data" "mask" n)
    · intro j hj
      exact hb.2.1 ⟨0, by simp [originalIO, inputs, outputs]⟩ (j, PUnit.unit) hj
    · intro j hj
      exact hb.2.1 ⟨1, by simp [originalIO, inputs, outputs]⟩ (j, PUnit.unit) hj
    · intro j hj
      exact hb.2.1 ⟨2, by simp [originalIO, inputs, outputs]⟩ (j, PUnit.unit) hj
  · apply AccessSafety.commuted_safe _ storage (parameters "grad" "data" "mask" n)
    · intro j hj
      exact hb.2.1 ⟨0, by simp [originalIO, inputs, outputs]⟩ (j, PUnit.unit) hj
    · intro j hj
      exact hb.2.1 ⟨1, by simp [originalIO, inputs, outputs]⟩ (j, PUnit.unit) hj
    · intro j hj
      exact hb.2.1 ⟨2, by simp [originalIO, inputs, outputs]⟩ (j, PUnit.unit) hj

/-- Shared IO assembly reuses the existing full source/candidate graph proof.
Only active grad leaves are required; no restriction is added to data graphs. -/
theorem equivalent (rest : Algebra Graph.Expr) (alphaWord : Graph.Word)
    (storage : MaskStorage) (n pid : Nat) (token : ℝ) :
    (originalIO rest alphaWord storage n pid token).EquivalentOn
      (candidate rest alphaWord storage n pid token) (LeafGradient n pid) := by
  apply ProgramIO.EquivalentOn.intro
  · intro values _ bounds hb initial _
    exact schedule_safe rest alphaWord storage n pid token Bool.false values bounds hb initial
  · intro values _ bounds hb initial _
    exact schedule_safe rest alphaWord storage n pid token Bool.true values bounds hb initial
  · intro values hd bounds hb initial hin
    let p := parameters "grad" "data" "mask" n
    let s : State Graph.Expr := {initial with pids := fun ax => if ax = 0 then pid else 0}
    let words : Nat → Graph.Word := fun a => leafWord ((s.mem "grad" a).read .real)
    have hg : ∀ i : TileIndex [blockSize], active p s i →
        (s.mem "grad" (offset p s i)).read .real = .leaf (words (offset p s i)) := by
      intro i hi
      have bound : pid * blockSize + i.1.val < n := hi.1
      have pin := hin ⟨0, by simp [originalIO, inputs, outputs]⟩ i bound
      have maskPin := hin ⟨2, by simp [originalIO, inputs, outputs]⟩ i bound
      have zero : veto values i.1 = 0 := by
        exact maskPin.symm.trans hi.2
      obtain ⟨w, hw⟩ := hd i.1 bound zero
      have cell : (s.mem "grad" (offset p s i)).read .real = .leaf w := pin.trans hw
      simp only [words, cell, leafWord]
    obtain ⟨left, right, leftRun, rightRun, same⟩ :=
      Variant.equivalent rest alphaWord storage p token s words hg
    have leftFrame : (originalIO rest alphaWord storage n pid token).Frame values initial left := by
      have memory : left.mem = expectedMemory (Graph.nativeAlgebra rest alphaWord) p token s := by
        have run := (MaskedAddCuda.execution (Graph.nativeAlgebra rest alphaWord) storage p token s).property
        have eql : left = (MaskedAddCuda.execution (Graph.nativeAlgebra rest alphaWord) storage p token s).val :=
          Option.some.inj (leftRun.symm.trans run)
        rw [eql, MaskedAddCuda.execution_memory]
      intro r address outside
      rw [memory]
      apply expected_frame
      by_cases hr : r = "grad"
      · right
        intro i hi
        have bound : pid * blockSize + i.1.val < n := hi.1
        have maskPin := hin ⟨2, by simp [originalIO, inputs, outputs]⟩ i bound
        have zero : veto values i.1 = 0 := maskPin.symm.trans hi.2
        exact (outside.1 ⟨0, by simp [originalIO, inputs, outputs]⟩ i ⟨bound, zero⟩).resolve_left (by simpa using hr)
      · exact Or.inl hr
    refine ⟨left, right, ?_, ?_, ?_, leftFrame, ?_⟩
    · change runSchedule [invocation rest alphaWord storage n pid token Bool.false] initial = some left
      rw [runSchedule_singleton]
      simpa only [Invocation.run, invocation_initial, invocation] using leftRun
    · change runSchedule [invocation rest alphaWord storage n pid token Bool.true] initial = some right
      rw [runSchedule_singleton]
      simpa only [Invocation.run, invocation_initial, invocation] using rightRun
    · intro o j active
      change mapCell id (left.mem _ _) = mapCell id (right.mem _ _)
      rw [same]
    · intro r address outside
      rw [← same]
      exact leftFrame r address outside

#axiomsClean equivalent
#axiomsClean schedule_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaskedAddCuda.NativeIO
