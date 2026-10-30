import bench.tritonbench_g.triton_argmax.IO
import bench.tritonbench_g.triton_argmax.Proofs.JITRealDimensionSafety
import bench.tritonbench_g.triton_argmax.Proofs.JITRealDimension

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITRealDimensionIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section
set_option maxHeartbeats 5000000

/-- Logical coordinates are independent of the source registers and memory. -/
def input (M N K : Nat) (xs : ChannelValues ℝ [⟨.real,[M,N,K],X⟩]) (r i c : Nat) : ℝ :=
  if hr : r < M then if hi : i < K then if hc : c < N then
    xs 0 (⟨r,hr⟩,⟨c,hc⟩,⟨i,hi⟩,PUnit.unit) else 0 else 0 else 0

def mathematical (rest : Algebra Math.Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base pid : Nat) (nonempty : 0 < N) (index64 : Bool)
    (xs : ChannelValues ℝ [⟨.real,[M,N,K],X⟩]) :
    ChannelValues ℝ [⟨.int,[BM],Region.cast O⟩] :=
  Fin.cases (fun j => Int.ofNat (Math.argmax N nonempty (fun col : Fin N =>
    input M N K xs (((pid+base)/K)*BM+j.1.val) ((pid+base)%K) col.val)).val) (fun i => Fin.elim0 i)

theorem implements (rest : Algebra Math.Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base pid : Nat) (index64 : Bool) (positive : 0 < BN)
    (nonempty : 0 < N) (innerPositive : 0 < K) :
    (JITIO.realDimension rest X O M N K BM BN base pid index64).Implements
      (mathematical rest X O M N K BM BN base pid nonempty index64) := by
  intro vals _ bounds bounded s binding
  let initial := (JITIO.atProgram (TypedReal.backend rest)
    (JITSource.dimension X O BM BN index64) pid).initial s
  have initialPid : initial.pids 0 = pid := by simp [initial]
  have initialMem : initial.mem = s.mem := Invocation.initial_mem _ _
  have reads : ∀ row col inner, row < M → col < N → inner < K →
      (initial.mem X (row*N*K+col*K+inner)).read .real =
        (input M N K vals row inner col : Math.Value) := by
    intro row col inner hr hc hi
    have h := binding 0 (⟨row,hr⟩,⟨col,hc⟩,⟨inner,hi⟩,PUnit.unit) trivial
    simpa [JITIO.realDimension, mapValue, input, hr, hc, hi, initialMem] using h
  constructor
  · rw [show (JITIO.realDimension rest X O M N K BM BN base pid index64).schedule vals =
      [JITIO.dimensionLaunch (TypedReal.backend rest) (JITSource.dimension X O BM BN index64) M N K base pid] from rfl,
      scheduleSafe_singleton]
    change TraceSafe (TypedReal.backend rest) bounds (JITSource.dimension X O BM BN index64) (JITRealDimension.arguments initial M N K base)
    apply JITRealDimensionSafety.source_safe rest initial X O M N K BM BN base index64 positive bounds
    · intro r col hr hc
      simpa [initialPid] using bounded.2.1 0
        (⟨r,hr⟩,⟨col,hc⟩,⟨(pid+base)%K,Nat.mod_lt _ innerPositive⟩,PUnit.unit) trivial
    · intro j hj
      simpa [initialPid] using bounded.2.2.1 0 (j,PUnit.unit) (by simpa [initialPid] using hj)
  · obtain ⟨final, ran, output, frame⟩ := JITRealDimension.source_reference rest initial X O
      M N K BM BN base index64 positive nonempty innerPositive (input M N K vals) reads
    refine ⟨final, ?_, ?_, ?_⟩
    · simpa only [JITIO.realDimension, runSchedule_singleton, Invocation.run] using ran
    · intro o j active
      have ho : o = 0 := Fin.eq_zero o
      subst o
      change mapCell (fun _ => (0 : ℝ))
        (final.mem (Region.cast O) ((((pid+base)/K)*BM+j.1.val)*K+(pid+base)%K)) = _
      have result := output j.1 (by simpa [initialPid] using active)
      simp only [initialPid] at result
      rw [result]
      rfl
    · intro r offset outside
      rw [frame, initialMem]
      by_cases region : r = Region.cast O
      · right
        intro j active
        have miss := outside.1 0 (j,PUnit.unit) (by simpa [initialPid] using active)
        change r ≠ Region.cast O ∨ offset ≠ (((pid+base)/K)*BM+j.val)*K+(pid+base)%K at miss
        simpa only [initialPid] using miss.resolve_left (not_not.mpr region)
      · exact Or.inl region

#axiomsClean implements
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITRealDimensionIO
