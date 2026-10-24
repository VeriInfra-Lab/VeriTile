import bench.tritonbench_g.reversed_cumsum.Proofs.RealExecution

/-! Complete-source regressions from empty registers. The nonempty case has
two 16-row chunks, a partial time tail, 32 feature lanes with only the first
active, nonzero launch IDs, and a nonzero unspecified-value oracle. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.RealRegression
open VeriTile.Triton RealProgram RealStep RealLoop
noncomputable section
set_option maxHeartbeats 1800000

def layout : Layout := ⟨561,33,1,17,33,16,32⟩

def initial : BlockState where
  mem := fun region _ => if region = "s" then .real 2 else MemCell.of .bool Bool.true
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun _ _ => 7

theorem no_collision : NoCollision layout (RealInit.launch initial) := by
  intro t u i j ht hu he
  simp only [active, layout, RealInit.launch, initial] at ht hu
  have hi : i.val = 0 := by omega
  have hj : j.val = 0 := by omega
  simp only [address, layout, RealInit.launch, initial, hi, hj] at he
  exact ⟨by omega, Fin.ext (hi.trans hj.symm)⟩

theorem expected_first : expected layout (RealInit.launch initial)
    (initial.readMem "s") (initial.undef "s") 0 ⟨0,by decide⟩ = 139 := by
  have hread : initial.readMem "s" = fun _ => 2 := by funext offset; rfl
  rw [hread]
  norm_num [expected, RealMath.suffixValue, RealMath.initialSum, numChunks,
    value, active, address, layout, RealInit.launch, initial, Finset.sum_range_succ]

theorem expected_last : expected layout (RealInit.launch initial)
    (initial.readMem "s") (initial.undef "s") 16 ⟨0,by decide⟩ = 107 := by
  have hread : initial.readMem "s" = fun _ => 2 := by funext offset; rfl
  rw [hread]
  norm_num [expected, RealMath.suffixValue, RealMath.initialSum, numChunks,
    value, active, address, layout, RealInit.launch, initial, Finset.sum_range_succ]

/-- The source's actual unspecified tail affects all earlier valid outputs.
The same successful specification is established independently for either
comparison spelling. All neighboring boolean cells retain their full tag. -/
theorem nonzero_tail_run (swapped : Bool) :
    ∃ final, exec (RealSource.source swapped .bf16 .fp16 "s" "o"
      561 33 1 17 33 16 32).eraseDType.toAlgKernel initial = some final ∧
      final.mem "o" 593 = MemCell.real 139 ∧
      final.mem "o" 1121 = MemCell.real 107 ∧
      final.mem "o" 594 = MemCell.of .bool Bool.true := by
  obtain ⟨final, he, hp⟩ := RealExecution.source_correct swapped layout "s" "o"
    .bf16 .fp16 initial (by decide) (by decide) no_collision
  refine ⟨final, he, ?_, ?_, ?_⟩
  · simpa only [expected_first] using hp.1 0 ⟨0,by decide⟩ (by decide)
  · simpa only [expected_last] using hp.1 16 ⟨0,by decide⟩ (by decide)
  · have hn : ¬ footprint layout (RealInit.launch initial) "o" ("o",594) := by
      rintro ⟨_,t,j,ha,heq⟩
      simp only [active, layout, RealInit.launch, initial] at ha
      have hj : j.val = 0 := by omega
      simp only [address, layout, RealInit.launch, initial, hj] at heq
      omega
    exact (hp.2 "o" 594 hn).symm

theorem empty_time (swapped : Bool) (p : Layout) (s o : RegionName)
    (dtype out_dtype : FloatDType) (u : BlockState)
    (hSO : s ≠ o) (hBT : 0 < p.BT) (hT : p.T = 0) :
    ∃ v, exec (RealSource.source swapped dtype out_dtype s o p.headStride
      p.timeStride p.channelStride p.T p.S p.BT p.BS).eraseDType.toAlgKernel u = some v ∧
      v.mem = u.mem := by
  have hinj : NoCollision p (RealInit.launch u) := by
    intro t x i j ht
    have := ht.1
    omega
  obtain ⟨v,he,hp⟩ := RealExecution.source_correct swapped p s o dtype out_dtype u hSO hBT hinj
  refine ⟨v,he,?_⟩
  funext r off
  apply (hp.2 r off ?_).symm
  rintro ⟨_,t,j,ha,_⟩
  have := ha.1
  omega

#axiomsClean nonzero_tail_run
#axiomsClean empty_time
end
end VeriTile.Bench.TritonBenchG.ReversedCumsum.RealRegression
