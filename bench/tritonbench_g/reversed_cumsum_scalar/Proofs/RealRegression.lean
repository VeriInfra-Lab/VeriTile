import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.RealExecution

/-! Concrete complete-source regressions start with empty registers and a
nonzero launch row. Two chunks and a nonzero unspecified tail expose the
source's padding contribution; neighboring cells retain their boolean tag. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealRegression
open VeriTile.Triton RealProgram RealValues RealLoop
noncomputable section
set_option maxHeartbeats 1800000

def initial : BlockState where
  mem := fun region _ => if region = "s" then .real 2 else MemCell.of .bool Bool.true
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun _ _ => 7

theorem expected_first : expected 17 16 1 (initial.readMem "s") (initial.undef "s") 0 = 139 := by
  have hread : initial.readMem "s" = fun _ => 2 := by funext offset; rfl
  rw [hread]
  norm_num [expected, Formula.suffix, Formula.initialSum, numChunks,
    value, address, initial, Finset.sum_range_succ]

theorem expected_last : expected 17 16 1 (initial.readMem "s") (initial.undef "s") 16 = 107 := by
  have hread : initial.readMem "s" = fun _ => 2 := by funext offset; rfl
  rw [hread]
  norm_num [expected, Formula.suffix, Formula.initialSum, numChunks,
    value, address, initial, Finset.sum_range_succ]

theorem nonzero_tail_run (swapped : Bool) :
    ∃ final, exec (TypedSource.source swapped .bf16 .fp16 "s" "o" 17 16).eraseDType.toAlgKernel
        initial = some final ∧
      final.mem "o" 17 = MemCell.real 139 ∧
      final.mem "o" 33 = MemCell.real 107 ∧
      final.mem "o" 34 = MemCell.of .bool Bool.true := by
  obtain ⟨final,he,hp⟩ := RealExecution.source_correct swapped 17 16 "s" "o" .bf16 .fp16
    initial (by decide) (by decide)
  refine ⟨final,he,?_,?_,?_⟩
  · have h := hp.1 0 (by decide)
    change final.mem "o" 17 = MemCell.real
      (expected 17 16 1 (initial.readMem "s") (initial.undef "s") 0) at h
    rwa [expected_first] at h
  · have h := hp.1 16 (by decide)
    change final.mem "o" 33 = MemCell.real
      (expected 17 16 1 (initial.readMem "s") (initial.undef "s") 16) at h
    rwa [expected_last] at h
  · have hn : ¬ RealMemory.footprint 17 1 "o" ("o",34) := by
      rintro ⟨_,t,ht,heq⟩
      simp only [address] at heq
      omega
    exact (hp.2 "o" 34 hn).symm

theorem empty_time (swapped : Bool) (BT : Nat) (s o : RegionName)
    (dtype out_dtype : FloatDType) (u : BlockState) (hSO : s ≠ o) (hBT : 0 < BT) :
    ∃ v, exec (TypedSource.source swapped dtype out_dtype s o 0 BT).eraseDType.toAlgKernel u =
      some v ∧ v.mem = u.mem := by
  obtain ⟨v,he,hp⟩ := RealExecution.source_correct swapped 0 BT s o dtype out_dtype u hSO hBT
  refine ⟨v,he,?_⟩
  funext r off
  apply (hp.2 r off ?_).symm
  rintro ⟨_,t,ht,_⟩
  omega

#axiomsClean nonzero_tail_run
#axiomsClean empty_time
end
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealRegression
