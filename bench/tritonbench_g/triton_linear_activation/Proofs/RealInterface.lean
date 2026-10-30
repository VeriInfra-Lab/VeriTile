import bench.tritonbench_g.triton_linear_activation.Proofs.IOBindings
import bench.tritonbench_g.triton_linear_activation.Proofs.KernelReal

namespace VeriTile.Bench.Repairs.TritonLinearActivation.LinearIO
open VeriTile.Triton FP.Structural KernelSource KernelValues KernelContract KernelRealArithmetic
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

theorem densePre_eq (mode : Mode) (p : Parameters) (pid : Nat) (values : Inputs ℝ mode p)
    (s : State ℝ) (hp : s.pids 0 = pid) (h : Bound id mode p pid values s)
    (i : TileIndex [p.blockRows,p.blockColumns]) (hi : active p pid i) :
    densePre mode p s (KernelValues.row p s i.1) (KernelValues.col p s i.2.1) =
      preactivation mode p values i := by
  simp only [densePre,preactivation,bias,row_eq p pid s hp,col_eq p pid s hp]
  congr 1
  · cases hb : p.hasBias
    · simp [hb]
    · simp only [hb,if_true]
      simpa [Nat.mod_eq_of_lt hi.2] using h.bias hb i.2.1 (by simpa [Nat.mod_eq_of_lt hi.2] using hi.2)
  · rw [← Fin.sum_univ_eq_sum_range]
    apply Finset.sum_congr rfl
    intro k _
    congr 1
    · simpa [x,Nat.mod_eq_of_lt hi.1] using h.left i.1 k
    · simpa [w,Nat.mod_eq_of_lt hi.2] using h.right i.2.1 k

/-- Assemble existing complete source execution and tensor values with the
shared typed IO judgment. Access safety is proved from the declared windows. -/
theorem real_implements (mode : Mode) (p : Parameters) (pid : Nat)
    (positive : 0 < p.blockInner) (strides : CanonicalStrides p) (separate : p.saved ≠ p.output) :
    (realIO mode p pid).Implements (realTarget mode p) := by
  apply ProgramIO.Implements.intro
  · intro values bounds hb s _
    exact schedule_safe algebra dot id id mode p pid values bounds hb s positive
  · intro values s hin
    let t := (invocation algebra dot mode p pid).initial s
    have hp : t.pids 0 = pid := launch_pid algebra dot mode p pid s
    have h := bound algebra dot id id mode p pid values s hin
    obtain ⟨final,ran,out,saved,exterior⟩ := KernelReal.outputs mode p t positive strides separate
    refine ⟨final,?_,?_,frame algebra dot id id mode p pid values s final exterior⟩
    · simpa only [realIO,programIO,runSchedule_singleton,Invocation.run] using ran
    · intro o i hi
      fin_cases o
      · have enabled := (active_eq p pid t hp i).mpr hi
        have result := out i enabled
        have target := densePre_eq mode p pid values t hp h i hi
        simp only [denseOutput,target] at result
        change mapCell id (final.mem p.output _) = _
        have offset : rowIndex p pid i.1*p.outputRowStride+colIndex p pid i.2.1*p.outputColStride =
            KernelValues.outputOffset p t i := by simp [KernelValues.outputOffset,row_eq p pid t hp,col_eq p pid t hp]
        dsimp only [realIO,nativeIO,programIO,memory,outputs,realTarget,nativeTarget,List.length_cons,List.length_nil,Nat.reduceAdd]
        rw [offset,result]
        cases mode <;> rfl
      · have enabled := (active_eq p pid t hp i).mpr hi.2
        have result := saved hi.1 i enabled
        rw [densePre_eq mode p pid values t hp h i hi.2] at result
        change mapCell id (final.mem p.saved _) = _
        have offset : rowIndex p pid i.1*p.savedRowStride+colIndex p pid i.2.1*p.savedColStride =
            savedOffset p t i := by simp [savedOffset,row_eq p pid t hp,col_eq p pid t hp]
        dsimp only [realIO,nativeIO,programIO,memory,outputs,realTarget,nativeTarget,List.length_cons,List.length_nil,Nat.reduceAdd]
        rw [offset,result]
        cases mode <;> rfl

#axiomsClean real_implements
end
end VeriTile.Bench.Repairs.TritonLinearActivation.LinearIO
