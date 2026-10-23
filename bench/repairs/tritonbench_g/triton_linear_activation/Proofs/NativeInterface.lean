import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.IOBindings
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelMemory
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelContract

namespace VeriTile.Bench.Repairs.TritonLinearActivation.LinearIO
open VeriTile.Triton FP.Structural KernelSource KernelValues KernelContract KernelNativeModel
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem native_left (mode : Mode) (p : Parameters) (pid : Nat) (values : Inputs Word mode p)
    (s : State Term) (hp : s.pids 0 = pid) (h : Bound Term.raw mode p pid values s)
    (positive : 0 < p.blockInner) (cur : Nat) (inside : cur < ceilDiv p.inner p.blockInner)
    (i : Fin p.blockRows) (k : Fin p.blockInner) :
    aValues algebra mode p s cur (i,k,PUnit.unit) = nativeLeft mode p values i (cur*p.blockInner+k.val) := by
  unfold aValues nativeLeft
  simp only [KernelRealArithmetic.mask_equiv p cur k.val positive inside k.isLt]
  split_ifs with hk
  · simpa [aOffset,row_eq p pid s hp,Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
      using h.left i ⟨cur*p.blockInner+k.val,hk⟩
  · rfl

theorem native_right (mode : Mode) (p : Parameters) (pid : Nat) (values : Inputs Word mode p)
    (s : State Term) (hp : s.pids 0 = pid) (h : Bound Term.raw mode p pid values s)
    (positive : 0 < p.blockInner) (cur : Nat) (inside : cur < ceilDiv p.inner p.blockInner)
    (j : Fin p.blockColumns) (k : Fin p.blockInner) :
    bValues algebra mode p s cur (k,j,PUnit.unit) = nativeRight mode p values j (cur*p.blockInner+k.val) := by
  unfold bValues nativeRight
  simp only [KernelRealArithmetic.mask_equiv p cur k.val positive inside k.isLt]
  split_ifs with hk
  · simpa [bOffset,col_eq p pid s hp,Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
      using h.right j ⟨cur*p.blockInner+k.val,hk⟩
  · rfl

/-- Existing source-order recurrence is connected to the independent logical
operand recurrence, without introducing native/Real equalities. -/
theorem native_accumulated (c : Fin 98) (mode : Mode) (p : Parameters) (pid : Nat)
    (values : Inputs Word mode p) (s : State Term) (hp : s.pids 0 = pid)
    (h : Bound Term.raw mode p pid values s) (positive : 0 < p.blockInner)
    (i : TileIndex [p.blockRows,p.blockColumns]) :
    accumulated algebra (dot c mode) mode p s i = nativePre c mode p pid values i := by
  unfold accumulated nativePre
  generalize hn : ceilDiv p.inner p.blockInner = count
  have bound : count ≤ ceilDiv p.inner p.blockInner := by omega
  clear hn
  induction count with
  | zero =>
    simp only [foldN,initial,biasValue,col_eq p pid s hp,algebra]
    cases hb : p.hasBias
    · simp [hb,zero,algebra]
    · simp only [hb,if_true]
      by_cases valid : colIndex p pid i.2.1 % p.columns < p.columns
      · rw [if_pos valid,if_pos valid,h.bias hb i.2.1 valid]
      · rw [if_neg valid,if_neg valid]
  | succ count ih =>
    simp only [foldN,update,algebra,dot,KernelRealArithmetic.laneDot]
    have previous := ih (by omega)
    simp only [algebra,dot] at previous
    rw [previous]
    congr 3
    · funext k
      exact native_left mode p pid values s hp h positive count (by omega) i.1 k
    · funext k
      exact native_right mode p pid values s hp h positive count (by omega) i.2.1 k

/-- Exact source/compiler-model native observations and complete frame for
all input words, both storage dtypes, and every existing policy/configuration. -/
theorem native_implements (profile : LocalProfile) (policy : CompilerPolicy) (c : Fin 98)
    (mode : Mode) (p : Parameters) (pid : Nat)
    (positive : 0 < p.blockInner) (strides : CanonicalStrides p) (separate : p.saved ≠ p.output) :
    (nativeIO profile policy c mode p pid).Implements (nativeTarget profile policy c mode p pid) := by
  apply ProgramIO.Implements.intro
  · intro values bounds hb s _
    exact schedule_safe algebra (dot c mode) Term.raw (evaluate profile policy) mode p pid values bounds hb s positive
  · intro values s hin
    let t := (invocation algebra (dot c mode) mode p pid).initial s
    have hp : t.pids 0 = pid := launch_pid algebra (dot c mode) mode p pid s
    have h := bound algebra (dot c mode) Term.raw (evaluate profile policy) mode p pid values s hin
    obtain ⟨final,ran,out,saved,exterior⟩ := KernelMemory.outputs algebra (dot c mode) mode p t positive separate
      (output_unique p t strides) (saved_unique p t strides)
    refine ⟨final,?_,?_,frame algebra (dot c mode) Term.raw (evaluate profile policy) mode p pid values s final exterior⟩
    · simpa only [nativeIO,programIO,runSchedule_singleton,Invocation.run] using ran
    · intro o i hi
      fin_cases o
      · have result := out i ((active_eq p pid t hp i).mpr hi)
        rw [native_accumulated c mode p pid values t hp h positive i] at result
        change FP.Structural.mapCell (evaluate profile policy) (final.mem p.output _) = _
        have offset : rowIndex p pid i.1*p.outputRowStride+colIndex p pid i.2.1*p.outputColStride =
            KernelValues.outputOffset p t i := by simp [KernelValues.outputOffset,row_eq p pid t hp,col_eq p pid t hp]
        dsimp only [realIO,nativeIO,programIO,memory,outputs,realTarget,nativeTarget,List.length_cons,List.length_nil,Nat.reduceAdd]
        rw [offset,result]
        cases mode <;> rfl
      · have result := saved hi.1 i ((active_eq p pid t hp i).mpr hi.2)
        rw [native_accumulated c mode p pid values t hp h positive i] at result
        change FP.Structural.mapCell (evaluate profile policy) (final.mem p.saved _) = _
        have offset : rowIndex p pid i.1*p.savedRowStride+colIndex p pid i.2.1*p.savedColStride =
            savedOffset p t i := by simp [savedOffset,row_eq p pid t hp,col_eq p pid t hp]
        dsimp only [realIO,nativeIO,programIO,memory,outputs,realTarget,nativeTarget,List.length_cons,List.length_nil,Nat.reduceAdd]
        rw [offset,result]
        cases mode <;> rfl

#axiomsClean native_implements
end
end VeriTile.Bench.Repairs.TritonLinearActivation.LinearIO
