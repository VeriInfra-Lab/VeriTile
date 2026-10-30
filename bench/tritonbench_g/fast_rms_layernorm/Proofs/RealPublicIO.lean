import bench.tritonbench_g.fast_rms_layernorm.Proofs.IOBinding
import bench.tritonbench_g.fast_rms_layernorm.Proofs.IOSafety

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealPublicIO
open VeriTile Triton KernelTensor KernelStorage
open IOExecution IOBinding
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem frame (h : Call) (backward : Bool) (vals : (RealIO.program h backward).Ctx)
    (initial final : BlockState)
    (old : KernelTensor.Frame h (if backward then .backward else .forward) (realHeap initial) (realHeap final)) :
    (RealIO.program h backward).Frame initial final vals := by
  intro r off outside scratch
  apply old r off
  rintro ⟨member,bound⟩
  dsimp only [RealIO.program,RealIO.memory] at outside
  cases backward
  · simp only [Bool.false_eq_true,if_false,KernelStorage.outputs,List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl
    · have got:=outside 0 ⟨off,by simpa [cells,Ports.outputSize] using bound⟩ trivial
      exact got.elim (fun ne=>ne rfl) (fun ne=>ne rfl)
    · have got:=outside 1 ⟨off,by simpa [cells,Ports.outputSize] using bound⟩ trivial
      exact got.elim (fun ne=>ne rfl) (fun ne=>ne rfl)
  · simp only [if_true,KernelStorage.outputs,List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl
    · have got:=outside 0 ⟨off,by simpa [cells,Ports.outputSize] using bound⟩ trivial
      exact got.elim (fun ne=>ne rfl) (fun ne=>ne rfl)
    · have got:=outside 1 ⟨off,by simpa [cells,Ports.outputSize] using bound⟩ trivial
      exact got.elim (fun ne=>ne rfl) (fun ne=>ne rfl)

theorem forward (h : Call) :
    (RealIO.program h Bool.false).Implements (Reference.real h Bool.false) := by
  intro vals _ initial binding
  refine ⟨fun bounds hb=>IOSafety.real h Bool.false bounds (IOBounds.real h Bool.false vals initial bounds hb) initial,?_⟩
  obtain ⟨heap,ran,y,r,oldFrame⟩ := RealTensor.forward_raw h (realHeap initial)
    (Ports.logical h vals) (real_reads h Bool.false vals initial binding)
  have mapped:=(real_forward h initial).trans ran
  obtain ⟨final,actual,hm⟩ := Option.map_eq_some_iff.mp mapped
  subst heap
  refine ⟨final,actual,?_,frame h Bool.false vals initial final oldFrame⟩
  intro o j active
  fin_cases o
  · let i : Fin h.rows:=⟨j.val/h.cols,(Nat.div_lt_iff_lt_mul h.positive).mpr j.isLt⟩
    let k : Fin h.cols:=⟨j.val%h.cols,Nat.mod_lt _ h.positive⟩
    have got:=y i k
    have index : i.val*h.cols+k.val=j.val:=by simpa [i,k,Nat.mul_comm] using Nat.div_add_mod j.val h.cols
    rw [index,row_eq,weight_eq] at got
    exact got
  · have got:=r j
    rw [row_eq] at got
    exact got

theorem backward (h : Call) :
    (RealIO.program h Bool.true).ImplementsOn (Reference.realDomain h Bool.true) (Reference.real h Bool.true) := by
  intro vals domain initial binding
  refine ⟨fun bounds hb=>IOSafety.real h Bool.true bounds (IOBounds.real h Bool.true vals initial bounds hb) initial,?_⟩
  by_cases empty : h.rows=0
  · refine ⟨initial,by simp [RealIO.program,RealIO.schedule,empty,runPreparedAlgorithmSchedule],?_,?_⟩
    · intro o j active
      fin_cases o
      · have bad:=j.isLt
        simp [RealIO.program,RealIO.memory,Ports.outputSize,Call.cells,empty] at bad
      · change Fin h.cols at j
        dsimp only [PreparedAlgorithmProgramIO.InputBinding,AlgorithmMemoryIO.InputBinding,RealIO.program,RealIO.memory] at binding
        have got:=binding 4 j
          (Or.inr ⟨rfl,Or.inr empty⟩)
        have zero:=(domain rfl).2 empty j
        have formula : RealReference.weightGradient h.eps (Reference.row h vals) (Reference.upstream h vals) j=0 := by
          apply Finset.sum_eq_zero
          intro i _
          have bad:=i.isLt
          omega
        change initial.readMem "dw" j.val=RealReference.weightGradient h.eps
          (Reference.row h vals) (Reference.upstream h vals) j
        rw [formula]
        exact got.trans zero
    · intro r off _ _;rfl
  · have saved (i : Fin h.rows) : Ports.logical h vals "r" i.val=
        RealReference.inv h.eps (rowValues h (Ports.logical h vals) "x" i) := by
      rw [saved_eq,row_eq]
      exact (domain rfl).1 i
    obtain ⟨heap,ran,dx,dw,oldFrame⟩ := RealTensor.backward_raw h (realHeap initial)
      (Ports.logical h vals) (real_reads h Bool.true vals initial binding) saved
    have mapped:=(real_backward h empty initial).trans ran
    obtain ⟨final,actual,hm⟩ := Option.map_eq_some_iff.mp mapped
    subst heap
    refine ⟨final,actual,?_,frame h Bool.true vals initial final oldFrame⟩
    intro o j active
    fin_cases o
    · let i : Fin h.rows:=⟨j.val/h.cols,(Nat.div_lt_iff_lt_mul h.positive).mpr j.isLt⟩
      let k : Fin h.cols:=⟨j.val%h.cols,Nat.mod_lt _ h.positive⟩
      have got:=dx i k
      have index : i.val*h.cols+k.val=j.val:=by simpa [i,k,Nat.mul_comm] using Nat.div_add_mod j.val h.cols
      rw [index,row_eq,weight_eq,upstream_eq] at got
      exact got
    · have got:=dw j
      rw [row_eq,upstream_eq] at got
      exact got

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealPublicIO
