import bench.tritonbench_g.fast_rms_layernorm.Proofs.IOBinding
import bench.tritonbench_g.fast_rms_layernorm.Proofs.NativeIOSafety

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.NativePublicIO
open VeriTile Triton FP.Structural NativeGraph KernelTensor KernelStorage
open IOExecution IOBinding
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem frame (R : Reductions) (h : Call) (backward : Bool) (epsilon : Term)
    (vals : (NativeIO.program R h backward epsilon).Inputs) (initial final : State Term)
    (old : KernelTensor.Frame h (if backward then .backward else .forward) (nativeHeap initial) (nativeHeap final)) :
    (NativeIO.program R h backward epsilon).Frame vals initial final := by
  intro r off outside
  apply old r off
  rintro ⟨member,bound⟩
  cases backward
  · simp only [Bool.false_eq_true,if_false,KernelStorage.outputs,List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl
    · have got:=outside.1 ⟨0,by simp [NativeIO.program,Ports.outputs]⟩
        (⟨off,by simpa [cells] using bound⟩,()) trivial
      exact got.elim (fun ne=>ne rfl) (fun ne=>ne rfl)
    · have got:=outside.1 ⟨1,by simp [NativeIO.program,Ports.outputs]⟩
        (⟨off,by simpa [cells] using bound⟩,()) trivial
      exact got.elim (fun ne=>ne rfl) (fun ne=>ne rfl)
  · simp only [if_true,KernelStorage.outputs,List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl
    · have got:=outside.1 ⟨0,by simp [NativeIO.program,Ports.outputs]⟩
        (⟨off,by simpa [cells] using bound⟩,()) trivial
      exact got.elim (fun ne=>ne rfl) (fun ne=>ne rfl)
    · have got:=outside.1 ⟨1,by simp [NativeIO.program,Ports.outputs]⟩
        (⟨off,by simpa [cells] using bound⟩,()) trivial
      exact got.elim (fun ne=>ne rfl) (fun ne=>ne rfl)

theorem forward (R : Reductions) (h : Call) (epsilon : Term) :
    (NativeIO.program R h Bool.false epsilon).Implements (Reference.native R h Bool.false epsilon) := by
  intro vals _ bounds bounded initial binding
  refine ⟨NativeIOSafety.complete R h Bool.false epsilon bounds (IOBounds.native R h Bool.false epsilon vals bounds bounded) initial,?_⟩
  obtain ⟨heap,ran,y,r,oldFrame⟩ := RuntimeForward.Tensor.forward_raw (algebra R .forward) h epsilon
    (nativeHeap initial) (Ports.logical h (Ports.ofNative h vals)) (native_reads R h Bool.false epsilon vals initial binding)
  have mapped:=(native_forward R h epsilon initial).trans ran
  obtain ⟨final,actual,hm⟩ := Option.map_eq_some_iff.mp mapped
  subst heap
  refine ⟨final,actual,?_,frame R h Bool.false epsilon vals initial final oldFrame⟩
  intro o j active
  fin_cases o
  · let i : Fin h.rows:=⟨j.1.val/h.cols,(Nat.div_lt_iff_lt_mul h.positive).mpr j.1.isLt⟩
    let k : Fin h.cols:=⟨j.1.val%h.cols,Nat.mod_lt _ h.positive⟩
    have got:=y i k
    have index : i.val*h.cols+k.val=j.1.val:=by simpa [i,k,Nat.mul_comm] using Nat.div_add_mod j.1.val h.cols
    rw [index] at got
    change mapCell id (final.mem "y" j.1.val)=_
    rw [show final.mem "y" j.1.val=_ from got]
    rfl
  · have got:=r j.1
    change mapCell id (final.mem "r" j.1.val)=_
    rw [show final.mem "r" j.1.val=_ from got]
    rfl

/-- A zero-initialized native DW port pins both the payload and its real-cell
representation. Other cell dtypes read the distinct default Term constructor. -/
theorem zero_cell (c : Cell Term) (zero : c.read .real=Term.bits 0) : c=Cell.mk .real (Term.bits 0) := by
  cases c with
  | mk d v =>
    cases d <;> simp only [Cell.read] at zero
    all_goals cases zero
    rfl

theorem backward (R : Reductions) (h : Call) (epsilon : Term) :
    (NativeIO.program R h Bool.true epsilon).ImplementsOn (Reference.nativeDomain h Bool.true)
      (Reference.native R h Bool.true epsilon) := by
  intro vals domain bounds bounded initial binding
  refine ⟨NativeIOSafety.complete R h Bool.true epsilon bounds (IOBounds.native R h Bool.true epsilon vals bounds bounded) initial,?_⟩
  by_cases empty : h.rows=0
  · refine ⟨initial,by simp [NativeIO.program,NativeIO.schedule,empty,runSchedule],?_,?_⟩
    · intro o j active
      fin_cases o
      · have bad:=j.1.isLt
        simp [NativeIO.program,Ports.outputs,Call.cells,empty] at bad
      · have got:=binding ⟨4,by simp [NativeIO.program,Ports.inputs]⟩ (j.1,())
          (Or.inr ⟨rfl,Or.inr empty⟩)
        change (initial.mem "dw" j.1.val).read .real=Ports.ofNative h vals 4 j.1 at got
        have stored:=zero_cell (initial.mem "dw" j.1.val) (got.trans (domain rfl empty j.1))
        change mapCell id (initial.mem "dw" j.1.val)=_
        rw [stored]
        simp only [Reference.native,if_pos empty,mapCell,mapValue]
        rfl
    · intro r off _;rfl
  · obtain ⟨heap,ran,dx,dw,oldFrame⟩ := FPTensor.backward_raw (algebra R .weight) (algebra R .input)
      h FPEquiv.zero (nativeHeap initial) (Ports.logical h (Ports.ofNative h vals))
      (native_reads R h Bool.true epsilon vals initial binding)
    have mapped:=(native_backward R h epsilon empty initial).trans ran
    obtain ⟨final,actual,hm⟩ := Option.map_eq_some_iff.mp mapped
    subst heap
    refine ⟨final,actual,?_,frame R h Bool.true epsilon vals initial final oldFrame⟩
    intro o j active
    fin_cases o
    · let i : Fin h.rows:=⟨j.1.val/h.cols,(Nat.div_lt_iff_lt_mul h.positive).mpr j.1.isLt⟩
      let k : Fin h.cols:=⟨j.1.val%h.cols,Nat.mod_lt _ h.positive⟩
      have got:=dx i k
      have index : i.val*h.cols+k.val=j.1.val:=by simpa [i,k,Nat.mul_comm] using Nat.div_add_mod j.1.val h.cols
      rw [index] at got
      change mapCell id (final.mem "dx" j.1.val)=_
      rw [show final.mem "dx" j.1.val=_ from got]
      rfl
    · have got:=dw j.1
      rw [if_neg empty] at got
      change mapCell id (final.mem "dw" j.1.val)=_
      rw [show final.mem "dw" j.1.val=_ from got]
      simp only [Reference.native,if_neg empty]
      rfl

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.NativePublicIO
