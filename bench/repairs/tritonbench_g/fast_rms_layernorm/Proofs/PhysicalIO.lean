import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.IOBinding

/-! Concrete finite-channel bindings for every original heap and the exact
existing zeroWeight initializer. KernelTensor.lift retains arbitrary physical
byte carriers, pointer arithmetic and the original read/read alias domain. -/
namespace VeriTile.Bench.Repairs.FastRmsLayernorm.PhysicalIO
open VeriTile Triton FP.Structural NativeGraph KernelTensor KernelStorage
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def realValues (h : Call) (heap : RealGridForward.Heap) : Ports.Values ℝ h :=
  fun i j=>RealGridForward.read (heap (Ports.inputRole i,j.val))

def nativeValues (h : Call) (heap : FPGridForward.Heap Term) : ChannelValues Term (Ports.inputs h) :=
  fun i=>Fin.cases (fun j=>(heap ("x",j.1.val)).read .real)
    (Fin.cases (fun j=>(heap ("w",j.1.val)).read .real)
      (Fin.cases (fun j=>(heap ("r",j.1.val)).read .real)
        (Fin.cases (fun j=>(heap ("dx",j.1.val)).read .real)
          (Fin.cases (fun j=>(heap ("dw",j.1.val)).read .real) (fun i=>Fin.elim0 i))))) i

theorem real_binding (h : Call) (backward : Bool) (heap : RealGridForward.Heap) :
    (RealIO.program h backward).InputBinding (RealGridForward.entry heap 0 h.rows) (realValues h heap) := by
  intro i j active
  rfl

theorem native_binding (R : Reductions) (h : Call) (backward : Bool) (epsilon : Term)
    (heap : FPGridForward.Heap Term) :
    (NativeIO.program R h backward epsilon).InputBinding (nativeValues h heap)
      (FPGridForward.entry heap 0 h.rows) := by
  intro i j active
  fin_cases i <;> rfl

/-- Every admitted physical reader instantiates the same regional IO without
canonicalizing its backing or excluding read-only aliases. -/
theorem real_physical {β : Type} (h : Call) (backward : Bool)
    (storage : Placement h (if backward then .backward else .forward))
    (heap : Nat → β) (decode : (Fin 4 → β) → MemCell) (other : MemCell) :
    (RealIO.program h backward).InputBinding
      (RealGridForward.entry (lift storage heap decode other) 0 h.rows)
      (realValues h (lift storage heap decode other)) := real_binding h backward _

theorem native_physical {β : Type} (R : Reductions) (h : Call) (backward : Bool) (epsilon : Term)
    (storage : Placement h (if backward then .backward else .forward))
    (heap : Nat → β) (decode : (Fin 4 → β) → Cell Term) (other : Cell Term) :
    (NativeIO.program R h backward epsilon).InputBinding (nativeValues h (lift storage heap decode other))
      (FPGridForward.entry (lift storage heap decode other) 0 h.rows) := native_binding R h backward epsilon _

theorem real_empty_domain (h : Call) (empty : h.rows=0) (heap : RealGridForward.Heap) :
    Reference.realDomain h Bool.true (realValues h (zeroWeight h (.real 0) heap)) := by
  intro _
  constructor
  · intro i
    have bad:=i.isLt
    omega
  · intro _ j
    change RealGridForward.read (zeroWeight h (.real 0) heap ("dw",j.val))=0
    simp [zeroWeight,j.isLt,RealGridForward.read]

theorem native_empty_domain (h : Call) (empty : h.rows=0) (heap : FPGridForward.Heap Term) :
    Reference.nativeDomain h Bool.true (nativeValues h (zeroWeight h FPEquiv.zero heap)) := by
  intro _ _ j
  change (zeroWeight h FPEquiv.zero heap ("dw",j.val)).read .real=Term.bits 0
  rw [show zeroWeight h FPEquiv.zero heap ("dw",j.val)=FPEquiv.zero by simp [zeroWeight,j.isLt]]
  rfl

def extent (h : Call) (backward : Bool) (storage : Placement h (if backward then .backward else .forward))
    (r : RegionName) : Nat :=
  if hr : r∈buffers (if backward then .backward else .forward) then (storage.allocation r hr).buffer.extent else 0

theorem capacity (h : Call) (backward : Bool) (storage : Placement h (if backward then .backward else .forward))
    (r : RegionName) (member : r∈buffers (if backward then .backward else .forward)) :
    cells h r≤extent h backward storage r := by
  simpa [extent,member] using (storage.allocation r member).holds

theorem real_bounds (h : Call) (backward : Bool) (storage : Placement h (if backward then .backward else .forward))
    (vals : (RealIO.program h backward).Ctx) (s : BlockState) :
    (RealIO.program h backward).InBounds (extent h backward storage) s vals := by
  dsimp only [PreparedAlgorithmProgramIO.InBounds,AlgorithmMemoryIO.InBounds,RealIO.program,RealIO.memory]
  refine ⟨?_,?_,?_⟩
  · intro i j active
    change Ports.active h backward i at active
    have member : Ports.inputRole i∈buffers (if backward then .backward else .forward) := by
      fin_cases i <;> cases backward <;> simp [Ports.inputRole,buffers,Ports.active] at *
      all_goals decide
    have size : Ports.inputSize h i=cells h (Ports.inputRole i) := by
      fin_cases i <;> rfl
    apply j.isLt.trans_le
    rw [size]
    exact capacity h backward storage _ member
  · intro o j active
    have member : Ports.outputRole backward o∈buffers (if backward then .backward else .forward) := by
      fin_cases o <;> cases backward <;> simp [Ports.outputRole,buffers]
      all_goals decide
    have size : Ports.outputSize h backward o=cells h (Ports.outputRole backward o) := by
      fin_cases o <;> cases backward <;> rfl
    apply j.isLt.trans_le
    rw [size]
    exact capacity h backward storage _ member
  · intro i;exact Fin.elim0 i

theorem native_bounds (R : Reductions) (h : Call) (backward : Bool) (epsilon : Term)
    (storage : Placement h (if backward then .backward else .forward))
    (vals : (NativeIO.program R h backward epsilon).Inputs) :
    (NativeIO.program R h backward epsilon).InBounds (extent h backward storage) vals := by
  refine ⟨?_,?_,?_,?_⟩
  · intro r outside
    by_cases member : r∈buffers (if backward then .backward else .forward)
    ·
      exfalso
      apply outside
      cases backward <;> simp_all [NativeIO.program,buffers]
    · simp [extent,member]
  · intro i j active
    cases backward
    · fin_cases i
      · have cap:=capacity h Bool.false storage "x" (by simp [buffers])
        have got:=j.1.isLt.trans_le (show h.cells≤extent h Bool.false storage "x" from by simpa [cells] using cap)
        exact got
      · have cap:=capacity h Bool.false storage "w" (by simp [buffers])
        have got:=j.1.isLt.trans_le (show h.cols≤extent h Bool.false storage "w" from by simpa [cells] using cap)
        exact got
      · simp [NativeIO.program,Ports.active] at active
      · simp [NativeIO.program,Ports.active] at active
      · simp [NativeIO.program,Ports.active] at active
    · fin_cases i
      · have cap:=capacity h Bool.true storage "x" (by simp [buffers])
        have got:=j.1.isLt.trans_le (show h.cells≤extent h Bool.true storage "x" from by simpa [cells] using cap)
        exact got
      · have cap:=capacity h Bool.true storage "w" (by simp [buffers])
        have got:=j.1.isLt.trans_le (show h.cols≤extent h Bool.true storage "w" from by simpa [cells] using cap)
        exact got
      · have cap:=capacity h Bool.true storage "r" (by simp [buffers])
        have got:=j.1.isLt.trans_le (show h.rows≤extent h Bool.true storage "r" from by simpa [cells] using cap)
        exact got
      · have cap:=capacity h Bool.true storage "dx" (by simp [buffers])
        have got:=j.1.isLt.trans_le (show h.cells≤extent h Bool.true storage "dx" from by simpa [cells] using cap)
        exact got
      · have cap:=capacity h Bool.true storage "dw" (by simp [buffers])
        have got:=j.1.isLt.trans_le (show h.cols≤extent h Bool.true storage "dw" from by simpa [cells] using cap)
        exact got
  · intro o j active
    cases backward
    · fin_cases o
      · have cap:=capacity h Bool.false storage "y" (by simp [buffers])
        have got:=j.1.isLt.trans_le (show h.cells≤extent h Bool.false storage "y" from by simpa [cells] using cap)
        exact got
      · have cap:=capacity h Bool.false storage "r" (by simp [buffers])
        have got:=j.1.isLt.trans_le (show h.rows≤extent h Bool.false storage "r" from by simpa [cells] using cap)
        exact got
    · fin_cases o
      · have cap:=capacity h Bool.true storage "dx" (by simp [buffers])
        have got:=j.1.isLt.trans_le (show h.cells≤extent h Bool.true storage "dx" from by simpa [cells] using cap)
        exact got
      · have cap:=capacity h Bool.true storage "dw" (by simp [buffers])
        have got:=j.1.isLt.trans_le (show h.cols≤extent h Bool.true storage "dw" from by simpa [cells] using cap)
        exact got
  · intro i;exact Fin.elim0 i

#axiomsClean real_binding
#axiomsClean native_binding
#axiomsClean real_empty_domain
#axiomsClean native_empty_domain
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.PhysicalIO
