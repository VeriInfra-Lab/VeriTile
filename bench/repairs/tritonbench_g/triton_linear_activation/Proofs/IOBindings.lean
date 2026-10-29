import bench.repairs.tritonbench_g.triton_linear_activation.IO
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.TritonLinearActivation.LinearIO
open VeriTile.Triton FP.Structural KernelSource KernelValues
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
variable {α β : Type} [Inhabited β]

theorem row_eq (p : Parameters) (pid : Nat) (s : State β) (hp : s.pids 0 = pid)
    (i : Fin p.blockRows) : KernelValues.row p s i = rowIndex p pid i := by
  have hs : KernelValues.groupSize p s = groupRows p pid := by
    unfold KernelValues.groupSize groupRows groupIdx group
    rw [hp]
    split <;> omega
  simp only [KernelValues.row,blockRow,groupIdx,rowIndex,group,hp,hs]

theorem col_eq (p : Parameters) (pid : Nat) (s : State β) (hp : s.pids 0 = pid)
    (i : Fin p.blockColumns) : KernelValues.col p s i = colIndex p pid i := by
  have hs : KernelValues.groupSize p s = groupRows p pid := by
    unfold KernelValues.groupSize groupRows groupIdx group
    rw [hp]
    split <;> omega
  simp only [KernelValues.col,blockCol,colIndex,hp,hs]

theorem active_eq (p : Parameters) (pid : Nat) (s : State β) (hp : s.pids 0 = pid)
    (i : TileIndex [p.blockRows,p.blockColumns]) :
    KernelValues.active p s i = Bool.true ↔ active p pid i := by
  simp [KernelValues.active,active,row_eq p pid s hp,col_eq p pid s hp]

@[simp] theorem launch_mem (M : Algebra β) (D : Dot β) (mode : Mode) (p : Parameters)
    (pid : Nat) (s : State β) : ((invocation M D mode p pid).initial s).mem = s.mem :=
  Invocation.initial_mem _ _

@[simp] theorem launch_pid (M : Algebra β) (D : Dot β) (mode : Mode) (p : Parameters)
    (pid : Nat) (s : State β) : ((invocation M D mode p pid).initial s).pids 0 = pid := by
  simp [invocation,Invocation.initial,bindRegisters]

structure Bound (encode : α → β) (mode : Mode) (p : Parameters) (pid : Nat)
    (values : Inputs α mode p) (s : State β) : Prop where
  left : ∀ (i : Fin p.blockRows) (k : Fin p.inner),
    read mode s p.input ((rowIndex p pid i % p.rows)*p.inputRowStride+k.val*p.inputInnerStride) =
      encode (toFloat (dtype mode) (values ⟨0,by simp [inputs]⟩ (i,k,PUnit.unit)))
  right : ∀ (j : Fin p.blockColumns) (k : Fin p.inner),
    read mode s p.weight ((colIndex p pid j % p.columns)*p.weightColStride+k.val*p.weightInnerStride) =
      encode (toFloat (dtype mode) (values ⟨1,by simp [inputs]⟩ (j,k,PUnit.unit)))
  bias : p.hasBias = Bool.true → ∀ j, colIndex p pid j % p.columns < p.columns →
    read mode s p.bias (colIndex p pid j % p.columns) =
      encode (toFloat (dtype mode) (values ⟨2,by simp [inputs]⟩ (j,PUnit.unit)))

theorem bound (M : Algebra β) (D : Dot β) (encode : α → β) (observe : β → α)
    (mode : Mode) (p : Parameters) (pid : Nat) (values : Inputs α mode p) (s : State β)
    (h : (programIO M D encode observe mode p pid).InputBinding values s) :
    Bound encode mode p pid values ((invocation M D mode p pid).initial s) := by
  constructor
  · intro i k
    have h' := h ⟨0,by simp [programIO,memory,inputs]⟩ (i,k,PUnit.unit) trivial
    cases mode <;> simpa [KernelValues.read,dtype,programIO,memory,inputs,mapValue,ofFloat,toFloat] using h'
  · intro i k
    have h' := h ⟨1,by simp [programIO,memory,inputs]⟩ (i,k,PUnit.unit) trivial
    cases mode <;> simpa [KernelValues.read,dtype,programIO,memory,inputs,mapValue,ofFloat,toFloat] using h'
  · intro bias j hj
    have h' := h ⟨2,by simp [programIO,memory,inputs]⟩ (j,PUnit.unit) ⟨bias,hj⟩
    cases mode <;> simpa [KernelValues.read,dtype,programIO,memory,inputs,mapValue,ofFloat,toFloat] using h'

theorem access_bounds (M : Algebra β) (D : Dot β) (encode : α → β) (observe : β → α)
    (mode : Mode) (p : Parameters) (pid : Nat) (values : Inputs α mode p)
    (bounds : RegionName → Nat) (hb : (programIO M D encode observe mode p pid).InBounds bounds values)
    (s : State β) (positive : 0 < p.blockInner) :
    SourceSafety.AccessBounds p ((invocation M D mode p pid).initial s) bounds := by
  let t := (invocation M D mode p pid).initial s
  have hp : t.pids 0 = pid := launch_pid M D mode p pid s
  constructor
  · intro bias j hj
    have h := hb.2.1 ⟨2,by simp [programIO,memory,inputs]⟩ (j,PUnit.unit)
      ⟨bias,by simpa [col_eq p pid _ (launch_pid M D mode p pid s)] using hj⟩
    simpa [programIO,memory,inputs,col_eq p pid _ (launch_pid M D mode p pid s)] using h
  · intro cur inside i mask
    have hk := (KernelRealArithmetic.mask_equiv p cur i.2.1.val positive inside i.2.1.isLt).mp mask
    have h := hb.2.1 ⟨0,by simp [programIO,memory,inputs]⟩
      (i.1,⟨cur*p.blockInner+i.2.1.val,hk⟩,PUnit.unit) trivial
    simpa [programIO,memory,inputs,aOffset,row_eq p pid _ (launch_pid M D mode p pid s),Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using h
  · intro cur inside i mask
    have hk := (KernelRealArithmetic.mask_equiv p cur i.1.val positive inside i.1.isLt).mp mask
    have h := hb.2.1 ⟨1,by simp [programIO,memory,inputs]⟩
      (i.2.1,⟨cur*p.blockInner+i.1.val,hk⟩,PUnit.unit) trivial
    simpa [programIO,memory,inputs,bOffset,col_eq p pid _ (launch_pid M D mode p pid s),Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using h
  · intro i hi
    have h := hb.2.2.1 ⟨0,by simp [programIO,memory,outputs]⟩ i ((active_eq p pid t hp i).mp hi)
    simpa [programIO,memory,outputs,KernelValues.outputOffset,row_eq p pid _ (launch_pid M D mode p pid s),col_eq p pid _ (launch_pid M D mode p pid s)] using h
  · intro saved i hi
    have h := hb.2.2.1 ⟨1,by simp [programIO,memory,outputs]⟩ i ⟨saved,(active_eq p pid t hp i).mp hi⟩
    simpa [programIO,memory,outputs,savedOffset,row_eq p pid _ (launch_pid M D mode p pid s),col_eq p pid _ (launch_pid M D mode p pid s)] using h

theorem schedule_safe (M : Algebra β) (D : Dot β) (encode : α → β) (observe : β → α)
    (mode : Mode) (p : Parameters) (pid : Nat) (values : Inputs α mode p)
    (bounds : RegionName → Nat) (hb : (programIO M D encode observe mode p pid).InBounds bounds values)
    (s : State β) (positive : 0 < p.blockInner) :
    ScheduleSafe bounds ((programIO M D encode observe mode p pid).schedule values) s := by
  change ScheduleSafe bounds [invocation M D mode p pid] s
  rw [scheduleSafe_singleton]
  exact SourceSafety.source_safe M D mode p _ bounds positive
    (access_bounds M D encode observe mode p pid values bounds hb s positive)

theorem frame (M : Algebra β) (D : Dot β) (encode : α → β) (observe : β → α)
    (mode : Mode) (p : Parameters) (pid : Nat) (values : Inputs α mode p) (s final : State β)
    (h : ∀ r o,
      (r ≠ p.output ∨ ∀ i, KernelValues.active p ((invocation M D mode p pid).initial s) i = Bool.true →
        o ≠ KernelValues.outputOffset p ((invocation M D mode p pid).initial s) i) →
      (p.saveInputs = Bool.false ∨ r ≠ p.saved ∨ ∀ i,
        KernelValues.active p ((invocation M D mode p pid).initial s) i = Bool.true →
        o ≠ savedOffset p ((invocation M D mode p pid).initial s) i) →
      final.mem r o = ((invocation M D mode p pid).initial s).mem r o) :
    (programIO M D encode observe mode p pid).Frame values s final := by
  intro r o outside
  apply h
  · by_cases hr : r = p.output
    · right
      intro i hi
      have hx := outside.1 ⟨0,by simp [programIO,memory,outputs]⟩ i
        ((active_eq p pid _ (launch_pid M D mode p pid s) i).mp hi)
      simpa [programIO,memory,outputs,hr,KernelValues.outputOffset,row_eq p pid _ (launch_pid M D mode p pid s),col_eq p pid _ (launch_pid M D mode p pid s)] using hx
    · exact Or.inl hr
  · cases saved : p.saveInputs
    · exact Or.inl rfl
    · right
      by_cases hr : r = p.saved
      · right
        intro i hi
        have hx := outside.1 ⟨1,by simp [programIO,memory,outputs]⟩ i
          ⟨saved,(active_eq p pid _ (launch_pid M D mode p pid s) i).mp hi⟩
        simpa [programIO,memory,outputs,hr,savedOffset,row_eq p pid _ (launch_pid M D mode p pid s),col_eq p pid _ (launch_pid M D mode p pid s)] using hx
      · exact Or.inl hr

#axiomsClean schedule_safe
end
end VeriTile.Bench.Repairs.TritonLinearActivation.LinearIO
