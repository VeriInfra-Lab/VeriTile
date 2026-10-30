import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.QuotedGridResult

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

theorem fillRange_congr (storage count : Nat) (f g : Nat → α) (m : Memory α)
    (he : ∀ i,i<count → f i=g i) : fillRange storage count f m=fillRange storage count g m := by
  funext address
  simp only [fillRange,fill_get,List.mem_range]
  split_ifs with h
  · exact he _ h.2.2
  · rfl

/-- Lower the proved exhaustive typed output stores to the fresh tensor
allocations. Payloads are read from the resulting Structural memory. -/
def exportForward [Inhabited α] (s : Shape) (alloc : Allocation 3 caller) (final : Bool)
    (m : Memory α) (result : State (Staged.Node α)) : Memory α :=
  let out := fillRange (alloc.storage 0) (s.NK*rowCount s s.DV) (readWord result "o") m
  if final then fillRange (alloc.storage 1) (s.BH*s.DK*s.DV) (readWord result "final_state") out else out

def exportBackward [Inhabited α] (s : Shape) (alloc : Allocation 6 caller)
    (m : Memory α) (result : State (Staged.Node α)) : Memory α :=
  let dq := fillRange (alloc.storage 0) (s.NV*rowCount s s.DK) (readWord result "dq") m
  let dk := fillRange (alloc.storage 1) (s.NV*rowCount s s.DK) (readWord result "dk") dq
  fillRange (alloc.storage 2) (s.NK*rowCount s s.DV) (readWord result "dv") dk

def forward [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (p : PlanBinding s)
    (alloc : Allocation 3 caller) (seed final : Bool) (m : Memory α) : Option (Memory α) :=
  (gridExec .forward a s b p.plans seed final (gridPrograms s) (initialState a b m)).map
    (fun result => reduction a p.plans.sumO (alloc.storage 0) (alloc.storage 2) (rowCount s s.DV)
      (exportForward s alloc final m result))

def backward [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (p : PlanBinding s)
    (alloc : Allocation 6 caller) (seed final : Bool) (m : Memory α) : Option (Memory α) :=
  (gridExec .backward a s b p.plans seed final (gridPrograms s) (initialState a b m)).map
    (fun result =>
      let partials := exportBackward s alloc m result
      let dq := reduction a p.plans.sumQ (alloc.storage 0) (alloc.storage 3) (rowCount s s.DK) partials
      let dk := reduction a p.plans.sumK (alloc.storage 1) (alloc.storage 4) (rowCount s s.DK) dq
      reduction a p.plans.sumV (alloc.storage 2) (alloc.storage 5) (rowCount s s.DV) dk)

theorem exportForward_eq [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (p : PlanBinding s)
    (alloc : Allocation 3 caller) (seed final : Bool) (m : Memory α) (result : State (Staged.Node α))
    (fresh : ∀ j,alloc.storage j∉b.storageIds)
    (hm : ∀ r i,result.mem r i=if writtenRange .forward s final r i then
      payload a s p.plans (inputs (layout s b) (initialState a b m)) seed final r i else (initialState a b m).mem r i) :
    exportForward s alloc final m result=
      (if final then sourcePhase a s p.plans b seed Bool.false .finalState (alloc.storage 1) (s.BH*s.DK*s.DV)
        (sourcePhase a s p.plans b seed Bool.false .output (alloc.storage 0) (s.NK*rowCount s s.DV) m)
      else sourcePhase a s p.plans b seed Bool.false .output (alloc.storage 0) (s.NK*rowCount s s.DV) m) := by
  have ho : ∀ i,i<s.NK*rowCount s s.DV → readWord result "o" i=
      sourceCell a s p.plans (b.observe s m) seed Bool.false .output i := by
    intro i hi
    have h := grid_read .forward a s b p seed final _ result hm .output i (Or.inl ⟨rfl,hi⟩)
    simpa only [initialState_inputs,sourceCell,region] using h
  have hf : final=Bool.true → ∀ i,i<s.BH*s.DK*s.DV → readWord result "final_state" i=
      sourceCell a s p.plans (b.observe s m) seed Bool.false .finalState i := by
    intro hfinal i hi
    have h := grid_read .forward a s b p seed final _ result hm .finalState i (Or.inr ⟨hfinal,rfl,hi⟩)
    simpa only [initialState_inputs,sourceCell,region] using h
  unfold exportForward sourcePhase
  rw [fillRange_congr _ _ _ _ m ho]
  cases final
  · rfl
  · simp only [reduceIte]
    rw [fillRange_observations b s _ _ _ m (fresh 0)]
    exact fillRange_congr _ _ _ _ _ (hf rfl)

theorem exportBackward_eq [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (p : PlanBinding s)
    (alloc : Allocation 6 caller) (seed final : Bool) (m : Memory α) (result : State (Staged.Node α))
    (fresh : ∀ j,alloc.storage j∉b.storageIds)
    (hm : ∀ r i,result.mem r i=if writtenRange .backward s final r i then
      payload a s p.plans (inputs (layout s b) (initialState a b m)) seed final r i else (initialState a b m).mem r i) :
    exportBackward s alloc m result=
      sourcePhase a s p.plans b seed final .dv (alloc.storage 2) (s.NK*rowCount s s.DV)
        (sourcePhase a s p.plans b seed final .dk (alloc.storage 1) (s.NV*rowCount s s.DK)
          (sourcePhase a s p.plans b seed final .dq (alloc.storage 0) (s.NV*rowCount s s.DK) m)) := by
  have hq : ∀ i,i<s.NV*rowCount s s.DK → readWord result "dq" i=
      sourceCell a s p.plans (b.observe s m) seed final .dq i := by
    intro i hi
    simpa only [initialState_inputs,region] using
      grid_read .backward a s b p seed final _ result hm .dq i (Or.inl ⟨rfl,hi⟩)
  have hk : ∀ i,i<s.NV*rowCount s s.DK → readWord result "dk" i=
      sourceCell a s p.plans (b.observe s m) seed final .dk i := by
    intro i hi
    simpa only [initialState_inputs,region] using
      grid_read .backward a s b p seed final _ result hm .dk i (Or.inr (Or.inl ⟨rfl,hi⟩))
  have hv : ∀ i,i<s.NK*rowCount s s.DV → readWord result "dv" i=
      sourceCell a s p.plans (b.observe s m) seed final .dv i := by
    intro i hi
    simpa only [initialState_inputs,region] using
      grid_read .backward a s b p seed final _ result hm .dv i (Or.inr (Or.inr ⟨rfl,hi⟩))
  unfold exportBackward sourcePhase
  rw [fillRange_congr _ _ _ _ m hq]
  simp only [fillRange_observations b s _ _ _ _ (fresh 0),
    fillRange_observations b s _ _ _ _ (fresh 1)]
  rw [fillRange_congr _ _ _ _ _ hk,fillRange_congr _ _ _ _ _ hv]

theorem forward_eq [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (p : PlanBinding s)
    (alloc : Allocation 3 caller) (seed final : Bool) (m : Memory α) (domain : StoreDomain s b)
    (fresh : ∀ j,alloc.storage j∉b.storageIds) :
    forward a s b p alloc seed final m=some (sourceForward a s p.plans b alloc seed final m) := by
  obtain ⟨result,he,hm⟩ := grid_execution_memory .forward a s b p (initialState a b m) seed final domain
  unfold forward
  rw [he,Option.map_some,exportForward_eq a s b p alloc seed final m result fresh hm]
  rfl

theorem backward_eq [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (p : PlanBinding s)
    (alloc : Allocation 6 caller) (seed final : Bool) (m : Memory α) (domain : StoreDomain s b)
    (fresh : ∀ j,alloc.storage j∉b.storageIds) :
    backward a s b p alloc seed final m=some (sourceBackward a s p.plans b alloc seed final m) := by
  obtain ⟨result,he,hm⟩ := grid_execution_memory .backward a s b p (initialState a b m) seed final domain
  unfold backward
  rw [he,Option.map_some,exportBackward_eq a s b p alloc seed final m result fresh hm]
  rfl

#axiomsClean exportForward_eq
#axiomsClean exportBackward_eq
#axiomsClean forward_eq
#axiomsClean backward_eq
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
