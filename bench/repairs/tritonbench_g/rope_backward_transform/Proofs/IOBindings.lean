import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.IOGrid

namespace VeriTile.Bench.Repairs.RopeBackwardTransform.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

theorem q_binding {A : Type} [Inhabited A] (op : Operations A) (project : A → A)
    (r : Request) (xs : Inputs A r) (s : State A)
    (hin : (sourceIO op project r).InputBinding xs s)
    (key : Bool) (row head col : Nat)
    (hr : row<r.rows) (hh : head<r.heads key) (hc : col<r.dimension) :
    (s.mem (region key) (r.offset key row head col)).read .fp32 = q xs key row head col := by
  cases key
  · change head<r.qHeads at hh
    have h := hin ⟨0,by simp [sourceIO,inputs]⟩
      (⟨row,hr⟩,⟨head,hh⟩,⟨col,hc⟩,PUnit.unit) trivial
    change (s.mem _ _).read .fp32 = _ at h
    simpa only [q,hr,hh,hc,↓reduceDIte,↓reduceIte,Bool.false_eq_true] using h
  · change head<r.kHeads at hh
    have h := hin ⟨1,by simp [sourceIO,inputs]⟩
      (⟨row,hr⟩,⟨head,hh⟩,⟨col,hc⟩,PUnit.unit) trivial
    change (s.mem _ _).read .fp32 = _ at h
    simpa only [q,hr,hh,hc,↓reduceDIte,↓reduceIte,Bool.false_eq_true] using h

theorem coefficient_binding {A : Type} [Inhabited A] (op : Operations A) (project : A → A)
    (r : Request) (xs : Inputs A r) (s : State A)
    (hin : (sourceIO op project r).InputBinding xs s) (on : KernelContract.launched r)
    (sine : Bool) (row lane : Nat) (hr : row<r.sequence) (hl : lane<r.half) :
    (s.mem (if sine then "S" else "C")
      (row*r.rowStride sine+lane*r.colStride sine)).read .fp32 = coefficient xs sine row lane := by
  cases sine
  · have h := hin ⟨2,by simp [sourceIO,inputs]⟩ (⟨row,hr⟩,⟨lane,hl⟩,PUnit.unit) on
    simpa [coefficient,hr,hl,sourceIO,inputs,mapValue,Request.rowStride,Request.colStride] using h
  · have h := hin ⟨3,by simp [sourceIO,inputs]⟩ (⟨row,hr⟩,⟨lane,hl⟩,PUnit.unit) on
    simpa [coefficient,hr,hl,sourceIO,inputs,mapValue,Request.rowStride,Request.colStride] using h

def logicalResult {A : Type} [Inhabited A] (op : Operations A) (r : Request)
    (xs : Inputs A r) (key : Bool) (row head col : Nat) : A :=
  if col<2*r.half then
    let j := if col<r.half then col else col-r.half
    scalarPair op (decide (r.half≤col)) (q xs key row head j)
      (q xs key row head (j+r.half))
      (coefficient xs Bool.false (row%r.sequence) j)
      (coefficient xs Bool.true (row%r.sequence) j)
  else q xs key row head col

theorem logical_binding {A : Type} [Inhabited A] (op : Operations A) (project : A → A)
    (r : Request) (xs : Inputs A r) (s : State A)
    (hin : (sourceIO op project r).InputBinding xs s)
    (key : Bool) (row head col : Nat)
    (hr : row<r.rows) (hh : head<r.heads key) (hc : col<r.dimension) :
    scalarResult op r s.mem key row head col = logicalResult op r xs key row head col := by
  unfold scalarResult logicalResult
  split
  · rename_i active
    let j := if col<r.half then col else col-r.half
    have hj : j<r.half := by dsimp [j]; split <;> omega
    have dim : 2*r.half≤r.dimension := by unfold Request.half; omega
    have on : KernelContract.launched r := by
      refine ⟨by omega,by omega,?_⟩
      cases key <;> simp only [Request.heads,Bool.false_eq_true,↓reduceIte] at hh
      · exact Or.inl (by omega)
      · exact Or.inr (by omega)
    have q0 := q_binding op project r xs s hin key row head j hr hh (by omega)
    have q1 := q_binding op project r xs s hin key row head (j+r.half) hr hh (by omega)
    have cs := coefficient_binding op project r xs s hin on Bool.false (row%r.sequence) j
      (Nat.mod_lt _ (sequence_positive r hr)) hj
    have sn := coefficient_binding op project r xs s hin on Bool.true (row%r.sequence) j
      (Nat.mod_lt _ (sequence_positive r hr)) hj
    simp only [Request.rowStride,Request.colStride,Bool.false_eq_true,↓reduceIte] at cs sn
    dsimp only
    rw [q0,q1,cs,sn]
  · exact q_binding op project r xs s hin key row head col hr hh hc

theorem outside_not_written {A : Type} (op : Operations A) (project : A → A)
    (r : Request) (xs : Inputs A r) (name : RegionName) (offset : Nat)
    (h : (sourceIO op project r).toProgramIO.Outside xs name offset) :
    ¬KernelInterface.Written r name offset := by
  rintro ⟨key,row,head,col,rfl,hr,hh,hc,rfl⟩
  have dim : 2*r.half≤r.dimension := by unfold Request.half; omega
  have hd : col<r.dimension := by omega
  cases key
  · have no := h.1 ⟨0,by simp [sourceIO,outputs]⟩
      (⟨row,hr⟩,⟨head,hh⟩,⟨col,hd⟩,PUnit.unit) hc
    exact no.elim (fun a => a rfl) (fun a => a rfl)
  · have no := h.1 ⟨1,by simp [sourceIO,outputs]⟩
      (⟨row,hr⟩,⟨head,hh⟩,⟨col,hd⟩,PUnit.unit) hc
    exact no.elim (fun a => a rfl) (fun a => a rfl)

theorem implements {A : Type} [Inhabited A] (op : Operations A)
    (project : A → A) (r : Request) :
    (sourceIO op project r).Implements
      (fun xs => result (fun key row head col => project (logicalResult op r xs key row head col))) := by
  apply ReadProgramIO.ImplementsOn.intro
  · intro xs _ bounds hb initial _
    exact schedule_safe op project r xs bounds hb initial
  · intro xs _ bounds hb initial hin
    obtain ⟨old,executed,values,frame⟩ := KernelInterface.complete op r initial.mem
    have oldMem := gridExec_memory op.backend r executed
    obtain ⟨final,ran,mem⟩ := schedule_run op r (KernelContract.launchRows r) initial
    have same : final.mem=old := mem.trans oldMem.symm
    refine ⟨final,ran,?_,?_⟩
    · intro port j _
      rcases port with ⟨port,hp⟩
      have limit : port<2 := hp
      interval_cases port
      · change project ((final.mem (region Bool.false)
            (r.offset Bool.false j.1.val j.2.1.val j.2.2.1.val)).read .fp32) =
            project (logicalResult op r xs Bool.false j.1.val j.2.1.val j.2.2.1.val)
        rw [same,values Bool.false _ _ _ j.1.isLt j.2.1.isLt j.2.2.1.isLt,
          logical_binding op project r xs initial hin Bool.false _ _ _ j.1.isLt j.2.1.isLt j.2.2.1.isLt]
      · change project ((final.mem (region Bool.true)
            (r.offset Bool.true j.1.val j.2.1.val j.2.2.1.val)).read .fp32) =
            project (logicalResult op r xs Bool.true j.1.val j.2.1.val j.2.2.1.val)
        rw [same,values Bool.true _ _ _ j.1.isLt j.2.1.isLt j.2.2.1.isLt,
          logical_binding op project r xs initial hin Bool.true _ _ _ j.1.isLt j.2.1.isLt j.2.2.1.isLt]
    · intro name offset outside
      rw [same]
      exact frame name offset (outside_not_written op project r xs name offset outside)

#axiomsClean implements

/-- Short public-entry composition with an independent scalar formula. -/
theorem implements_formula {A : Type} [Inhabited A] (op : Operations A)
    (project : A → A) (r : Request)
    (f : Inputs A r → Bool → Nat → Nat → Nat → A)
    (formula : ∀ xs key row head col,
      project (logicalResult op r xs key row head col)=f xs key row head col) :
    (sourceIO op project r).Implements (fun xs => result (f xs)) := by
  simpa only [formula] using implements op project r

#axiomsClean implements_formula
end
end VeriTile.Bench.Repairs.RopeBackwardTransform.PublicIO
