import bench.repairs.tritonbench_g.context_attn_fwd.NativeIO

namespace VeriTile.Bench.Repairs.ContextAttnFwd.NativeArrayResults
open VeriTile.Triton FP.Structural NativeFormula NativeArrayBridge
noncomputable section
open Classical
variable {α : Type} [Inhabited α]

theorem mem_writes (p : Layout) (M : Algebra α) (dots : FPSuccess.DotSupport p M)
    (scale : ℝ) (job : Launch) (a : Arrays α) (off : Nat) (value : α) :
    (off,value) ∈ writes p M dots scale job a ↔
      ∃ i,active p (operands p job a) i ∧ address p (operands p job a) i=off ∧
        FPReference.output p M dots scale (operands p job a) i=value := by
  simp only [writes,List.mem_filterMap]
  constructor
  · rintro ⟨i,_,hi⟩
    split_ifs at hi with ha
    · exact ⟨i,ha,(Prod.mk.inj (Option.some.inj hi)).1,(Prod.mk.inj (Option.some.inj hi)).2⟩
  · rintro ⟨i,ha,ho,hv⟩
    exact ⟨i,TileShape.mem_allIndices _ i,by simp [ha,ho,hv]⟩

theorem mem_allWrites (p : Layout) (r : Regions) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (scale : ℝ) (jobs : List Launch) (a : Arrays α)
    (w : Nat×α) : w ∈ allWrites p r M dots scale jobs a ↔
      ∃ k : Fin jobs.length,w ∈ writes p M dots scale jobs[k]
        (before p r M dots scale jobs a k.val) := by
  induction jobs generalizing a with
  | nil => simp [allWrites]
  | cons job rest ih =>
    rw [allWrites,List.mem_append,ih]
    constructor
    · intro h
      rcases h with head | ⟨k,tail⟩
      · exact ⟨⟨0,by simp⟩,head⟩
      · exact ⟨k.succ,tail⟩
    · rintro ⟨k,hk⟩
      rcases k with ⟨k,hl⟩
      cases k with
      | zero => exact Or.inl hk
      | succ k => exact Or.inr ⟨⟨k,by simpa using hl⟩,hk⟩

theorem lastValue_append_one (xs : List (Nat×α)) (w : Nat×α) (off : Nat) :
    lastValue (xs++[w]) off=if off=w.1 then some w.2 else lastValue xs off := by
  simp only [lastValue,List.foldl_append,List.foldl_cons,List.foldl_nil]

theorem lastValue_none (xs : List (Nat×α)) (off : Nat) :
    lastValue xs off=none ↔ ∀ w ∈ xs,w.1≠off := by
  induction xs using List.reverseRecOn with
  | nil => simp [lastValue]
  | append_singleton xs w ih =>
    rw [lastValue_append_one]
    by_cases h : off=w.1
    · rw [if_pos h]
      constructor
      · intro impossible; cases impossible
      · intro absent
        exact False.elim (absent w (List.mem_append_right _ (by simp)) h.symm)
    · rw [if_neg h,ih]
      constructor
      · intro absent q member
        rcases List.mem_append.mp member with old | last
        · exact absent q old
        · have eq := List.mem_singleton.mp last
          subst q
          exact Ne.symm h
      · intro absent q member
        exact absent q (List.mem_append_left _ member)

theorem applyWrites_read (r : Regions) (xs : List (Nat×α)) (s : State α) (off : Nat) :
    (applyWrites r xs s).mem r.out off =
      (lastValue xs off).elim (s.mem r.out off) (Cell.mk .real) := by
  induction xs using List.reverseRecOn with
  | nil => rfl
  | append_singleton xs w ih =>
    rw [applyWrites_append,lastValue_append_one]
    by_cases h : off=w.1
    · simp [applyWrites,State.write,h]
    · simpa only [applyWrites,List.foldl_cons,List.foldl_nil,State.write,
        true_and,h,if_false,Option.elim] using ih

theorem applyWrites_value (r : Regions) (xs : List (Nat×α)) (s : State α) (off : Nat)
    (present : ∃ value,(off,value)∈xs) :
    (applyWrites r xs s).mem r.out off=.mk .real ((lastValue xs off).getD default) := by
  rw [applyWrites_read]
  cases h : lastValue xs off with
  | none =>
    obtain ⟨v,hv⟩ := present
    exact False.elim ((lastValue_none xs off).mp h (off,v) hv rfl)
  | some v => rfl

theorem applyWrites_frame (r : Regions) (xs : List (Nat×α)) (s : State α) (rg : RegionName) (off : Nat)
    (outside : ∀ w∈xs,rg≠r.out ∨ off≠w.1) :
    (applyWrites r xs s).mem rg off=s.mem rg off := by
  induction xs generalizing s with
  | nil => rfl
  | cons w rest ih =>
    change (applyWrites r rest (s.write r.out w.1 (.mk .real w.2))).mem rg off=_
    rw [ih _ (fun q hq => outside q (List.mem_cons_of_mem _ hq))]
    exact State.write_other _ _ _ _ _ _ (outside w List.mem_cons_self)

theorem bound_arrays (p : Layout) (r : Regions) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (scale : ℝ) (jobs : List Launch)
    (v : (NativeIO.io p r M dots scale jobs).Inputs) (s : State α)
    (bound : (NativeIO.io p r M dots scale jobs).InputBinding v s) :
    NativeArrayBridge.arrays r s=NativeIO.arrays r v := by
  have h0 := funext (bound (0 : Fin 6))
  have h1 := funext (bound (1 : Fin 6))
  have h2 := funext (bound (2 : Fin 6))
  have h3 := funext (bound (3 : Fin 6))
  have h4 := funext (bound (4 : Fin 6))
  have h5 := funext (bound (5 : Fin 6))
  change (fun k=>(s.mem r.query k).read .real)=_ at h0
  change (fun k=>(s.mem r.key k).read .real)=_ at h1
  change (fun k=>(s.mem r.value k).read .real)=_ at h2
  change (fun k=>(s.mem r.start k).read .int)=_ at h3
  change (fun k=>(s.mem r.sequence k).read .int)=_ at h4
  change (fun k=>(s.mem r.prompt k).read .int)=_ at h5
  unfold NativeArrayBridge.arrays NativeIO.arrays
  rw [h0,h1,h2,h3,h4,h5]
  rfl

theorem outputs_frame (p : Layout) (r : Regions) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (scale : ℝ) (jobs : List Launch)
    (v : (NativeIO.io p r M dots scale jobs).Inputs) (s t : State α)
    (memory : t.mem=(applyWrites r (allWrites p r M dots scale jobs (NativeIO.arrays r v)) s).mem) :
    (NativeIO.io p r M dots scale jobs).OutputBinding v (NativeIO.formula p r M dots scale jobs v) t ∧
      (NativeIO.io p r M dots scale jobs).Frame v s t := by
  constructor
  · intro o i hi
    have ho : o=(0 : Fin 1) := Fin.eq_zero o
    subst o
    change mapCell id (t.mem r.out (address p (NativeIO.eventOperands p r M dots scale jobs v i.1) i.2))=_
    rw [memory,applyWrites_value]
    · rfl
    · refine ⟨FPReference.output p M dots scale
        (NativeIO.eventOperands p r M dots scale jobs v i.1) i.2,?_⟩
      apply (mem_allWrites p r M dots scale jobs _ _).mpr
      refine ⟨i.1,?_⟩
      exact (mem_writes p M dots scale _ _ _ _).mpr ⟨i.2,hi,rfl,rfl⟩
  · intro rg off outside
    rw [memory]
    apply applyWrites_frame
    intro w hw
    obtain ⟨k,hk⟩ := (mem_allWrites p r M dots scale jobs _ w).mp hw
    obtain ⟨i,hi,ho,hv⟩ := (mem_writes p M dots scale _ _ w.1 w.2).mp hk
    have miss := outside.1 (0 : Fin 1) (k,i) hi
    change rg≠r.out ∨ off≠address p (NativeIO.eventOperands p r M dots scale jobs v k) i at miss
    simpa only [NativeIO.eventOperands,ho] using miss

theorem correct (p : Layout) (r : Regions) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (scale : ℝ) (jobs : List Launch) (positive : 0 < p.BN)
    (v : (NativeIO.io p r M dots scale jobs).Inputs) (s : State α)
    (bound : (NativeIO.io p r M dots scale jobs).InputBinding v s) :
    ∃ t,runSchedule ((NativeIO.io p r M dots scale jobs).schedule v) s=some t ∧
      (NativeIO.io p r M dots scale jobs).OutputBinding v (NativeIO.formula p r M dots scale jobs v) t ∧
      (NativeIO.io p r M dots scale jobs).Frame v s t := by
  obtain ⟨t,ran,memory⟩ := schedule p r scale M dots positive jobs s
  rw [bound_arrays p r M dots scale jobs v s bound] at memory
  exact ⟨t,ran,outputs_frame p r M dots scale jobs v s t memory⟩

#axiomsClean correct
end
end VeriTile.Bench.Repairs.ContextAttnFwd.NativeArrayResults
