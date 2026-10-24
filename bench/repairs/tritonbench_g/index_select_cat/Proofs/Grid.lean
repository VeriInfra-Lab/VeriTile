import bench.repairs.tritonbench_g.index_select_cat.Proofs.Program

namespace VeriTile.Bench.Repairs.IndexSelectCat.Grid
open VeriTile.Triton FP.Structural Program
noncomputable section

def flat (D row col : Nat) : Nat := row*D+col

def plan (N D : Nat) (indices : Nat → Nat) (row colTile : Nat) : Plan where
  inputBase := 0
  outputBase := 0
  indexBase := 0
  localRow := row
  colTile := colTile
  rows := N
  cols := D
  stride := D
  indices := indices

/-- The exact two-dimensional host grid, with no payload staging or chunks. -/
def plans (N D : Nat) (indices : Nat → Nat) : List Plan :=
  (List.range N).flatMap fun row =>
    (List.range ((D+511)/512)).map fun ct => plan N D indices row ct

theorem plan_key (N D : Nat) (indices : Nat → Nat) (r ct : Nat) (i : Lane) :
    key (plan N D indices r ct) i=flat D r (ct*512+i.2.1.val) := by
  simp [key,localOutput,plan,flat]

theorem plan_value {α : Type} (input : Nat → α) (N D : Nat) (indices : Nat → Nat)
    (r ct : Nat) (i : Lane) :
    value input (plan N D indices r ct) i=input (flat D (indices r) (ct*512+i.2.1.val)) := by
  simp [value,plan,flat]

theorem event_geometry {α : Type} (input : Nat → α) (N D : Nat) (indices : Nat → Nat)
    (event : Plan × Lane) (he : event∈expanded (plans N D indices)) :
    ∃ r c,r<N ∧ c<D ∧ key event.1 event.2=flat D r c ∧
      value input event.1 event.2=input (flat D (indices r) c) := by
  obtain ⟨p,hp,he⟩ := List.mem_flatMap.mp he
  obtain ⟨i,hi,rfl⟩ := List.mem_map.mp he
  obtain ⟨r,hr,hp⟩ := List.mem_flatMap.mp hp
  obtain ⟨ct,hct,rfl⟩ := List.mem_map.mp hp
  have mask : r<N ∧ ct*512+i.2.1.val<D := of_decide_eq_true (List.mem_filter.mp hi).2
  exact ⟨r,ct*512+i.2.1.val,mask.1,mask.2,plan_key ..,plan_value ..⟩

def coordinateLane (d : Nat) : Lane := (0,⟨d%512,Nat.mod_lt _ (by decide)⟩,PUnit.unit)

theorem coordinate_event (N D : Nat) (indices : Nat → Nat) (r c : Nat)
    (hr : r<N) (hc : c<D) :
    (plan N D indices r (c/512),coordinateLane c)∈expanded (plans N D indices) := by
  apply List.mem_flatMap.mpr
  refine ⟨plan N D indices r (c/512),?_,?_⟩
  · apply List.mem_flatMap.mpr
    refine ⟨r,List.mem_range.mpr hr,?_⟩
    exact List.mem_map.mpr ⟨c/512,List.mem_range.mpr (by omega),rfl⟩
  · apply List.mem_map.mpr
    refine ⟨coordinateLane c,List.mem_filter.mpr ⟨TileShape.mem_allIndices _ _,?_⟩,rfl⟩
    apply decide_eq_true
    change r<N ∧ c/512*512+c%512<D
    omega

theorem flat_injective (D r s c d : Nat) (hc : c<D) (hd : d<D)
    (same : flat D r c=flat D s d) : r=s ∧ c=d := by
  have positive : 0<D := by omega
  have h := congrArg (fun n=>n/D) same
  dsimp only [flat] at h
  rw [Nat.mul_comm r D,Nat.mul_add_div positive,Nat.mul_comm s D,Nat.mul_add_div positive] at h
  have hr : r=s := by simpa [Nat.div_eq_of_lt hc,Nat.div_eq_of_lt hd] using h
  constructor
  · exact hr
  · simp only [flat,hr] at same
    omega

/-- Full source-grid readback, allowing any number of repeated valid IDs. -/
theorem readback {α : Type} [Inhabited α] (rest : Algebra α) (N D : Nat)
    (indices : Nat → Nat) (input output : Nat → α) (r c : Nat) (hr : r<N) (hc : c<D) :
    Program.run rest input (plans N D indices) output (flat D r c)=input (flat D (indices r) c) := by
  rw [Program.run_eq_scatter]
  let selected := (plan N D indices r (c/512),coordinateLane c)
  have present := coordinate_event N D indices r c hr hc
  have ck : key selected.1 selected.2=flat D r c := by
    rw [plan_key]
    simp only [selected,coordinateLane]
    congr 1
    omega
  have cv : value input selected.1 selected.2=input (flat D (indices r) c) := by
    rw [plan_value]
    simp only [selected,coordinateLane]
    congr 2
    omega
  rw [←ck,←cv]
  apply Scatter.readback_on _ _ _ _ selected present
  intro event he same
  obtain ⟨u,d,hu,hd,hkey,hvalue⟩ := event_geometry input N D indices event he
  obtain ⟨ru,cd⟩ := flat_injective D u r d c hd hc (hkey.symm.trans (same.trans ck))
  rw [hvalue,cv,ru,cd]

theorem frame {α : Type} [Inhabited α] (rest : Algebra α) (N D : Nat)
    (indices : Nat → Nat) (input output : Nat → α) (q : Nat)
    (miss : ∀ r c,r<N → c<D → q≠flat D r c) :
    Program.run rest input (plans N D indices) output q=output q := by
  rw [Program.run_eq_scatter]
  apply Scatter.frame
  intro event he same
  obtain ⟨r,c,hr,hc,hkey,_⟩ := event_geometry input N D indices event he
  exact miss r c hr hc (same.trans hkey)

theorem empty (N D : Nat) (indices : Nat → Nat) (h : N=0 ∨ D=0) : plans N D indices=[] := by
  rcases h with h|h <;> subst_vars <;> simp [plans]

/-- Every complete ordering of the source word stores gives the same memory.
The value at a destination is unique even when source IDs repeat. -/
theorem order_independent {α : Type} (N D : Nat) (indices : Nat → Nat)
    (input output : Nat → α) (events : List (Plan × Lane))
    (complete : events.Perm (expanded (plans N D indices))) :
    Scatter.run events (fun e=>key e.1 e.2) (fun e=>value input e.1 e.2) output=
    Scatter.run (expanded (plans N D indices)) (fun e=>key e.1 e.2) (fun e=>value input e.1 e.2) output := by
  classical
  funext q
  have consistent : ∀ a∈expanded (plans N D indices), ∀ b∈expanded (plans N D indices),
      key a.1 a.2=key b.1 b.2 → value input a.1 a.2=value input b.1 b.2 := by
    intro a ha b hb equal
    obtain ⟨ar,ac,har,hac,ak,av⟩ := event_geometry input N D indices a ha
    obtain ⟨br,bc,hbr,hbc,bk,bv⟩ := event_geometry input N D indices b hb
    obtain ⟨rows,cols⟩ := flat_injective D ar br ac bc hac hbc (ak.symm.trans (equal.trans bk))
    rw [av,bv,rows,cols]
  by_cases present : ∃ e∈expanded (plans N D indices),key e.1 e.2=q
  · obtain ⟨e,he,hkey⟩ := present
    rw [←hkey]
    have lhs := Scatter.readback_on events (fun e=>key e.1 e.2) (fun e=>value input e.1 e.2)
      output e (complete.mem_iff.mpr he) (fun a ha eq=>consistent a (complete.mem_iff.mp ha) e he eq)
    have rhs := Scatter.readback_on (expanded (plans N D indices)) (fun e=>key e.1 e.2)
      (fun e=>value input e.1 e.2) output e he (fun a ha eq=>consistent a ha e he eq)
    exact lhs.trans rhs.symm
  · have lhs := Scatter.frame events (fun e=>key e.1 e.2) (fun e=>value input e.1 e.2) output q
      (fun a ha eq=>present ⟨a,complete.mem_iff.mp ha,eq.symm⟩)
    have rhs := Scatter.frame (expanded (plans N D indices)) (fun e=>key e.1 e.2)
      (fun e=>value input e.1 e.2) output q (fun a ha eq=>present ⟨a,ha,eq.symm⟩)
    exact lhs.trans rhs.symm

#axiomsClean readback
#axiomsClean frame
#axiomsClean empty
#axiomsClean order_independent
end
end VeriTile.Bench.Repairs.IndexSelectCat.Grid
