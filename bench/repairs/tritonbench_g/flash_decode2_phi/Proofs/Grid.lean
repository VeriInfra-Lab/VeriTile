import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.Execution

namespace VeriTile.Bench.Repairs.FlashDecode2.Grid
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Ordinary two-dimensional launch indexed by its bijective flattened row. -/
def setup (B H row : Nat) (s : State α) : State α :=
  {s with regs := fun _ _ _ => none
          pids := fun a => if a=0 then row/H else if a=1 then row%H else 0
          numPids := fun a => if a=0 then B else if a=1 then H else 1}

def InputsEq (s t : State α) : Prop :=
  (∀ a,s.mem "L" a=t.mem "L" a) ∧
  (∀ a,s.mem "M" a=t.mem "M" a) ∧ (∀ a,s.mem "E" a=t.mem "E" a)

theorem output_congr (A : Algebra α) (s t : State α) (masked : Bool) (H S D Q BD : Nat)
    (hm : InputsEq s t) (hp : s.pids=t.pids) :
    output A s masked H S D Q BD=output A t masked H S D Q BD := by
  have hc : count s Q=count t Q := by simp [count,hp,hm.1]
  have hx : ∀ j, logvalue A s H S j=logvalue A t H S j := by
    intro j; simp [logvalue,eoff,hp,hm.2.2]
  have hv : ∀ j (k : TileIndex [BD]), value A s masked H S D j k=value A t masked H S D j k := by
    intro j k; simp [value,voff,hp,hm.2.1]
  have hr : ∀ j, values A s masked H S D BD j=values A t masked H S D BD j := by
    intro j; induction j with
    | zero => rfl
    | succ j ih => simp [values,next,ih,hx,hv]
  funext k
  simp [output,hc,hr]

theorem output_offset (B H row : Nat) (s : State α) (D : Nat) (k : TileIndex [BD]) :
    ooff (setup B H row s) H D k=row*D+lane k := by
  dsimp [ooff,setup]
  have h := Nat.div_add_mod row H
  nlinarith

theorem active_iff (masked : Bool) (D BD : Nat) (domain : masked=Bool.true ∨ BD=D)
    (k : TileIndex [BD]) : active masked D k ↔ lane k<D := by
  rcases domain with h|h
  · simp [active,h]
  · subst BD
    simp [active,lane,k.1.isLt]

def value (A : Algebra α) (s : State α) (masked : Bool) (B H S D Q BD row : Nat) :=
  output A (setup B H row s) masked H S D Q BD

def block (A : Algebra α) (s : State α) (masked : Bool) (B H S D Q BD row : Nat) : State α :=
  scatter (setup B H row s) H D BD masked (value A s masked B H S D Q BD row)

theorem value_congr (A : Algebra α) (s t : State α) (masked : Bool) (B H S D Q BD row : Nat)
    (h : InputsEq s t) : value A s masked B H S D Q BD row=value A t masked B H S D Q BD row :=
  output_congr A _ _ masked H S D Q BD h rfl

theorem writes_congr {ι : Type} (xs : List ι) (off : ι → Nat)
    (active : ι → Prop) [DecidablePred active] (v w : ι → Cell α)
    (s t : State α) (hm : s.mem=t.mem) (hv : ∀ i,active i → v i=w i) :
    (xs.foldl (fun st i => if active i then st.write "O" (off i) (v i) else st) s).mem =
      (xs.foldl (fun st i => if active i then st.write "O" (off i) (w i) else st) t).mem := by
  induction xs generalizing s t with
  | nil => exact hm
  | cons i xs ih =>
    simp only [List.foldl_cons]
    apply ih
    by_cases h : active i
    · simp [h,State.write,hm,hv i h]
    · simp [h,hm]

theorem block_congr (A : Algebra α) (s t : State α) (masked : Bool) (B H S D Q BD row : Nat)
    (h : s.mem=t.mem) :
    (block A s masked B H S D Q BD row).mem=(block A t masked B H S D Q BD row).mem := by
  have hi : InputsEq s t := ⟨congrFun (congrFun h "L"),
    congrFun (congrFun h "M"),congrFun (congrFun h "E")⟩
  unfold block scatter
  simp only [output_offset]
  apply writes_congr _ _ _ _ _ (setup B H row s) (setup B H row t)
    (show (setup B H row s).mem=(setup B H row t).mem from h)
  intro k _
  rw [value_congr A s t masked B H S D Q BD row hi]

def grid (A : Algebra α) (masked : Bool) (B H S D Q BD : Nat) : List Nat → State α → State α
  | [],s => s
  | row::rest,s => grid A masked B H S D Q BD rest (block A s masked B H S D Q BD row)

def source (A : Algebra α) (masked : Bool) (B H S D Q BD : Nat) : List Nat → State α → Option (State α)
  | [],s => some s
  | row::rest,s => (FP.Structural.exec A (kernel masked H S D Q BD) (setup B H row s)).bind
      (source A masked B H S D Q BD rest)

theorem grid_congr (A : Algebra α) (masked : Bool) (B H S D Q BD : Nat)
    (rows : List Nat) (s t : State α) (h : s.mem=t.mem) :
    (grid A masked B H S D Q BD rows s).mem=(grid A masked B H S D Q BD rows t).mem := by
  induction rows generalizing s t with
  | nil => exact h
  | cons row rest ih => exact ih _ _ (block_congr A s t masked B H S D Q BD row h)

theorem source_complete (A : Algebra α) (masked : Bool) (B H S D Q BD : Nat)
    (rows : List Nat) (s : State α) : ∃ t, source A masked B H S D Q BD rows s=some t ∧
      t.mem=(grid A masked B H S D Q BD rows s).mem := by
  induction rows generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons row rest ih =>
    obtain ⟨t,ht,hm⟩ := complete A (setup B H row s) masked H S D Q BD
    obtain ⟨u,hu,hv⟩ := ih t
    refine ⟨u,by simpa [source,ht] using hu,?_⟩
    exact hv.trans (grid_congr A masked B H S D Q BD rest _ _ hm)

theorem block_frame (A : Algebra α) (s : State α) (masked : Bool) (B H S D Q BD row : Nat)
    (domain : masked=Bool.true ∨ BD=D) (r : RegionName) (a : Nat)
    (outside : r≠("O" : RegionName) ∨ ∀ k,k<D → a≠row*D+k) :
    (block A s masked B H S D Q BD row).mem r a=s.mem r a := by
  apply scatter_frame
  rcases outside with h|h
  · exact Or.inl h
  · right
    intro k hk
    rw [output_offset]
    exact h (lane k) ((active_iff masked D BD domain k).mp hk)

theorem block_inputs (A : Algebra α) (s : State α) (masked : Bool) (B H S D Q BD row : Nat)
    (domain : masked=Bool.true ∨ BD=D) : InputsEq (block A s masked B H S D Q BD row) s := by
  refine ⟨?_,?_,?_⟩ <;> intro a <;>
    exact block_frame A s masked B H S D Q BD row domain _ a (.inl (by decide))

theorem grid_frame (A : Algebra α) (masked : Bool) (B H S D Q BD : Nat)
    (domain : masked=Bool.true ∨ BD=D) (rows : List Nat) (s : State α) (r : RegionName) (a : Nat)
    (outside : r≠("O" : RegionName) ∨ ∀ row∈rows,∀ k,k<D → a≠row*D+k) :
    (grid A masked B H S D Q BD rows s).mem r a=s.mem r a := by
  induction rows generalizing s with
  | nil => rfl
  | cons row rest ih =>
    rw [grid,ih]
    · apply block_frame A s masked B H S D Q BD row domain
      exact outside.imp_right (fun h k hk => h row (by simp) k hk)
    · exact outside.imp_right (fun h q hq => h q (by simp [hq]))

theorem row_offset_ne (D r q k j : Nat) (hk : k<D) (hj : j<D) (hr : r≠q) :
    r*D+k≠q*D+j := by
  intro h
  have hD : 0<D := by omega
  have same := congrArg (fun a => a/D) h
  have left : (r*D+k)/D=r := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hD,Nat.div_eq_of_lt hk]; simp
  have right : (q*D+j)/D=q := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hD,Nat.div_eq_of_lt hj]; simp
  exact hr (left.symm.trans (same.trans right))

theorem block_readback (A : Algebra α) (s : State α) (masked : Bool) (B H S D Q BD row : Nat)
    (domain : masked=Bool.true ∨ BD=D) (k : TileIndex [BD]) (hk : lane k<D) :
    (block A s masked B H S D Q BD row).mem "O" (row*D+lane k)=
      .mk .real (value A s masked B H S D Q BD row k) := by
  rw [←output_offset B H row s D k]
  exact scatter_readback _ H D BD masked _ k ((active_iff masked D BD domain k).mpr hk)

theorem grid_readback (A : Algebra α) (masked : Bool) (B H S D Q BD : Nat)
    (domain : masked=Bool.true ∨ BD=D) (rows : List Nat) (s : State α)
    (row : Nat) (member : row∈rows) (k : TileIndex [BD]) (hk : lane k<D) :
    (grid A masked B H S D Q BD rows s).mem "O" (row*D+lane k)=
      .mk .real (value A s masked B H S D Q BD row k) := by
  induction rows generalizing s with
  | nil => simp at member
  | cons r rest ih =>
    by_cases tail : row∈rest
    · rw [grid,ih _ tail]
      rw [value_congr A _ s masked B H S D Q BD row
        (block_inputs A s masked B H S D Q BD r domain)]
    · have same : row=r := by simpa [tail] using member
      subst r
      rw [grid,grid_frame]
      · exact block_readback A s masked B H S D Q BD row domain k hk
      · exact domain
      · right
        intro q hq j hj
        exact row_offset_ne D row q (lane k) j hk hj (fun h => tail (h ▸ hq))

theorem grid_coverage (B H b h : Nat) (hb : b<B) (hh : h<H) :
    b*H+h∈List.range (B*H) ∧ (b*H+h)/H=b ∧ (b*H+h)%H=h := by
  have hH : 0<H := by omega
  constructor
  · simp only [List.mem_range]
    nlinarith
  · constructor
    · rw [Nat.add_comm,Nat.add_mul_div_right _ _ hH,Nat.div_eq_of_lt hh]; simp
    · simp [Nat.add_mod,Nat.mod_eq_of_lt hh]

#axiomsClean source_complete
#axiomsClean grid_readback
#axiomsClean grid_frame
#axiomsClean grid_coverage
end
end VeriTile.Bench.Repairs.FlashDecode2.Grid
