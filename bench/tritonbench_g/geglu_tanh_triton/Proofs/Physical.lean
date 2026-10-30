import bench.tritonbench_g.geglu_tanh_triton.Kernels
import bench.tritonbench_g.swiglu_triton.Contract

namespace VeriTile.Bench.Repairs.GegluTanhTriton.Physical
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option maxHeartbeats 8000000

abbrev Role := VeriTile.Bench.Repairs.SwigluTriton.DirectPhysical.Role

def outputRole : Channel → Role
  | .c => .third | .da => .a | .db => .b

structure Layout where
  allocation : Role → Allocation
  count : Nat

def Layout.key (p : Layout) (role : Role) (i : Nat) : Key :=
  (p.allocation role).key i

structure Valid (p : Layout) : Prop where
  capacity : ∀ role,p.count≤(p.allocation role).words
  separate : ∀ a b,a≠b → (p.allocation a).separate (p.allocation b)

theorem backed (p : Layout) (valid : Valid p) (role : Role) (i : Nat) (hi : i<p.count) :
    (p.allocation role).contains (p.key role i) :=
  Allocation.key_contains _ _ (lt_of_lt_of_le hi (valid.capacity role))

theorem keys_unique (p : Layout) (valid : Valid p) (a b : Role) (i j : Nat)
    (hi : i<p.count) (hj : j<p.count) (equal : p.key a i=p.key b j) : a=b ∧ i=j := by
  by_cases same : a=b
  · subst b
    have words := congrArg Key.address equal
    simp only [Layout.key,Allocation.key] at words
    exact ⟨rfl,by omega⟩
  · exact False.elim ((Allocation.keys_separate _ _ (valid.separate a b same) _ _
      (backed p valid a i hi) (backed p valid b j hj)) equal)

theorem keys_disjoint (p : Layout) (valid : Valid p) (a b : Role) (i j : Nat)
    (hi : i<p.count) (hj : j<p.count) (different : a≠b ∨ i≠j) :
    (p.key a i).disjoint (p.key b j) := by
  by_cases same : a=b
  · subst b
    exact Allocation.indices_disjoint _ _ _ (different.resolve_left (by simp))
  · exact Allocation.spans_separate _ _ (valid.separate a b same) _ _
      (backed p valid a i hi) (backed p valid b j hj)

def lanes (count pid : Nat) : List Nat :=
  (List.range 128).filter (fun lane => pid*128+lane<count)

theorem lanes_mem (count pid lane : Nat) :
    lane∈lanes count pid ↔ lane<128 ∧ pid*128+lane<count := by
  simp [lanes]

def value {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : Channel → α → α → α → α) (backward : Bool) (p : Layout)
    (s : M) (ch : Channel) (i : Nat) : α :=
  compute ch (ops.read s (p.key .a i)) (ops.read s (p.key .b i))
    (if backward then ops.read s (p.key .third i) else default)

/-- Both scatters use values loaded from this exact tile-entry physical heap.
Only active lanes are written, in the source's A-then-B store order. -/
def tile {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : Channel → α → α → α → α) (backward : Bool) (p : Layout) (pid : Nat) (s : M) : M :=
  let scatter := fun ch current => stores ops (fun lane => p.key (outputRole ch) (pid*128+lane))
    (fun lane => value ops compute backward p s ch (pid*128+lane)) (lanes p.count pid) current
  if backward then scatter .db (scatter .da s) else scatter .c s

theorem tile_frame {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : Channel → α → α → α → α) (backward : Bool) (p : Layout)
    (pid : Nat) (s : M) (observed : Key)
    (outside : ∀ ch,enabled backward ch → ∀ lane,lane∈lanes p.count pid →
      observed.disjoint (p.key (outputRole ch) (pid*128+lane))) :
    ops.read (tile ops compute backward p pid s) observed=ops.read s observed := by
  cases backward <;> simp only [tile,if_false,if_true]
  · exact read_frame ops _ _ _ _ _ (outside .c rfl)
  · rw [read_frame ops _ _ _ _ _ (outside .db (by decide))]
    exact read_frame ops _ _ _ _ _ (outside .da (by decide))

theorem tile_bytes {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : Channel → α → α → α → α) (backward : Bool) (p : Layout)
    (pid : Nat) (s : M) (owner address : Nat)
    (outside : ∀ ch,enabled backward ch → ∀ lane,lane∈lanes p.count pid →
      ¬(p.key (outputRole ch) (pid*128+lane)).covers owner address) :
    ops.observe (tile ops compute backward p pid s) owner address=ops.observe s owner address := by
  cases backward <;> simp only [tile,if_false,if_true]
  · exact bytes_frame ops _ _ _ _ _ _ (outside .c rfl)
  · rw [bytes_frame ops _ _ _ _ _ _ (outside .db (by decide))]
    exact bytes_frame ops _ _ _ _ _ _ (outside .da (by decide))

theorem tile_values {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : Channel → α → α → α → α) (backward : Bool) (p : Layout) (valid : Valid p)
    (pid : Nat) (s : M) (ch : Channel) (on : enabled backward ch)
    (lane : Nat) (active : lane∈lanes p.count pid) :
    ops.read (tile ops compute backward p pid s) (p.key (outputRole ch) (pid*128+lane))=
      value ops compute backward p s ch (pid*128+lane) := by
  have hi := (lanes_mem _ _ _).mp active |>.2
  have injective (role : Role) (j : Nat) (hj : j∈lanes p.count pid)
      (equal : p.key role (pid*128+j)=p.key role (pid*128+lane)) : j=lane := by
    have h := (keys_unique p valid role role _ _ ((lanes_mem _ _ _).mp hj).2 hi equal).2
    omega
  have separate (role : Role) (j : Nat) (hj : j∈lanes p.count pid)
      (different : p.key role (pid*128+j)≠p.key role (pid*128+lane)) :
      (p.key role (pid*128+lane)).disjoint (p.key role (pid*128+j)) := by
    apply keys_disjoint p valid role role _ _ hi ((lanes_mem _ _ _).mp hj).2
    exact Or.inr (fun equal => different (congrArg (p.key role) equal.symm))
  cases backward <;> cases ch <;> simp only [tile,if_false,if_true,outputRole]
  · exact readback ops _ _ _ _ _ active (fun j hj h => by rw [injective .third j hj h]) (separate .third)
  · simp [enabled] at on
  · simp [enabled] at on
  · simp [enabled] at on
  · rw [read_frame ops _ _ _ _ _ (fun j hj =>
      keys_disjoint p valid .a .b _ _ hi ((lanes_mem _ _ _).mp hj).2 (Or.inl (by decide)))]
    exact readback ops _ _ _ _ _ active (fun j hj h => by rw [injective .a j hj h]) (separate .a)
  · exact readback ops _ _ _ _ _ active (fun j hj h => by rw [injective .b j hj h]) (separate .b)

theorem tile_other {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : Channel → α → α → α → α) (backward : Bool) (p : Layout) (valid : Valid p)
    (pid : Nat) (s : M) (role : Role) (i : Nat) (hi : i<p.count) (different : i/128≠pid) :
    ops.read (tile ops compute backward p pid s) (p.key role i)=ops.read s (p.key role i) := by
  apply tile_frame
  intro ch on lane member
  obtain ⟨laneBound,inside⟩ := (lanes_mem _ _ _).mp member
  apply keys_disjoint p valid role (outputRole ch) i _ hi inside
  right
  intro equal
  apply different
  omega

def run {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : Channel → α → α → α → α) (backward : Bool) (p : Layout) : List Nat → M → M
  | [],s => s
  | pid::rest,s => run ops compute backward p rest (tile ops compute backward p pid s)

theorem run_frame {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : Channel → α → α → α → α) (backward : Bool) (p : Layout)
    (schedule : List Nat) (s : M) (observed : Key)
    (outside : ∀ pid∈schedule,∀ ch,enabled backward ch → ∀ lane,lane∈lanes p.count pid →
      observed.disjoint (p.key (outputRole ch) (pid*128+lane))) :
    ops.read (run ops compute backward p schedule s) observed=ops.read s observed := by
  induction schedule generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    rw [run,ih _ (fun j hj => outside j (by simp [hj]))]
    exact tile_frame ops compute backward p pid s observed (outside pid (by simp))

theorem run_bytes {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : Channel → α → α → α → α) (backward : Bool) (p : Layout)
    (schedule : List Nat) (s : M) (owner address : Nat)
    (outside : ∀ pid∈schedule,∀ ch,enabled backward ch → ∀ lane,lane∈lanes p.count pid →
      ¬(p.key (outputRole ch) (pid*128+lane)).covers owner address) :
    ops.observe (run ops compute backward p schedule s) owner address=ops.observe s owner address := by
  induction schedule generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    rw [run,ih _ (fun j hj => outside j (by simp [hj]))]
    exact tile_bytes ops compute backward p pid s owner address (outside pid (by simp))

theorem run_values {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : Channel → α → α → α → α) (backward : Bool) (p : Layout) (valid : Valid p)
    (schedule : List Nat) (unique : schedule.Nodup) (s : M)
    (ch : Channel) (on : enabled backward ch) (i : Nat) (hi : i<p.count)
    (member : i/128∈schedule) :
    ops.read (run ops compute backward p schedule s) (p.key (outputRole ch) i)=
      value ops compute backward p s ch i := by
  induction schedule generalizing s with
  | nil => simp at member
  | cons head rest ih =>
    obtain ⟨absent,uniqueRest⟩ := List.nodup_cons.mp unique
    rcases List.mem_cons.mp member with equal | tail
    · have keep : ops.read (run ops compute backward p rest (tile ops compute backward p head s))
          (p.key (outputRole ch) i)=ops.read (tile ops compute backward p head s) (p.key (outputRole ch) i) := by
        apply run_frame
        intro pid hp other onOther lane hl
        obtain ⟨laneBound,inside⟩ := (lanes_mem _ _ _).mp hl
        apply keys_disjoint p valid _ _ _ _ hi inside
        right
        intro same
        have pidEq : head=pid := by omega
        exact absent (pidEq ▸ hp)
      rw [run,keep]
      have index : head*128+i%128=i := by omega
      have active : i%128∈lanes p.count head := (lanes_mem _ _ _).mpr ⟨Nat.mod_lt _ (by decide),by omega⟩
      simpa only [index] using tile_values ops compute backward p valid head s ch on (i%128) active
    · have different : i/128≠head := fun equal => absent (equal ▸ tail)
      rw [run,ih uniqueRest _ tail]
      simp only [value,tile_other ops compute backward p valid head s _ i hi different]

#axiomsClean tile_values
#axiomsClean run_values
#axiomsClean run_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.GegluTanhTriton.Physical
