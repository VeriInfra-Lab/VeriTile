import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Memory

namespace VeriTile.Bench.Repairs.SwigluTriton.Physical
open Memory
noncomputable section
open Classical
set_option maxHeartbeats 6000000

inductive Role where
  | a | b | third
  deriving DecidableEq

def outputRole : Channel → Role
  | .c => .third | .da => .a | .db => .b

structure Layout where
  allocation : Role → Allocation
  rows : Nat
  cols : Nat
  block : Nat

def Layout.key (p : Layout) (role : Role) (row col : Nat) : Key :=
  (p.allocation role).key (row*p.block+col)

structure Valid (p : Layout) : Prop where
  cols : p.cols≤p.block
  capacity : ∀ role,p.rows*p.block≤(p.allocation role).words
  separate : ∀ a b,a≠b → (p.allocation a).separate (p.allocation b)

theorem index_bound (rows block row col : Nat) (hr : row<rows) (hc : col<block) :
    row*block+col<rows*block := by
  have h := Nat.mul_le_mul_right block (show row+1≤rows by omega)
  simp only [Nat.add_mul,Nat.one_mul] at h
  omega

theorem key_backed (p : Layout) (valid : Valid p) (role : Role) (row col : Nat)
    (hr : row<p.rows) (hc : col<p.block) : (p.allocation role).contains (p.key role row col) :=
  Allocation.key_contains _ _ (lt_of_lt_of_le (index_bound _ _ _ _ hr hc) (valid.capacity role))

theorem keys_unique (p : Layout) (valid : Valid p) (a b : Role) (r c r' c' : Nat)
    (hr : r<p.rows) (hc : c<p.block) (hr' : r'<p.rows) (hc' : c'<p.block)
    (equal : p.key a r c=p.key b r' c') : a=b ∧ r=r' ∧ c=c' := by
  by_cases same : a=b
  · subst b
    have words := congrArg Key.address equal
    simp only [Layout.key,Allocation.key] at words
    exact ⟨rfl,Grid.row_unique p.block r c r' c' hc hc' (by omega)⟩
  · exact False.elim ((Allocation.keys_separate _ _ (valid.separate a b same) _ _
      (key_backed p valid a r c hr hc) (key_backed p valid b r' c' hr' hc')) equal)

theorem keys_disjoint (p : Layout) (valid : Valid p) (a b : Role) (r c r' c' : Nat)
    (hr : r<p.rows) (hc : c<p.block) (hr' : r'<p.rows) (hc' : c'<p.block)
    (different : p.key a r c≠p.key b r' c') : (p.key a r c).disjoint (p.key b r' c') := by
  by_cases same : a=b
  · subst b
    apply Allocation.indices_disjoint
    intro equal
    exact different (congrArg (p.allocation a).key equal)
  · exact Allocation.spans_separate _ _ (valid.separate a b same) _ _
      (key_backed p valid a r c hr hc) (key_backed p valid b r' c' hr' hc')

def value {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : Channel → α → α → α → α) (backward : Bool) (p : Layout)
    (s : M) (ch : Channel) (row col : Nat) : α :=
  compute ch (ops.read s (p.key .a row col)) (ops.read s (p.key .b row col))
    (if backward then ops.read s (p.key .third row col) else default)

/-- A source row loads all operands before either store. The second scatter
therefore uses exactly the same entry values as the first, even in backward. -/
def row {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : Channel → α → α → α → α) (backward : Bool) (p : Layout) (r : Nat) (s : M) : M :=
  let scatter := fun ch current => stores ops (p.key (outputRole ch) r)
    (value ops compute backward p s ch r) (List.range p.cols) current
  if backward then scatter .db (scatter .da s) else scatter .c s

theorem row_frame {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : Channel → α → α → α → α) (backward : Bool) (p : Layout)
    (r : Nat) (s : M) (observed : Key)
    (outside : ∀ ch,enabled backward ch → ∀ col,col<p.cols → observed.disjoint (p.key (outputRole ch) r col)) :
    ops.read (row ops compute backward p r s) observed=ops.read s observed := by
  cases backward <;> simp only [row,if_false,if_true]
  · apply read_frame
    intro i hi; exact outside .c rfl i (List.mem_range.mp hi)
  · rw [read_frame ops _ _ _ _ observed (fun i hi => outside .db (by decide) i (List.mem_range.mp hi))]
    apply read_frame
    intro i hi; exact outside .da (by decide) i (List.mem_range.mp hi)

theorem row_bytes {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : Channel → α → α → α → α) (backward : Bool) (p : Layout)
    (r : Nat) (s : M) (owner address : Nat)
    (outside : ∀ ch,enabled backward ch → ∀ col,col<p.cols → ¬(p.key (outputRole ch) r col).covers owner address) :
    ops.observe (row ops compute backward p r s) owner address=ops.observe s owner address := by
  cases backward <;> simp only [row,if_false,if_true]
  · apply bytes_frame
    intro i hi; exact outside .c rfl i (List.mem_range.mp hi)
  · rw [bytes_frame ops _ _ _ _ owner address (fun i hi => outside .db (by decide) i (List.mem_range.mp hi))]
    apply bytes_frame
    intro i hi; exact outside .da (by decide) i (List.mem_range.mp hi)

theorem row_values {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : Channel → α → α → α → α) (backward : Bool) (p : Layout) (valid : Valid p)
    (r : Nat) (s : M) (hr : r<p.rows) (ch : Channel) (on : enabled backward ch)
    (col : Nat) (active : col<p.cols) :
    ops.read (row ops compute backward p r s) (p.key (outputRole ch) r col)=
      value ops compute backward p s ch r col := by
  have injective (role : Role) (j : Nat) (hj : j∈List.range p.cols)
      (equal : p.key role r j=p.key role r col) : j=col :=
    (keys_unique p valid role role r j r col hr
      (lt_of_lt_of_le (List.mem_range.mp hj) valid.cols) hr (lt_of_lt_of_le active valid.cols) equal).2.2
  cases backward <;> cases ch <;> simp only [row,if_false,if_true,outputRole]
  · exact readback ops _ _ _ _ _ (List.mem_range.mpr active)
      (fun j hj h => by rw [injective .third j hj h])
      (fun j hj h => keys_disjoint p valid .third .third r col r j hr
        (lt_of_lt_of_le active valid.cols) hr (lt_of_lt_of_le (List.mem_range.mp hj) valid.cols) (Ne.symm h))
  · simp [enabled] at on
  · simp [enabled] at on
  · simp [enabled] at on
  · rw [read_frame ops _ _ _ _ _ (fun j hj =>
      Allocation.spans_separate _ _ (valid.separate .a .b (by decide)) _ _
        (key_backed p valid .a r col hr (lt_of_lt_of_le active valid.cols))
        (key_backed p valid .b r j hr (lt_of_lt_of_le (List.mem_range.mp hj) valid.cols)))]
    exact readback ops _ _ _ _ _ (List.mem_range.mpr active)
      (fun j hj h => by rw [injective .a j hj h])
      (fun j hj h => keys_disjoint p valid .a .a r col r j hr
        (lt_of_lt_of_le active valid.cols) hr (lt_of_lt_of_le (List.mem_range.mp hj) valid.cols) (Ne.symm h))
  · exact readback ops _ _ _ _ _ (List.mem_range.mpr active)
      (fun j hj h => by rw [injective .b j hj h])
      (fun j hj h => keys_disjoint p valid .b .b r col r j hr
        (lt_of_lt_of_le active valid.cols) hr (lt_of_lt_of_le (List.mem_range.mp hj) valid.cols) (Ne.symm h))

def run {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : Channel → α → α → α → α) (backward : Bool) (p : Layout) : List Nat → M → M
  | [],s => s
  | r::rest,s => run ops compute backward p rest (row ops compute backward p r s)

theorem run_frame {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : Channel → α → α → α → α) (backward : Bool) (p : Layout)
    (rows : List Nat) (s : M) (observed : Key)
    (outside : ∀ r∈rows,∀ ch,enabled backward ch → ∀ col,col<p.cols →
      observed.disjoint (p.key (outputRole ch) r col)) :
    ops.read (run ops compute backward p rows s) observed=ops.read s observed := by
  induction rows generalizing s with
  | nil => rfl
  | cons r rest ih =>
    rw [run,ih _ (fun j hj => outside j (by simp [hj]))]
    exact row_frame ops compute backward p r s observed (outside r (by simp))

theorem run_bytes {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : Channel → α → α → α → α) (backward : Bool) (p : Layout)
    (rows : List Nat) (s : M) (owner address : Nat)
    (outside : ∀ r∈rows,∀ ch,enabled backward ch → ∀ col,col<p.cols →
      ¬(p.key (outputRole ch) r col).covers owner address) :
    ops.observe (run ops compute backward p rows s) owner address=ops.observe s owner address := by
  induction rows generalizing s with
  | nil => rfl
  | cons r rest ih =>
    rw [run,ih _ (fun j hj => outside j (by simp [hj]))]
    exact row_bytes ops compute backward p r s owner address (outside r (by simp))

theorem run_values {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : Channel → α → α → α → α) (backward : Bool) (p : Layout) (valid : Valid p)
    (rows : List Nat) (unique : rows.Nodup) (bounded : ∀ r∈rows,r<p.rows)
    (s : M) (ch : Channel) (on : enabled backward ch) (r col : Nat)
    (member : r∈rows) (active : col<p.cols) :
    ops.read (run ops compute backward p rows s) (p.key (outputRole ch) r col)=
      value ops compute backward p s ch r col := by
  induction rows generalizing s with
  | nil => simp at member
  | cons head rest ih =>
    obtain ⟨notMember,uniqueRest⟩ := List.nodup_cons.mp unique
    have hr := bounded r member
    have hh := bounded head (by simp)
    have hc := lt_of_lt_of_le active valid.cols
    rcases List.mem_cons.mp member with same | tail
    · subst r
      have outside : ∀ r∈rest,∀ other,enabled backward other → ∀ c,c<p.cols →
          (p.key (outputRole ch) head col).disjoint (p.key (outputRole other) r c) := by
        intro r hr' other otherOn c hc'
        apply keys_disjoint p valid (outputRole ch) (outputRole other) head col r c hr hc
          (bounded r (by simp [hr'])) (lt_of_lt_of_le hc' valid.cols)
        intro equal
        have equalRow := (keys_unique p valid (outputRole ch) (outputRole other)
          head col r c hr hc (bounded r (by simp [hr'])) (lt_of_lt_of_le hc' valid.cols) equal).2.1
        exact notMember (equalRow ▸ hr')
      rw [run,run_frame ops compute backward p rest _ _ outside]
      exact row_values ops compute backward p valid head s hh ch on col active
    · have different : r≠head := fun equal => notMember (equal ▸ tail)
      rw [run,ih uniqueRest (fun r hr' => bounded r (by simp [hr'])) _ tail]
      have keep (role : Role) :
          ops.read (row ops compute backward p head s) (p.key role r col)=ops.read s (p.key role r col) := by
        apply row_frame
        intro other otherOn c hc'
        apply keys_disjoint p valid role (outputRole other) r col head c hr hc hh
          (lt_of_lt_of_le hc' valid.cols)
        intro equal
        exact different (keys_unique p valid role (outputRole other) r col head c hr hc hh
          (lt_of_lt_of_le hc' valid.cols) equal).2.1
      simp only [value,keep]

#axiomsClean row_values
#axiomsClean run_values
#axiomsClean run_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton.Physical
