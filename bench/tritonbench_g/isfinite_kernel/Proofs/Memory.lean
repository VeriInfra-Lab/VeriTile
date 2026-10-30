import bench.tritonbench_g.isfinite_kernel.Proofs.Contract
import bench.tritonbench_g.f8_conversion_utils.Proofs.Chunks

namespace VeriTile.Bench.Repairs.IsfiniteKernel.Memory
noncomputable section
set_option maxHeartbeats 4000000

def boolByte (b : Bool) : BitVec 8 := if b then 1 else 0

/-- Ordinary single-byte store and disjoint typed-load laws. There is no
numerical classifier or whole-kernel conclusion in this interface. -/
structure Primitives (M A : Type) (width : Nat) where
  read : M → Nat → A
  byte : M → Nat → BitVec 8
  write : M → Nat → Bool → M
  write_byte : ∀ s a b q,byte (write s a b) q = if q=a then boolByte b else byte s q
  input_frame : ∀ s a b q,(∀ k,k<width → q+k≠a) → read (write s a b) q=read s q

variable {M A : Type} {width : Nat} (ops : Primitives M A width)

def stores (output : Nat → Nat) (values : Nat → Bool) : List Nat → M → M
  | [],s => s
  | i::rest,s => stores output values rest (ops.write s (output i) (values i))

theorem stores_append (output : Nat → Nat) (values : Nat → Bool) (xs ys : List Nat) (s : M) :
    stores ops output values (xs++ys) s=stores ops output values ys (stores ops output values xs s) := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih => exact ih _

theorem stores_value (output : Nat → Nat) (values : Nat → Bool) (n : Nat)
    (injective : ∀ i,i<n → ∀ j,j<n → output i=output j → i=j)
    (jobs : List Nat) (active : ∀ i∈jobs,i<n) (s : M) (i : Nat) (hi : i<n) :
    ops.byte (stores ops output values jobs s) (output i)=
      if i∈jobs then boolByte (values i) else ops.byte s (output i) := by
  induction jobs generalizing s with
  | nil => simp [stores]
  | cons j rest ih =>
    rw [stores,ih (fun k hk => active k (by simp [hk]))]
    by_cases member : i∈rest
    · simp [member]
    · have eq_iff : output i=output j ↔ i=j := ⟨injective i hi j (active j (by simp)),congrArg output⟩
      rw [ops.write_byte]
      by_cases same : i=j
      · subst j; simp [member]
      · simp [member,same,eq_iff]

theorem stores_input (output : Nat → Nat) (values : Nat → Bool) (jobs : List Nat) (s : M)
    (q : Nat) (apart : ∀ j∈jobs,∀ k,k<width → q+k≠output j) :
    ops.read (stores ops output values jobs s) q=ops.read s q := by
  induction jobs generalizing s with
  | nil => rfl
  | cons j rest ih =>
    rw [stores,ih _ (fun k hk => apart k (by simp [hk]))]
    exact ops.input_frame _ _ _ _ (apart j (by simp))

theorem stores_frame (output : Nat → Nat) (values : Nat → Bool) (jobs : List Nat) (s : M)
    (q : Nat) (outside : ∀ j∈jobs,q≠output j) :
    ops.byte (stores ops output values jobs s) q=ops.byte s q := by
  induction jobs generalizing s with
  | nil => rfl
  | cons j rest ih =>
    rw [stores,ih _ (fun k hk => outside k (by simp [hk])),ops.write_byte,
      if_neg (outside j (by simp))]

def run (classify : A → Bool) (input output : Nat → Nat) : List (List Nat) → M → M
  | [],s => s
  | jobs::rest,s =>
    let values := fun i => classify (ops.read s (input i))
    run classify input output rest (stores ops output values jobs s)

theorem buffered (classify : A → Bool) (input output : Nat → Nat) (n : Nat)
    (groups : List (List Nat)) (original current : M)
    (active : ∀ i∈groups.flatten,i<n)
    (apart : ∀ i,i<n → ∀ k,k<width → ∀ j,j<n → input i+k≠output j)
    (preserved : ∀ i,i<n → ops.read current (input i)=ops.read original (input i)) :
    run ops classify input output groups current=
      stores ops output (fun i => classify (ops.read original (input i))) groups.flatten current := by
  induction groups generalizing current with
  | nil => rfl
  | cons jobs rest ih =>
    have hj (i : Nat) (hi : i∈jobs) : i<n := active i (by simp [hi])
    have values : stores ops output (fun i => classify (ops.read current (input i))) jobs current=
        stores ops output (fun i => classify (ops.read original (input i))) jobs current := by
      have congrStores (xs : List Nat) (ha : ∀ i∈xs,i<n) (s : M) :
          stores ops output (fun i => classify (ops.read current (input i))) xs s=
          stores ops output (fun i => classify (ops.read original (input i))) xs s := by
        induction xs generalizing s with
        | nil => rfl
        | cons i xs ih =>
          simp only [stores,preserved i (ha i (by simp))]
          exact ih (fun j h => ha j (by simp [h])) _
      exact congrStores _ hj _
    rw [run,values,ih]
    · simp only [List.flatten_cons,stores_append]
    · intro i hi; exact active i (by simp [hi])
    · intro i hi
      rw [stores_input ops output _ jobs _ (input i) (fun j hj' k hk => apart i hi k hk j (hj j hj'))]
      exact preserved i hi

theorem complete (classify : A → Bool) (input output : Nat → Nat) (n : Nat)
    (groups : List (List Nat)) (original : M)
    (coverage : ∀ i,i∈groups.flatten ↔ i<n)
    (injective : ∀ i,i<n → ∀ j,j<n → output i=output j → i=j)
    (apart : ∀ i,i<n → ∀ k,k<width → ∀ j,j<n → input i+k≠output j) :
    (∀ i,i<n → ops.byte (run ops classify input output groups original) (output i)=
      boolByte (classify (ops.read original (input i)))) ∧
    (∀ q,(∀ i,i<n → q≠output i) →
      ops.byte (run ops classify input output groups original) q=ops.byte original q) ∧
    (∀ i,i<n → ops.read (run ops classify input output groups original) (input i)=ops.read original (input i)) := by
  rw [buffered ops classify input output n groups original original
    (fun i hi => (coverage i).mp hi) apart (fun _ _ => rfl)]
  refine ⟨?_,?_,?_⟩
  · intro i hi
    rw [stores_value ops output _ n injective _ (fun i hi => (coverage i).mp hi) _ i hi,
      if_pos ((coverage i).mpr hi)]
  · intro q outside
    exact stores_frame ops output _ _ _ q (fun i hi => outside i ((coverage i).mp hi))
  · intro i hi
    exact stores_input ops output _ _ _ _ (fun j hj k hk => apart i hi k hk j ((coverage j).mp hj))

#axiomsClean complete
end
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Repairs.IsfiniteKernel.Memory
