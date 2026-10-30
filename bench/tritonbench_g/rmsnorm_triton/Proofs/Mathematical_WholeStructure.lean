import bench.tritonbench_g.rmsnorm_triton.Sources.Mathematical_WholeSource
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 10000

structure Separation (p : Parameters) : Prop where
  input : p.input.allocation ≠ p.output.allocation
  weights : p.weights.allocation ≠ p.output.allocation

/-- Ordinary non-overlap of the fresh empty_like result, including arbitrary
permutation-preserving strides. This places no restriction on input aliases. -/
def OutputUnique (p : Parameters) : Prop :=
  ∀ b m k b' m' k', b < p.batches → m < p.rows → k < p.width →
    b' < p.batches → m' < p.rows → k' < p.width →
    outputOffset p b m k = outputOffset p b' m' k' → b=b' ∧ m=m' ∧ k=k'

def RowOutside (p : Parameters) (b m : Nat) (ks : List Nat) (a i : Nat) : Prop :=
  ∀ k ∈ ks, k < p.width → ¬ (a=p.output.allocation ∧ i=outputOffset p b m k)

def Outside (p : Parameters) (a i : Nat) : Prop :=
  ∀ b m k, b < p.batches → m < p.rows → k < p.width →
    ¬ (a=p.output.allocation ∧ i=outputOffset p b m k)

@[simp] theorem store_same (memory : Memory α) (a i : Nat) (v : α) :
    store memory a i v a i = v := by simp [store]

theorem store_other (memory : Memory α) (a i a' i' : Nat) (v : α)
    (h : ¬ (a'=a ∧ i'=i)) : store memory a i v a' i' = memory a' i' := by
  simp only [store,if_neg h]

theorem outputLane_frame (Q : Numerics α) (p : Parameters) (b m k : Nat)
    (std : α) (u : Nat → α) (memory : Memory α) (a i : Nat)
    (h : k < p.width → ¬ (a=p.output.allocation ∧ i=outputOffset p b m k)) :
    outputLane Q p b m k std u memory a i = memory a i := by
  by_cases active : k < p.width
  · simp only [outputLane,active,if_true]
    exact store_other _ _ _ _ _ _ (h active)
  · simp only [outputLane,active,if_false]

theorem outputLane_readonly (Q : Numerics α) (p : Parameters) (b m k : Nat)
    (std : α) (u : Nat → α) (memory : Memory α) (a : Nat)
    (h : a ≠ p.output.allocation) :
    outputLane Q p b m k std u memory a = memory a := by
  funext i
  exact outputLane_frame Q p b m k std u memory a i (fun _ same => h same.1)

theorem outputPass_frame (Q : Numerics α) (p : Parameters) (b m : Nat)
    (std : α) (u : Nat → α) (ks : List Nat) (memory : Memory α) (a i : Nat)
    (h : RowOutside p b m ks a i) :
    outputPass Q p b m std u ks memory a i = memory a i := by
  induction ks generalizing memory with
  | nil => rfl
  | cons k ks ih =>
    rw [outputPass,ih _ (fun j hj => h j (List.mem_cons_of_mem k hj))]
    exact outputLane_frame Q p b m k std u memory a i (h k List.mem_cons_self)

theorem outputPass_readonly (Q : Numerics α) (p : Parameters) (b m : Nat)
    (std : α) (u : Nat → α) (ks : List Nat) (memory : Memory α) (a : Nat)
    (h : a ≠ p.output.allocation) :
    outputPass Q p b m std u ks memory a = memory a := by
  funext i
  exact outputPass_frame Q p b m std u ks memory a i (fun _ _ _ same => h same.1)

theorem outputPass_value (Q : Numerics α) (p : Parameters) (sep : Separation p)
    (unique : OutputUnique p) (b m : Nat) (hb : b < p.batches) (hm : m < p.rows)
    (std : α) (u : Nat → α) (ks : List Nat) (memory : Memory α)
    (k : Nat) (hk : k < p.width) (member : k ∈ ks) :
    outputPass Q p b m std u ks memory p.output.allocation (outputOffset p b m k) =
      Q.mul (Q.mul std (memory p.input.allocation (inputOffset p b m k)))
        (memory p.weights.allocation (weightOffset p k)) := by
  induction ks generalizing memory with
  | nil => simp at member
  | cons j ks ih =>
    rw [outputPass]
    by_cases tail : k ∈ ks
    · rw [ih _ tail,
        outputLane_readonly Q p b m j std u memory _ sep.input,
        outputLane_readonly Q p b m j std u memory _ sep.weights]
    · have equal : k=j := (List.mem_cons.mp member).resolve_right tail
      subst j
      rw [outputPass_frame]
      · simp only [outputLane,hk,if_true,store_same]
      · intro j hj active same
        have equal := (unique b m k b m j hb hm hk hb hm active same.2).2.2
        exact tail (equal ▸ hj)

theorem columns_mem (p : Parameters) (k : Nat) : k ∈ columns p ↔ k < trips p*1024 := by
  simp only [columns,List.mem_flatMap,List.mem_range,List.mem_map]
  constructor
  · rintro ⟨t,ht,j,hj,rfl⟩
    omega
  · intro h
    refine ⟨k/1024,by omega,k%1024,Nat.mod_lt _ (by decide),?_⟩
    omega

theorem width_le_padded (p : Parameters) : p.width ≤ trips p*1024 := by
  unfold trips
  omega

theorem active_columns (p : Parameters) (k : Nat) (hk : k < p.width) : k ∈ columns p :=
  (columns_mem p k).mpr (Nat.lt_of_lt_of_le hk (width_le_padded p))

theorem programGrid_mem (p : Parameters) (b m : Nat) :
    (b,m) ∈ programGrid p ↔ b < p.batches ∧ m < p.rows := by
  simp [programGrid]

theorem variance_fold (Q : Numerics α) (red : Reduction) (p : Parameters) (b m n : Nat)
    (memory : Memory α) :
    (List.range n).foldl (fun acc t => varianceStep Q red p b m t memory acc)
      (fun _ => Q.literal 0) =
    laneAccumulator Q red.initialRightZero p.width
      (fun k => memory p.input.allocation (inputOffset p b m k)) n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [List.range_succ,List.foldl_append]
    simp only [List.foldl_cons,List.foldl_nil,ih]
    rfl

theorem executeRow_eq (Q : Numerics α) (red : Reduction) (p : Parameters)
    (b m : Nat) (u : Nat → α) (memory : Memory α) :
    executeRow Q red p b m u memory = outputPass Q p b m
      (logicalStd Q red p.width p.eps
        (fun k => memory p.input.allocation (inputOffset p b m k))) u (columns p) memory := by
  unfold executeRow
  rw [variance_fold]
  rfl

theorem row_frame (Q : Numerics α) (red : Reduction) (p : Parameters)
    (b m : Nat) (u : Nat → α) (memory : Memory α) (a i : Nat)
    (h : ∀ k, k < p.width → ¬ (a=p.output.allocation ∧ i=outputOffset p b m k)) :
    executeRow Q red p b m u memory a i = memory a i := by
  rw [executeRow_eq]
  exact outputPass_frame _ _ _ _ _ _ _ _ _ _ (fun k _ hk => h k hk)

theorem row_readonly (Q : Numerics α) (red : Reduction) (p : Parameters)
    (b m : Nat) (u : Nat → α) (memory : Memory α) (a : Nat)
    (h : a ≠ p.output.allocation) :
    executeRow Q red p b m u memory a = memory a := by
  funext i
  exact row_frame Q red p b m u memory a i (fun _ _ same => h same.1)

theorem row_value (Q : Numerics α) (red : Reduction) (p : Parameters)
    (sep : Separation p) (unique : OutputUnique p) (b m : Nat)
    (hb : b < p.batches) (hm : m < p.rows) (u : Nat → α) (memory : Memory α)
    (k : Nat) (hk : k < p.width) :
    executeRow Q red p b m u memory p.output.allocation (outputOffset p b m k) =
    logicalValue Q red p.width p.eps
      (fun j => memory p.input.allocation (inputOffset p b m j))
      (fun j => memory p.weights.allocation (weightOffset p j)) k := by
  rw [executeRow_eq]
  exact outputPass_value Q p sep unique b m hb hm _ u _ memory k hk (active_columns p k hk)

theorem grid_frame (Q : Numerics α) (red : Reduction) (p : Parameters)
    (u : Nat → α) (order : List (Nat × Nat)) (memory : Memory α) (a i : Nat)
    (outside : ∀ bm ∈ order, ∀ k, k < p.width →
      ¬ (a=p.output.allocation ∧ i=outputOffset p bm.1 bm.2 k)) :
    executeGrid Q red p u order memory a i = memory a i := by
  induction order generalizing memory with
  | nil => rfl
  | cons bm rest ih =>
    rcases bm with ⟨b,m⟩
    rw [executeGrid,ih _ (fun bm h => outside bm (List.mem_cons_of_mem _ h))]
    exact row_frame Q red p b m u memory a i (outside (b,m) List.mem_cons_self)

theorem grid_value (Q : Numerics α) (red : Reduction) (p : Parameters)
    (sep : Separation p) (unique : OutputUnique p) (u : Nat → α)
    (order : List (Nat × Nat)) (valid : ∀ bm ∈ order, bm.1 < p.batches ∧ bm.2 < p.rows)
    (memory : Memory α) (b m k : Nat) (member : (b,m) ∈ order) (hk : k < p.width) :
    executeGrid Q red p u order memory p.output.allocation (outputOffset p b m k) =
      logicalValue Q red p.width p.eps
        (fun j => memory p.input.allocation (inputOffset p b m j))
        (fun j => memory p.weights.allocation (weightOffset p j)) k := by
  induction order generalizing memory with
  | nil => simp at member
  | cons bm rest ih =>
    rcases bm with ⟨b',m'⟩
    rw [executeGrid]
    have target := valid (b,m) member
    by_cases tail : (b,m) ∈ rest
    · rw [ih (fun bm h => valid bm (List.mem_cons_of_mem _ h)) _ tail,
        row_readonly Q red p b' m' u memory _ sep.input,
        row_readonly Q red p b' m' u memory _ sep.weights]
    · have equal : (b,m)=(b',m') := (List.mem_cons.mp member).resolve_right tail
      cases equal
      rw [grid_frame]
      · exact row_value Q red p sep unique b m target.1 target.2 u memory k hk
      · intro bm hbm j hj same
        have h := valid bm (List.mem_cons_of_mem _ hbm)
        have equal := unique b m k bm.1 bm.2 j target.1 target.2 hk h.1 h.2 hj same.2
        have pair : bm=(b,m) := Prod.ext equal.1.symm equal.2.1.symm
        exact tail (pair ▸ hbm)

/-- Complete grid, arbitrary block order, and every output cell's value plus the
entire physical-memory frame. No primitive arithmetic law is used. -/
theorem grid_result (Q : Numerics α) (red : Reduction) (p : Parameters)
    (sep : Separation p) (unique : OutputUnique p) (u : Nat → α)
    (order : List (Nat × Nat)) (schedule : order.Perm (programGrid p)) (memory : Memory α) :
    (∀ b m k, b < p.batches → m < p.rows → k < p.width →
      executeGrid Q red p u order memory p.output.allocation (outputOffset p b m k) =
        logicalValue Q red p.width p.eps
          (fun j => memory p.input.allocation (inputOffset p b m j))
          (fun j => memory p.weights.allocation (weightOffset p j)) k) ∧
    (∀ a i, Outside p a i → executeGrid Q red p u order memory a i = memory a i) := by
  have valid : ∀ bm ∈ order, bm.1 < p.batches ∧ bm.2 < p.rows := by
    intro bm hbm
    exact (programGrid_mem p bm.1 bm.2).mp (schedule.mem_iff.mp hbm)
  constructor
  · intro b m k hb hm hk
    exact grid_value Q red p sep unique u order valid memory b m k
      (schedule.mem_iff.mpr ((programGrid_mem p b m).mpr ⟨hb,hm⟩)) hk
  · intro a i outside
    exact grid_frame Q red p u order memory a i
      (fun bm hbm k hk => outside bm.1 bm.2 k (valid bm hbm).1 (valid bm hbm).2 hk)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
