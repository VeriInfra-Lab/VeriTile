import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Source
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Contract

namespace VeriTile.Bench.Repairs.IsfiniteKernel.Kernel
open VeriTile.Triton FP.Structural
noncomputable section

/-- Independent pointwise classifier and exact unwritten-cell frame for one
kernel program. No kernel execution appears inside the mathematical formula. -/
def Post {A : Type} [Inhabited A] (classify : A → Bool)
    (input output : View) (n B : Nat) (before after : State A) : Prop :=
  (∀ lane : Fin B, before.pids 0 * B + lane.val < n →
    after.mem "y" (output.address (before.pids 0 * B + lane.val)) =
      .mk .bool (classify ((before.mem "x"
        (input.address (before.pids 0 * B + lane.val))).read .real))) ∧
  (∀ r q, (r ≠ ("y" : RegionName) ∨ ∀ lane : Fin B,
      before.pids 0 * B + lane.val < n →
      q ≠ output.address (before.pids 0 * B + lane.val)) →
    after.mem r q = before.mem r q)

private theorem fold_preserves {A I : Type} (xs : List I)
    (active : I → Prop) [DecidablePred active] (offset : I → Nat) (value : I → Cell A)
    (s : State A) (q : Nat) (v : Cell A) (initial : s.mem "y" q = v)
    (same : ∀ i ∈ xs, active i → offset i = q → value i = v) :
    (xs.foldl (fun t i => if active i then t.write "y" (offset i) (value i) else t) s).mem "y" q = v := by
  induction xs generalizing s with
  | nil => exact initial
  | cons i xs ih =>
    apply ih _ ?_ (fun j hj => same j (by simp [hj]))
    by_cases ha : active i
    · by_cases hq : q = offset i
      · simp [ha,State.write,hq,same i (by simp) ha hq.symm]
      · simp [ha,State.write,hq,initial]
    · simpa [ha] using initial

private theorem fold_readback {A I : Type} (xs : List I)
    (active : I → Prop) [DecidablePred active] (offset : I → Nat) (value : I → Cell A)
    (s : State A) (q : Nat) (v : Cell A)
    (hit : ∃ i ∈ xs, active i ∧ offset i = q)
    (same : ∀ i ∈ xs, active i → offset i = q → value i = v) :
    (xs.foldl (fun t i => if active i then t.write "y" (offset i) (value i) else t) s).mem "y" q = v := by
  induction xs generalizing s with
  | nil => simp at hit
  | cons i xs ih =>
    by_cases head : active i ∧ offset i = q
    · apply fold_preserves xs active offset value _ q v
      · simp [head.1,State.write,head.2,same i (by simp) head.1 head.2]
      · exact fun j hj => same j (by simp [hj])
    · apply ih _ ?_ (fun j hj => same j (by simp [hj]))
      rcases hit with ⟨j,hj,ha,hq⟩
      rcases List.mem_cons.mp hj with rfl | hj
      · exact (head ⟨ha,hq⟩).elim
      · exact ⟨j,hj,ha,hq⟩

theorem stores_post {A : Type} [Inhabited A] (classify : A → Bool)
    (input output : View) (n B : Nat) (s : State A)
    (injective : ∀ i, i < n → ∀ j, j < n → output.address i = output.address j → i = j) :
    Post classify input output n B s (Source.stores classify input output n B (s.pids 0) s s) := by
  constructor
  · intro lane active
    apply fold_readback (TileShape.allIndices [B])
      (fun i => s.pids 0 * B + i.1.val < n)
      (fun i => output.address (s.pids 0 * B + i.1.val))
      (fun i => .mk .bool (classify ((s.mem "x"
        (input.address (s.pids 0 * B + i.1.val))).read .real)))
      s _ _
    · exact ⟨(lane,PUnit.unit),TileShape.mem_allIndices _ _,active,rfl⟩
    · intro i _ hi same
      rw [injective _ hi _ active same]
  · intro r q outside
    apply State.masked_scatter_frame
    exact outside.elim Or.inl (fun h => Or.inr (fun i _ => h i.1))

theorem shifted_address (v : View) (start i : Nat) :
    (v.shift start).address i = v.address (start+i) := by
  unfold View.address View.signedAddress View.shift
  congr 1
  push_cast
  ring

theorem output_addresses (x y : ViewMetadata) (h : InputContract x y)
    (start count : Nat) (inside : ∀ i, i < count → start+i < x.n) :
    ∀ i, i < count → ∀ j, j < count →
      (y.outputView.shift start).address i = (y.outputView.shift start).address j → i = j := by
  intro i hi j hj same
  rw [shifted_address,shifted_address] at same
  have := output_injective x y h _ _ (inside i hi) (inside j hj) same
  omega

theorem correct {A : Type} [Inhabited A] (classify : A → Bool)
    (x y : ViewMetadata) (h : InputContract x y) (start count B : Nat)
    (inside : ∀ i, i < count → start+i < x.n) (before : State A) :
    ∃ after, FP.Structural.exec (Source.algebra x.precision classify)
        (source x.precision (x.view.shift start) (y.outputView.shift start) count B) before = some after ∧
      Post classify (x.view.shift start) (y.outputView.shift start) count B before after := by
  obtain ⟨after,executed,memory⟩ := Source.executes x.precision classify
    (x.view.shift start) (y.outputView.shift start) count B before
  refine ⟨after,executed,?_⟩
  have hpost := stores_post classify (x.view.shift start) (y.outputView.shift start)
    count B before (output_addresses x y h start count inside)
  simpa only [Post,memory] using hpost

#axiomsClean correct
#axiomsClean stores_post
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IsfiniteKernel.Kernel
