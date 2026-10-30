import bench.tritonbench_g.f8_conversion_utils.Proofs.Memory

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Memory
noncomputable section
variable {M A B C : Type} {iw ow : Nat} (ops : Primitives M A B iw ow)

/-- An observation may be a raw byte or an independent Real cell. Only a
single-store frame law is trusted, never an execution or result law. -/
structure Observer (C : Type) where
  observe : M → Nat → C
  write_frame : ∀ s a v q,q<a ∨ a+ow≤q → observe (ops.writeOutput s a v) q=observe s q

def Frame (obs : Observer ops C) (owned : Nat → Prop) (after before : M) : Prop :=
  ∀ q,owned q → obs.observe after q=obs.observe before q

theorem observer_stores (obs : Observer ops C) (base : Nat) (values : Nat → B)
    (jobs : List Nat) (before : M) (q : Nat)
    (outside : ∀ i∈jobs,q<address base ow i ∨ address base ow i+ow≤q) :
    obs.observe (stores ops base values jobs before) q=obs.observe before q := by
  induction jobs generalizing before with
  | nil => rfl
  | cons i rest ih =>
    rw [stores,ih _ (fun j hj => outside j (by simp [hj]))]
    exact obs.write_frame _ _ _ _ (outside i (by simp))

theorem observer_run (obs : Observer ops C) (twice : Bool) (compute : A → B)
    (input output : Nat) (groups : List (List Nat)) (before : M) (q : Nat)
    (outside : ∀ i∈groups.flatten,q<address output ow i ∨ address output ow i+ow≤q) :
    obs.observe (run ops twice compute input output groups before) q=obs.observe before q := by
  induction groups generalizing before with
  | nil => rfl
  | cons jobs rest ih =>
    rw [run,ih _ (fun i hi => outside i (by simp [hi]))]
    apply observer_stores
    intro i hi
    exact outside i (by simp only [List.flatten_cons,List.mem_append];exact Or.inl ((mem_writes twice jobs i).mp hi))

theorem run_append (twice : Bool) (compute : A → B) (input output : Nat)
    (xs ys : List (List Nat)) (before : M) :
    run ops twice compute input output (xs++ys) before=
      run ops twice compute input output ys (run ops twice compute input output xs before) := by
  induction xs generalizing before with
  | nil => rfl
  | cons jobs rest ih => exact ih _

theorem address_shift (base width start i : Nat) :
    address (base+start*width) width i=address base width (start+i) := by
  unfold address
  ring

theorem stores_shift (base start : Nat) (values : Nat → B) (jobs : List Nat) (before : M) :
    stores ops (base+start*ow) (fun i => values (start+i)) jobs before=
      stores ops base values (jobs.map (start+·)) before := by
  induction jobs generalizing before with
  | nil => rfl
  | cons i rest ih =>
    simp only [stores,List.map_cons,address_shift,ih]

@[simp] theorem writes_map (twice : Bool) (start : Nat) (jobs : List Nat) :
    writes twice (jobs.map (start+·))=(writes twice jobs).map (start+·) := by
  cases twice <;> simp [writes]

theorem run_shift (twice : Bool) (compute : A → B) (input output start : Nat)
    (groups : List (List Nat)) (before : M) :
    run ops twice compute (input+start*iw) (output+start*ow) groups before=
      run ops twice compute input output (groups.map (fun jobs => jobs.map (start+·))) before := by
  induction groups generalizing before with
  | nil => rfl
  | cons jobs rest ih =>
    simp only [run,List.map_cons,ih,address_shift,writes_map]
    rw [stores_shift ops output start (fun i => compute (ops.readInput before (address input iw i))) (writes twice jobs) before]

/-- The empty wrapper attempts setup once but launches no program. -/
theorem launch_batches (n : Nat) :
    (launchChunks n).flatMap (chunkBatches n)=batches n := by rfl

#axiomsClean observer_run
#axiomsClean run_shift
#axiomsClean launch_batches
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.F8ConversionUtils.Memory
