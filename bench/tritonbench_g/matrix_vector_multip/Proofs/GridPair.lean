import bench.tritonbench_g.matrix_vector_multip.Proofs.ExplicitFMA
import bench.tritonbench_g.matrix_vector_multip.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.GridPair
open VeriTile.Triton FP.Structural KernelSource
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeNative
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000

theorem run_equal (native : Scalars) (N M BN BM : Nat) (g : Graph BN BM)
    (width : 0<BM) (scope : SourceBounds N M BN BM) (order : List Nat) (small : ∀ pid ∈ order,pid<2^64) (s : State Payload) :
    runSchedule (order.map (ExplicitFMA.invocation native N M BN BM · g)) s =
      runSchedule (order.map (IO.invocation (WholeNative.algebra native g) N M BN BM)) s := by
  induction order generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    change (exec (ExplicitFMA.algebra native g) (Kernels.explicitFMA N M BN BM)
      ((ExplicitFMA.invocation native N M BN BM pid g).initial s)).bind _ = _
    rw [ExplicitFMA.execution_equal native g N M width scope _ (small pid (by simp))]
    change (exec (WholeNative.algebra native g) (Kernels.source N M BN BM)
      ((IO.invocation (WholeNative.algebra native g) N M BN BM pid).initial s)).bind _ = _
    congr 1
    funext next
    exact ih (fun pid member => small pid (by simp [member])) next

theorem executes (native : Scalars) (N M BN BM : Nat) (g : Graph BN BM)
    (width : 0<BM) (scope : SourceBounds N M BN BM) (order : List Nat) (small : ∀ pid ∈ order,pid<2^64) (s : State Payload) :
    ∃ final,
      runSchedule (order.map (IO.invocation (WholeNative.algebra native g) N M BN BM)) s=some final ∧
      ∀ (r : RegionName) (i : Nat), (r ≠ "C" ∨ N ≤ i) → final.mem r i=s.mem r i := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl,fun _ _ _ => rfl⟩
  | cons pid rest ih =>
    let entry := (IO.invocation (WholeNative.algebra native g) N M BN BM pid).initial s
    obtain ⟨next,ran,_,frame⟩ := KernelSource.execution (WholeNative.algebra native g) N M BN BM width scope (by intros; rfl) (by intros; rfl) entry (small pid (by simp))
    obtain ⟨final,restRun,restFrame⟩ := ih (fun pid member => small pid (by simp [member])) next
    refine ⟨final,?_,?_⟩
    · change (exec (WholeNative.algebra native g) (Kernels.source N M BN BM) entry).bind _=some final
      rw [ran]
      exact restRun
    · intro r i outside
      rw [restFrame r i outside]
      apply frame
      rcases outside with h | h
      · exact Or.inl h
      · right
        intro lane active equal
        unfold KernelSource.row at active equal
        omega

theorem safe (native : Scalars) (N M BN BM : Nat) (g : Graph BN BM)
    (width : 0<BM) (scope : SourceBounds N M BN BM) (bounds : RegionName → Nat)
    (matrix : ∀ i j, i<N → j<M → i*M+j<bounds "A")
    (vector : ∀ j, j<M → j<bounds "B") (output : ∀ i,i<N → i<bounds "C")
    (order : List Nat) (small : ∀ pid ∈ order,pid<2^64) (s : State Payload) :
    ScheduleSafe bounds (order.map (IO.invocation (WholeNative.algebra native g) N M BN BM)) s ∧
      ScheduleSafe bounds (order.map (ExplicitFMA.invocation native N M BN BM · g)) s := by
  induction order generalizing s with
  | nil => exact ⟨trivial,trivial⟩
  | cons pid rest ih =>
    constructor
    · refine ⟨?_,fun next _ => (ih (fun pid member => small pid (by simp [member])) next).1⟩
      apply AccessSafety.source_safe (WholeNative.algebra native g) N M BN BM width scope
        (by intros; rfl) (by intros; rfl) bounds _ (small pid (by simp))
      · simpa [parameters] using matrix
      · simpa [parameters] using vector
      · intro i active
        simpa [parameters] using output _ active
    · refine ⟨?_,fun next _ => (ih (fun pid member => small pid (by simp [member])) next).2⟩
      apply ExplicitFMA.source_safe native g N M width scope bounds _ (small pid (by simp))
      · exact matrix
      · exact vector
      · intro i active
        exact output _ active

theorem equivalent (native : Scalars) (N M BN BM : Nat) (g : Graph BN BM)
    (width : 0<BM) (scope : SourceBounds N M BN BM) (order : List Nat) (small : ∀ pid ∈ order,pid<2^64) :
    (io native N M BN BM g order).Equivalent (variant native N M BN BM g order) := by
  intro vals _ bounds covered initial pins
  obtain ⟨leftSafe,rightSafe⟩ := safe native N M BN BM g width scope bounds
    (fun i j hi hj => covered.2.1 (0 : Fin 2) (⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial)
    (fun j hj => covered.2.1 (1 : Fin 2) (⟨j,hj⟩,PUnit.unit) trivial)
    (fun i hi => covered.2.2.1 (0 : Fin 1) (⟨i,hi⟩,PUnit.unit) trivial) order small initial
  obtain ⟨final,ran,frame⟩ := executes native N M BN BM g width scope order small initial
  refine ⟨leftSafe,rightSafe,final,final,ran,?_,?_,?_,?_⟩
  · change runSchedule (order.map (ExplicitFMA.invocation native N M BN BM · g)) initial=some final
    rw [run_equal native N M BN BM g width scope order small]
    exact ran
  · intro o j active
    rfl
  all_goals
    intro r i outside
    apply frame r i
    by_cases same : r="C"
    · right
      by_contra h
      have small : i<N := by omega
      have no := outside.1 (0 : Fin 1) (⟨i,small⟩,PUnit.unit) trivial
      exact no.elim (fun h => h same) (fun h => h rfl)
    · exact Or.inl same

#axiomsClean run_equal
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.GridPair
