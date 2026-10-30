import bench.tritonbench_g.quantize_global.IO
import bench.tritonbench_g.quantize_global.Proofs.KernelSafety

namespace QuantizeGlobalRepair.IOCorrect
open VeriTile.Triton FP.Structural QuantizeIO
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
variable {α : Type} [Inhabited α]

theorem invocation_run (rest : Algebra α) (a : Scalar α) (p : Parameters)
    (v : QuantizeGlobalRepair.Invocation) (s : State α) :
    (invocation rest a p v).run s = executeProgram rest a p v s := rfl

theorem schedule_run (rest : Algebra α) (a : Scalar α) (p : Parameters)
    (vs : List QuantizeGlobalRepair.Invocation) (s : State α) :
    runIntegerSchedule (vs.map (invocation rest a p)) s = execute rest a p vs s := by
  induction vs generalizing s with
  | nil => rfl
  | cons v vs ih => simp only [List.map_cons, runIntegerSchedule, invocation_run, execute, ih]

theorem schedule_safe (rest : Algebra α) (a : Scalar α) (p : Parameters)
    (vs : List QuantizeGlobalRepair.Invocation) (bounds : RegionName → Nat)
    (hx : ∀ i, i < p.n → i < bounds p.X) (hm : 0 < bounds p.Maximum)
    (ho : ∀ i, i < p.n → i < bounds p.Out) (s : State α) :
    IntegerScheduleSafe bounds (vs.map (invocation rest a p)) s := by
  induction vs generalizing s with
  | nil => trivial
  | cons v vs ih =>
    constructor
    · exact KernelSafety.source_safe rest a _ p.X p.Maximum p.Out p.n v.base v.config.block
        bounds hx hm ho
    · intro t _
      exact ih t

theorem safe (rest : Algebra α) (a : Scalar α) (p : Parameters)
    (trials : Nat → List Config) (vals : (io rest a p trials).Inputs)
    (bounds : RegionName → Nat) (inside : (io rest a p trials).InBounds bounds vals)
    (s : State α) :
    IntegerScheduleSafe bounds ((io rest a p trials).schedule vals) s := by
  apply schedule_safe rest a p _ bounds
  · intro i hi
    exact inside.2.1 (0 : Fin 2) (⟨i,hi⟩,()) trivial
  · exact inside.2.1 (1 : Fin 2) () trivial
  · intro i hi
    exact inside.2.2.1 (0 : Fin 1) (⟨i,hi⟩,()) trivial

/-- The old complete-grid execution result is connected to logical inputs,
all signed integer outputs and the shared exact exterior frame. -/
theorem result (rest : Algebra α) (a : Scalar α) (p : Parameters) (sep : p.Separated)
    (trials : Nat → List Config) (nonempty : ∀ k, k < segmentCount p.n → trials k ≠ [])
    (f : (TileIndex [p.n] → α) → α → TileIndex [p.n] → Int)
    (vals : (io rest a p trials).Inputs)
    (equation : ∀ i, (scalar a (vals (0 : Fin 2) i) (vals (1 : Fin 2) ())).toInt =
      f (vals (0 : Fin 2)) (vals (1 : Fin 2) ()) i)
    (s : State α) (binding : (io rest a p trials).InputBinding vals s) :
    ∃ final, runIntegerSchedule ((io rest a p trials).schedule vals) s = some final ∧
      (io rest a p trials).OutputBinding vals (target rest a p trials f vals) final ∧
      (io rest a p trials).Frame vals s final := by
  let xs : Nat → α := fun i => if h : i < p.n then vals (0 : Fin 2) (⟨i,h⟩,()) else default
  have inputs : Inputs p xs (vals (1 : Fin 2) ()) s := by
    constructor
    · intro i hi
      simpa only [xs, dif_pos hi] using binding (0 : Fin 2) (⟨i,hi⟩,()) trivial
    · exact binding (1 : Fin 2) () trivial
  obtain ⟨final, executed, memory⟩ := whole_grid rest a p xs (vals (1 : Fin 2) ())
    trials nonempty sep s inputs
  refine ⟨final, (schedule_run rest a p _ s).trans executed, ?_, ?_⟩
  · intro port i _
    have eq : port = (0 : Fin 1) := Fin.eq_zero port
    subst port
    change mapCell id (final.mem p.Out i.1.val) =
      Cell.mk .int (f (vals (0 : Fin 2)) (vals (1 : Fin 2) ()) i)
    rw [memory]
    rw [show complete p (fun j => scalar a (xs j) (vals (1 : Fin 2) ())) s.mem p.Out i.1.val =
      Cell.mk .int (scalar a (xs i.1.val) (vals (1 : Fin 2) ())).toInt from if_pos ⟨rfl,i.1.isLt⟩]
    change Cell.mk .int (scalar a (xs i.1.val) (vals (1 : Fin 2) ())).toInt = _
    apply congrArg (Cell.mk .int)
    simpa only [xs, dif_pos i.1.isLt] using equation i
  · intro region offset outside
    rw [memory]
    by_cases hr : region = p.Out
    · have hi : ¬ offset < p.n := by
        intro active
        have h := outside.1 (0 : Fin 1) (⟨offset,active⟩,()) trivial
        exact h.elim (fun ne => ne hr) (fun ne => ne rfl)
      simp only [complete, hi, and_false, if_false]
    · simp only [complete, hr, false_and, if_false]

theorem native (rest : Algebra α) (a : Scalar α) (p : Parameters) (sep : p.Separated)
    (trials : Nat → List Config) (nonempty : ∀ k, k < segmentCount p.n → trials k ≠ []) :
    (io rest a p trials).Implements (target rest a p trials (nativeValues a)) :=
  IntegerProgramIO.Implements.intro
    (fun vals bounds inside s _ => safe rest a p trials vals bounds inside s)
    (fun vals s binding => result rest a p sep trials nonempty (nativeValues a)
      vals (fun _ => rfl) s binding)

theorem real (rest : Algebra ℝ) (p : Parameters) (sep : p.Separated)
    (trials : Nat → List Config) (nonempty : ∀ k, k < segmentCount p.n → trials k ≠ []) :
    (io rest Real.arithmetic p trials).ImplementsOn (domain rest p trials)
      (target rest Real.arithmetic p trials realValues) := by
  intro vals admitted bounds inside s binding
  refine ⟨safe rest Real.arithmetic p trials vals bounds inside s,
    result rest Real.arithmetic p sep trials nonempty realValues vals ?_ s binding⟩
  intro i
  exact Real.signed_readback _ _ admitted.1 (admitted.2 i)

theorem equivalent (rest : Algebra α) (a : Scalar α) (p : Parameters) (sep : p.Separated)
    (left right : Nat → List Config)
    (leftNonempty : ∀ k, k < segmentCount p.n → left k ≠ [])
    (rightNonempty : ∀ k, k < segmentCount p.n → right k ≠ []) :
    (io rest a p left).Equivalent (variant rest a p left right) :=
  IntegerProgramIO.EquivalentOn.of_implementsOn
    (native rest a p sep left leftNonempty) (native rest a p sep right rightNonempty)

#axiomsClean native
#axiomsClean real
#axiomsClean equivalent
end
end QuantizeGlobalRepair.IOCorrect
