import bench.tritonbench_g.rowwise_quantization_triton.Proofs.StoreSafety

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.StoreSchedule
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]
set_option linter.unusedSimpArgs false

/-- Exact source-to-source run connection, not an execution premise. -/
theorem invocation_run (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) (row : Nat) (s : State α) :
    (invocation M round equal p row).run s = (IO.invocation M round equal p row).run s := by
  exact execution_eq M round equal p.columns p.width (lt_of_lt_of_le p.positive p.covers)
    ((IO.invocation M round equal p row).initial s)

theorem schedule_run (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) (order : List Nat) (s : State α) :
    runIntegerSchedule (order.map (invocation M round equal p)) s =
      runIntegerSchedule (order.map (IO.invocation M round equal p)) s := by
  induction order generalizing s with
  | nil => rfl
  | cons row rest ih =>
    simp only [List.map_cons,runIntegerSchedule,invocation_run]
    cases (IO.invocation M round equal p row).run s <;> simp [ih]

theorem schedule_safe (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) (schedule : Schedule p) (vals : ChannelValues α (IO.inputs p))
    (bounds : RegionName → Nat) (h : (IO.quantizerIO M round equal p schedule).InBounds bounds vals)
    (s : State α) :
    IntegerScheduleSafe bounds ((variant M round equal p schedule).schedule vals) s := by
  have one : ∀ row, row < p.rows → ∀ s : State α,
      (invocation M round equal p row).Safe bounds s := by
    intro row hr s
    change QuantizeIntegerStage.TraceSafe (KernelExecution.backend M equal) (some round) bounds
      ⟨Candidate.source "x" "q" "m" p.columns p.width,fun _ => none,QuantizeIntegerStage.outputRegister⟩
      ((IO.invocation M round equal p row).initial s)
    rw [IOImplements.invocation_initial]
    apply candidate_safe M round equal p.columns p.width (KernelGrid.atRow s row)
      (lt_of_lt_of_le p.positive p.covers) bounds
    · intro j hj
      exact h.2.1 (0:Fin 1) (⟨row,hr⟩,⟨j.val,hj⟩,()) trivial
    · intro j hj
      exact h.2.2.1 (0:Fin 2) (⟨row,hr⟩,⟨j.val,hj⟩,()) trivial
    · exact h.2.2.1 (1:Fin 2) (⟨row,hr⟩,()) trivial
  have go : ∀ order : List Nat, (∀ row ∈ order,row < p.rows) → ∀ s : State α,
      IntegerScheduleSafe bounds (order.map (invocation M round equal p)) s := by
    intro order
    induction order with
    | nil => intros;trivial
    | cons row rest ih =>
      intro valid s
      exact ⟨one row (valid _ List.mem_cons_self) s,
        fun t _ => ih (fun row hr => valid row (List.mem_cons_of_mem _ hr)) t⟩
  exact go schedule.order (fun row hr => List.mem_range.mp (schedule.complete.mem_iff.mp hr)) s

/-- Both complete actual row permutations succeed safely, agree on both full
typed outputs and preserve both exterior frames, for every scalar payload.
Original llrint, signed64/low-byte INT8 interpretation and FP16 casts are fixed. -/
theorem equivalent (ops : Operations α) (half : α → α) (round : α → BitVec 64)
    (rest : Algebra α) (p : Launch) (schedule : Schedule p) :
    (IO.quantizerIO (KernelProfile.algebra ops half p.tree rest) round (KernelProfile.equal ops) p schedule).Equivalent
      (variant (KernelProfile.algebra ops half p.tree rest) round (KernelProfile.equal ops) p schedule) := by
  intro vals _ bounds hb initial hin
  obtain ⟨safe,final,ran,outputs,frame⟩ :=
    IOImplements.implements ops half round rest p schedule vals trivial bounds hb initial hin
  refine ⟨safe,schedule_safe _ round _ p schedule vals bounds hb initial,final,final,ran,?_,?_,frame,frame⟩
  · exact (schedule_run _ round _ p schedule.order initial).trans ran
  · intro o j _
    rfl

#axiomsClean invocation_run
#axiomsClean schedule_run
#axiomsClean schedule_safe
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.StoreSchedule
