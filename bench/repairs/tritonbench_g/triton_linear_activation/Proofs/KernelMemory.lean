import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelTail

namespace VeriTile.Bench.Repairs.TritonLinearActivation.KernelMemory
open VeriTile.Triton FP.Structural KernelSource KernelValues
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 4000000

theorem masked_readback {ι : Type} [DecidableEq ι] (l : List ι) (region : RegionName)
    (off : ι → Nat) (value : ι → Cell α) (active : ι → Bool) (s : State α) (i : ι)
    (member : i∈l) (enabled : active i=Bool.true)
    (unique : ∀ j, active j=Bool.true → off j=off i → j=i) :
    (l.foldl (fun t j => if active j then t.write region (off j) (value j) else t) s).mem region (off i)=value i := by
  induction l generalizing s with
  | nil => simp at member
  | cons j rest ih =>
    by_cases later : i∈rest
    · simp only [List.foldl_cons]
      exact ih _ later
    · have same : j=i := by
        have eq : i=j := by simpa [List.mem_cons,later] using member
        exact eq.symm
      subst j
      simp only [List.foldl_cons,enabled,if_true]
      rw [State.masked_scatter_frame region off value (fun j => active j=Bool.true) rest region (off i)
        (Or.inr (by
          intro j hj activej equal
          exact later (unique j activej equal.symm ▸ hj)))]
      exact State.write_same s region (off i) (value i)

def InjectiveStores (p : Parameters) (s : State α) (offset : TileIndex [p.blockRows,p.blockColumns] → Nat) : Prop :=
  ∀ i j, active p s i=Bool.true → active p s j=Bool.true → offset i=offset j → i=j

theorem scatter_read (M : Algebra α) (mode : Mode) (p : Parameters) (geometry base : State α)
    (region : RegionName) (off : TileIndex [p.blockRows,p.blockColumns] → Nat) (values : Accumulator α p)
    (injective : InjectiveStores p geometry off) (i : TileIndex [p.blockRows,p.blockColumns])
    (enabled : active p geometry i=Bool.true) :
    (scatter M mode p geometry base region off values).mem region (off i)=outputCell M mode (values i) := by
  exact masked_readback _ _ _ _ _ _ i (TileShape.mem_allIndices _ _) enabled
    (fun j hj he => injective j i hj enabled he)

theorem scatter_frame (M : Algebra α) (mode : Mode) (p : Parameters) (geometry base : State α)
    (region : RegionName) (off : TileIndex [p.blockRows,p.blockColumns] → Nat) (values : Accumulator α p)
    (r : RegionName) (o : Nat)
    (outside : r≠region ∨ ∀ i, active p geometry i=Bool.true → o≠off i) :
    (scatter M mode p geometry base region off values).mem r o=base.mem r o := by
  apply State.masked_scatter_frame
  rcases outside with hr | ho
  · exact Or.inl hr
  · exact Or.inr (fun i _ hi => ho i hi)

/-- Every source-written output and saved preactivation, with the complete
untouched tagged-cell frame. Masked lanes are not required to have distinct
formed addresses; only actual stores need unique ownership. -/
theorem outputs (M : Algebra α) (D : Dot α) (mode : Mode) (p : Parameters) (s : State α)
    (positive : 0<p.blockInner) (separate : p.saved≠p.output)
    (outputUnique : InjectiveStores p s (outputOffset p s))
    (savedUnique : p.saveInputs=Bool.true → InjectiveStores p s (savedOffset p s)) :
    ∃ final, exec (engine M D) (source mode p) s=some final ∧
      (∀ i, active p s i=Bool.true → final.mem p.output (outputOffset p s i)=
        outputCell M mode (KernelValues.activation M p.activation (accumulated M D mode p s i))) ∧
      (p.saveInputs=Bool.true → ∀ i, active p s i=Bool.true → final.mem p.saved (savedOffset p s i)=
        outputCell M mode (accumulated M D mode p s i)) ∧
      (∀ r o, (r≠p.output ∨ ∀ i, active p s i=Bool.true → o≠outputOffset p s i) →
        (p.saveInputs=Bool.false ∨ r≠p.saved ∨ ∀ i, active p s i=Bool.true → o≠savedOffset p s i) →
        final.mem r o=s.mem r o) := by
  obtain ⟨final,execution,memory⟩ := KernelTail.source_executes M D mode p s positive
  refine ⟨final,execution,?_,?_,?_⟩
  · intro i hi
    rw [memory]
    exact scatter_read M mode p s _ _ _ _ outputUnique i hi
  · intro saved i hi
    rw [memory,written,if_pos saved,scatter_frame M mode p s _ _ _ _ p.saved _ (Or.inl separate)]
    exact scatter_read M mode p s _ _ _ _ (savedUnique saved) i hi
  · intro r o ho hs
    rw [memory,written,scatter_frame M mode p s _ _ _ _ r o ho]
    cases saved : p.saveInputs
    · simp [saved]
    · simp only [saved,if_true]
      apply scatter_frame
      simpa [saved] using hs

#axiomsClean masked_readback
#axiomsClean outputs
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.KernelMemory
