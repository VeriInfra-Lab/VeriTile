import bench.tritonbench_g.int4_matmul.Proofs.Kernel.GraphResults

namespace VeriTile.Bench.Repairs.Int4Matmul.SourceGraph
open VeriTile.Triton FP.Structural KernelSource KernelExecution
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

theorem writeLane_frame (o : Ops α) (cfg : Parameters) (initial current : RegisterState α)
    (values : CIndex cfg → α) (i : CIndex cfg) (r : RegionName) (a : Nat)
    (outside : active cfg initial i → r≠cfg.output ∨ a≠outputOffset cfg initial i) :
    (writeLane o cfg initial values current i).mem r a=current.mem r a := by
  unfold writeLane
  by_cases live : active cfg initial i
  · have miss : ¬ (r=cfg.output ∧ a=outputOffset cfg initial i) := by
      intro hit
      exact (outside live).elim (fun h=>h hit.1) (fun h=>h hit.2)
    split_ifs <;> simp only [State.write,miss,ite_false]
  · simp only [live,ite_false]

theorem fold_frame (o : Ops α) (cfg : Parameters) (initial current : RegisterState α)
    (values : CIndex cfg → α) (xs : List (CIndex cfg)) (r : RegionName) (a : Nat)
    (outside : ∀ i∈xs,active cfg initial i → r≠cfg.output ∨ a≠outputOffset cfg initial i) :
    (xs.foldl (writeLane o cfg initial values) current).mem r a=current.mem r a := by
  induction xs generalizing current with
  | nil => rfl
  | cons i xs ih =>
    rw [List.foldl_cons,ih _ (fun j hj=>outside j (by simp [hj]))]
    exact writeLane_frame o cfg initial current values i r a (outside i (by simp))

theorem fold_selected (o : Ops α) (cfg : Parameters) (initial current : RegisterState α)
    (values : CIndex cfg → α) (xs : List (CIndex cfg)) (distinct : xs.Nodup)
    (i : CIndex cfg) (member : i∈xs) (live : active cfg initial i)
    (unique : ∀ j∈xs,active cfg initial j → outputOffset cfg initial j=outputOffset cfg initial i → j=i) :
    (xs.foldl (writeLane o cfg initial values) current).mem cfg.output (outputOffset cfg initial i)=
      Cell.mk .real (if cfg.splits=1 then values i else
        o.atomicAdd ((current.mem cfg.output (outputOffset cfg initial i)).read .real) (values i)) := by
  induction xs generalizing current with
  | nil => simp at member
  | cons j xs ih =>
    have dj := List.nodup_cons.mp distinct
    rw [List.foldl_cons]
    by_cases same : j=i
    · subst j
      rw [fold_frame o cfg initial _ values xs cfg.output (outputOffset cfg initial i)]
      · simp [writeLane,live,State.write]
      · intro j hj livej
        right
        intro address
        have eq := unique j (by simp [hj]) livej address.symm
        subst j
        exact dj.1 hj
    · have tail : i∈xs := (List.mem_cons.mp member).resolve_left (Ne.symm same)
      rw [ih _ dj.2 tail (fun j hj=>unique j (by simp [hj]))]
      rw [writeLane_frame o cfg initial current values j cfg.output (outputOffset cfg initial i)]
      intro livej
      right
      intro address
      exact same (unique j (by simp) livej address.symm)

/-- A complete source invocation changes only active output addresses. -/
theorem source_frame (o : Ops α) (rest : Algebra α) (cfg : Parameters)
    (initial final : RegisterState (Node α))
    (ran : FP.Structural.exec (engine o rest) (kernel cfg) initial=some final)
    (r : RegionName) (a : Nat)
    (outside : ∀ i : CIndex cfg,active cfg initial i → r≠cfg.output ∨ a≠outputOffset cfg initial i) :
    final.mem r a=initial.mem r a := by
  obtain ⟨result,run,memory⟩ := source_memory o rest cfg initial
  rw [ran] at run
  injection run with same
  subst result
  rw [memory]
  exact fold_frame (ops o) cfg initial initial _ _ r a (fun i _=>outside i)

/-- Exact selected source cell, including the current destination at atomic
commit. The accumulation remains the original fixed-K8 live-carry recurrence. -/
theorem source_cell (o : Ops α) (rest : Algebra α) (cfg : Parameters)
    (initial final : RegisterState (Node α))
    (ran : FP.Structural.exec (engine o rest) (kernel cfg) initial=some final)
    (i : CIndex cfg) (live : active cfg initial i)
    (unique : ∀ j : CIndex cfg,active cfg initial j →
      outputOffset cfg initial j=outputOffset cfg initial i → j=i) :
    final.mem cfg.output (outputOffset cfg initial i)=
      Cell.mk .real (.value (if cfg.splits=1 then
        accumulation o cfg (observeState o initial) (steps cfg) i else
        o.atomicAdd (observe o ((initial.mem cfg.output (outputOffset cfg initial i)).read .real))
          (accumulation o cfg (observeState o initial) (steps cfg) i))) := by
  obtain ⟨result,run,memory⟩ := source_memory o rest cfg initial
  rw [ran] at run
  injection run with same
  subst result
  rw [memory]
  unfold written
  rw [fold_selected (ops o) cfg initial initial _ _
    (TileShape.allIndices_nodup _) i (TileShape.mem_allIndices _ i) live (fun j _=>unique j)]
  rw [accumulation_value]
  split_ifs <;> rfl

#axiomsClean source_frame
#axiomsClean source_cell
end
end VeriTile.Bench.Repairs.Int4Matmul.SourceGraph
