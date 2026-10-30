import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceForwardGraph
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceCandidate

/-! Complete observations of the actual forward DSL execution. The scalar
model is arbitrary: no cancellation, reassociation, cast elimination, or
special-value exclusion occurs in these statements. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceForwardResult
open VeriTile.Triton FP.Structural SourceModel SourceForwardGraph
noncomputable section

def Outside (stride N : Nat) (s : State α) (r : RegionName) (o : Nat) : Prop :=
  (r≠"Mean" ∨ o≠s.pids 0) ∧ (r≠"Rstd" ∨ o≠s.pids 0) ∧
  (r≠"Y" ∨ ∀ j,Nat.lt j N → o≠s.pids 0*stride+j)

theorem final_output [Inhabited α] (m : Model α) (stride N B : Nat) (eps : ℝ)
    (covers : N≤B) (s : State α) (j : Nat) (hj : j<N) :
    (final m stride N B eps s).mem "Y" (s.pids 0*stride+j)=
      .mk .real (output m N B eps (row s stride) (vector s "W") (vector s "Bias") j) := by
  let i : TileIndex [B] := (⟨j,by omega⟩,PUnit.unit)
  apply State.masked_scatter_readback (active:=fun k : TileIndex [B] => k.1.val<N) (i:=i)
  · exact hj
  · intro a b hab
    apply Prod.ext
    · apply Fin.ext
      exact Nat.add_left_cancel hab
    · exact Subsingleton.elim _ _

theorem final_mean [Inhabited α] (m : Model α) (stride N B : Nat) (eps : ℝ) (s : State α) :
    (final m stride N B eps s).mem "Mean" (s.pids 0)=
      .mk .real (mean m N B (row s stride)) := by
  unfold final
  rw [State.masked_scatter_frame]
  · simp [State.write]
  · left; decide

theorem final_inverse [Inhabited α] (m : Model α) (stride N B : Nat) (eps : ℝ) (s : State α) :
    (final m stride N B eps s).mem "Rstd" (s.pids 0)=
      .mk .real (inverse m N B eps (row s stride)) := by
  unfold final
  rw [State.masked_scatter_frame]
  · simp [State.write]
  · left; decide

theorem final_frame [Inhabited α] (m : Model α) (stride N B : Nat) (eps : ℝ)
    (s : State α) (r : RegionName) (o : Nat) (h : Outside stride N s r o) :
    (final m stride N B eps s).mem r o=s.mem r o := by
  unfold final
  rw [State.masked_scatter_frame]
  · rw [State.write_other _ _ _ _ _ _ h.2.1,State.write_other _ _ _ _ _ _ h.1]
  · rcases h.2.2 with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i _ hi => ho i.1.val hi)

/-- Both programs terminate successfully in the same complete state. All
written cells and every exterior cell are characterized independently. -/
theorem executes [Inhabited α] (m : Model α) (stride N B : Nat) (eps : ℝ)
    (positive : 0<N) (covers : N≤B) (s : State α) :
    ∃ t,
      FP.Structural.exec (engine m)
        (TypedSource.forward "X" "Y" "W" "Bias" "Mean" "Rstd" stride N B eps) s=some t ∧
      FP.Structural.exec (engine m) (SourceCandidate.forward stride N B eps) s=some t ∧
      t.mem=(final m stride N B eps s).mem ∧
      (∀ j,j<N → t.mem "Y" (s.pids 0*stride+j)=
        .mk .real (output m N B eps (row s stride) (vector s "W") (vector s "Bias") j)) ∧
      t.mem "Mean" (s.pids 0)=.mk .real (mean m N B (row s stride)) ∧
      t.mem "Rstd" (s.pids 0)=.mk .real (inverse m N B eps (row s stride)) ∧
      (∀ r o,Outside stride N s r o → t.mem r o=s.mem r o) := by
  obtain ⟨t,ht⟩ := SourceForward.source_executes m stride N B eps positive covers s
  have hm := source_memory m stride N B eps positive covers s t ht
  refine ⟨t,ht,?_,hm,?_,?_,?_,?_⟩
  · rw [SourceCandidate.forward_equiv _ _ _ _ _ positive covers]
    exact ht
  · intro j hj
    rw [hm]
    exact final_output m stride N B eps covers s j hj
  · rw [hm]; exact final_mean m stride N B eps s
  · rw [hm]; exact final_inverse m stride N B eps s
  · intro r o ho
    rw [hm]
    exact final_frame m stride N B eps s r o ho

#axiomsClean executes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceForwardResult
