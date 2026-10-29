import bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeBinding

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.NativeInterface.SafetyProof
open VeriTile Triton FP.Structural Reused Geometry NativeInstructions NativeStaging
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000

theorem invocation_initial (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (job : WholeGrid.Job c) (s : State NativeStaging.Expr) :
    (invocation c offset size scale window complement job).initial s=FPGrid.programState c s job := rfl

theorem invocation_run (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (job : WholeGrid.Job c) (s : State NativeStaging.Expr) :
    (invocation c offset size scale window complement job).run s=
      FPGrid.invoke algebra dotModel c offset size scale window complement job s := by
  unfold Invocation.run
  rw [invocation_initial]
  change FP.Structural.exec _ (kernel c offset size scale window complement) _=_
  rw [shared_source]
  rfl

theorem schedule_run (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (order : List (WholeGrid.Job c)) (s : State NativeStaging.Expr) :
    runSchedule (order.map (invocation c offset size scale window complement)) s=
      FPGrid.run algebra dotModel c offset size scale window complement order s := by
  induction order generalizing s with
  | nil => rfl
  | cons job rest ih =>
    simp only [List.map_cons,runSchedule,FPGrid.run,invocation_run]
    congr 1
    funext next
    exact ih next

theorem values (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (xs : Inputs c) (domain : initialOutput c xs) (s : State NativeStaging.Expr)
    (binding : (programIO c offset size scale window complement).InputCells xs s) :
    ∃ t, runSchedule ((programIO c offset size scale window complement).schedule xs) s=some t ∧
      (programIO c offset size scale window complement).OutputBinding xs
        (formula c offset size scale window complement xs) t ∧
      (programIO c offset size scale window complement).Frame xs s t := by
  by_cases he : RealInterface.empty c
  · refine ⟨s,?_,?_,?_⟩
    · simp [programIO,he,runSchedule]
    · intro o j active
      change Fin 1 at o
      have ho : o=(0 : Fin 1) := Subsingleton.elim _ _
      subst o
      have hz := binding (3 : Fin 4) j (Or.inr he)
      change s.mem "out" j.1.val=Cell.mk .real (encode c (xs (3 : Fin 4) j)) at hz
      change mapCell lower (s.mem "out" j.1.val)=Cell.mk .real (formula c offset size scale window complement xs (0 : Fin 1) j)
      rw [hz]
      change Cell.mk .real (lower (encode c (xs (3 : Fin 4) j)))=_
      rw [lower_encode]
      have hz : xs (3 : Fin 4) j=NativeFormula.zero32 := by
        simpa only [Prod.mk.eta,Subsingleton.elim j.2 ()] using domain he j.1
      rw [hz]
      congr 1
      rcases he with hn|ho
      · simp [formula,NativeReference.column,foldRange,hn]
      · exact Fin.elim0 (j.1.cast ho)
    · intro r a outside
      rfl
  · have hn : c.N≠0 := fun h => he (Or.inl h)
    obtain ⟨t,run,same,values⟩ := FPReturned.grid_tensor algebra dotModel c s offset size scale window complement
    refine ⟨t,?_,?_,?_⟩
    · simpa only [programIO,if_neg he,schedule_run] using run
    · intro o j active
      change Fin 1 at o
      have ho : o=(0 : Fin 1) := Subsingleton.elim _ _
      subst o
      let coord := RealInterface.outputCoordinates c j.1
      change mapCell lower (t.mem "out" j.1.val)=Cell.mk .real (formula c offset size scale window complement xs (0 : Fin 1) j)
      rw [←RealInterface.output_address c j.1,values]
      change Cell.mk .real (lower (FPReturned.target algebra dotModel c offset size scale window complement s
        coord.1 coord.2.1 coord.2.2))=Cell.mk .real (NativeReference.column c
          (RealInterface.logicalInputs c (flatInputs c xs)) coord.1 coord.2.1 offset size scale window complement coord.2.2)
      congr 1
      have lowered := NativeLoop.returned_lower c s offset size scale window complement coord.1 coord.2.1 coord.2.2
      simp only [FPReturned.returnedValue,if_neg hn] at lowered
      exact lowered.trans (Binding.returned c offset size scale window complement s xs binding _ _ _)
    · intro r a outside
      apply FPReturned.run_frame algebra dotModel c offset size scale window complement (WholeGrid.jobs c) s t run r a
      by_cases hr : r="out"
      · right
        by_contra ha
        have hl : a<c.outputSize := Nat.lt_of_not_ge ha
        have h := outside.1 (0 : Fin 1) (⟨a,hl⟩,()) trivial
        exact h.elim (fun h => h hr) (fun h => h rfl)
      · exact Or.inl hr

#axiomsClean values
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.NativeInterface.SafetyProof
