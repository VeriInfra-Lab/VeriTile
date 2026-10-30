import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.IO
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Proofs.IOExecution

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicExecution
open VeriTile.Triton FP.Structural Direct Direct.FPValues
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

def StepMemory (M : Algebra A) (R : Reduction A) (h : Call) (p : Geometry.Program h)
    (s : State A) : RegionName → Nat → Cell A :=
  if (data s).indices p.batch=-1 then s.mem else
    (s.write "Y" (SourceGrid.address h p) (.mk .real (SourceGrid.value M R h p (data s)))).mem

theorem step_indices (M : Algebra A) (R : Reduction A) (h : Call) (p : Geometry.Program h)
    (s t : State A) (same : t.mem=StepMemory M R h p s) :
    (data t).indices=(data s).indices := by
  funext b
  simp [data,same,StepMemory,State.write,ite_apply]

theorem step_other_value (M : Algebra A) (R : Reduction A) (h : Call)
    (p q : Geometry.Program h) (s t : State A) (same : t.mem=StepMemory M R h p s)
    (different : SourceGrid.address h q≠SourceGrid.address h p) :
    SourceGrid.value M R h q (data t)=SourceGrid.value M R h q (data s) := by
  have output : (t.mem "Y" (SourceGrid.address h q)).read .real=
      (s.mem "Y" (SourceGrid.address h q)).read .real := by
    have differentBase : q.batch*h.columns+q.row≠p.batch*h.columns+p.row := by
      intro equal
      apply different
      simp only [SourceGrid.address,Direct.FPGrid.address,equal]
    simp [same,StepMemory,State.write,differentBase,ite_apply]
  have inputs : (data t).input=(data s).input := by
    funext b
    simp [data,same,StepMemory,State.write,ite_apply]
  have weights : (data t).weights=(data s).weights := by
    funext b
    simp [data,same,StepMemory,State.write,ite_apply]
  have indices := step_indices M R h p s t same
  have productsEq : Direct.FPValues.products M h q.batch ((data s).indices q.batch).toNat q.row (data t)=
      Direct.FPValues.products M h q.batch ((data s).indices q.batch).toNat q.row (data s) := by
    unfold Direct.FPValues.products Direct.FPValues.loaded Direct.FPValues.weight
    rw [inputs,weights]
  simp only [SourceGrid.value,SourceValues.result,Direct.FPValues.sums,Direct.FPValues.old,indices,
    Fin.val_zero,Nat.zero_lt_one,↓reduceIte,Nat.add_zero,productsEq]
  rw [show (data t).output (q.batch*h.columns+q.row+h.slice)=
      (data s).output (q.batch*h.columns+q.row+h.slice) from output]

def Outside (h : Call) (s : State A) (jobs : List (Geometry.Program h))
    (name : RegionName) (offset : Nat) : Prop :=
  name≠"Y" ∨ ∀ p∈jobs,(data s).indices p.batch≠-1 → offset≠SourceGrid.address h p

/-- Whole source schedule with exact single-store frames. Its existing job
uniqueness ensures optional Y input-add always reads the initial value. -/
theorem grid_execution (M : Algebra A) (R : Reduction A) (h : Call)
    (jobs : List (Geometry.Program h)) (s : State A)
    (valid : ∀ b,b<h.batches → (data s).indices b=-1 ∨ 0≤(data s).indices b)
    (unique : jobs.Pairwise (fun p q => SourceGrid.address h p≠SourceGrid.address h q)) :
    ∃ final,runSchedule (jobs.map (PublicIO.invocation M R h)) s=some final ∧
      (∀ p∈jobs,(data s).indices p.batch≠-1 →
        final.mem "Y" (SourceGrid.address h p)=.mk .real (SourceGrid.value M R h p (data s))) ∧
      (∀ name offset,Outside h s jobs name offset → final.mem name offset=s.mem name offset) := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl,by simp,fun _ _ _ => rfl⟩
  | cons p jobs ih =>
    obtain ⟨different,uniqueRest⟩ := List.pairwise_cons.mp unique
    obtain ⟨middle,ran,memory⟩ := source_executes M R h p s (valid p.batch p.batchInside)
    have hm : middle.mem=StepMemory M R h p s := memory
    have indices := step_indices M R h p s middle hm
    obtain ⟨final,ranRest,values,frame⟩ := ih middle
      (fun b hb => by rw [indices];exact valid b hb) uniqueRest
    refine ⟨final,?_,?_,?_⟩
    · change ((PublicIO.invocation M R h p).run s).bind _=some final
      change (invocation M R h p.batch p.row).run s=some middle at ran
      rw [show (PublicIO.invocation M R h p).run s=some middle from ran]
      exact ranRest
    · intro q member active
      rcases List.mem_cons.mp member with same | later
      · subst q
        rw [frame "Y" (SourceGrid.address h p) (Or.inr (fun q hq _ => different q hq))]
        simp [hm,StepMemory,active,State.write]
      · rw [values q later (by rwa [indices])]
        exact congrArg (Cell.mk .real)
          (step_other_value M R h p q s middle hm (Ne.symm (different q later)))
    · intro name offset outside
      rw [frame name offset (outside.imp id (fun miss q hq active =>
        miss q (by simp [hq]) (by rwa [indices] at active)))]
      rcases outside with other | miss
      · simp [hm,StepMemory,State.write,other,ite_apply]
      · by_cases active : (data s).indices p.batch≠-1
        · simp [hm,StepMemory,State.write,active,miss p (by simp) active]
        · have skipped : (data s).indices p.batch=-1 := by simpa using active
          simp [hm,StepMemory,skipped]

#axiomsClean grid_execution
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicExecution
