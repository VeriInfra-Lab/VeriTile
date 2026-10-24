import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.IOInputs

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
open VeriTile.Triton FP.Structural Quoted
noncomputable section
namespace PublicData
set_option maxHeartbeats 1600000
variable {α : Type} [Inhabited α]

theorem output_range (mode : Mode) (s : Shape) (final : Bool) (o : Fin 5)
    (j : TileIndex (outputPorts s)[o].shape) (active : outputActive mode final o) :
    writtenRange mode s final (region (kind o)) (outputWindow s o j) := by
  fin_cases o <;> cases mode <;>
    simp [outputActive] at active
  all_goals have hj := j.1.isLt
  all_goals simp_all [writtenRange,forwardRange,backwardRange,kind,region,outputWindow,outputPorts]

theorem outside_written (mode : Mode) (a : Arithmetic α) (s : Shape) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool)
    (xs : (io mode a s p incoming finalGradient seed final).Inputs)
    (r : RegionName) (i : Nat) (outside : (io mode a s p incoming finalGradient seed final).Outside xs r i) :
    ¬writtenRange mode s final r i := by
  intro hi
  cases mode
  · rcases hi with ⟨rfl,hi⟩|⟨hf,rfl,hi⟩
    · have h := outside.1 0 (⟨i,hi⟩,PUnit.unit) (by simp [inputActive,outputActive])
      exact h.elim (fun h => h rfl) (fun h => h rfl)
    · have h := outside.1 4 (⟨i,hi⟩,PUnit.unit) (by simp [outputActive,hf])
      exact h.elim (fun h => h rfl) (fun h => h rfl)
  · rcases hi with ⟨rfl,hi⟩|⟨rfl,hi⟩|⟨rfl,hi⟩
    · have h := outside.1 1 (⟨i,hi⟩,PUnit.unit) (by simp [outputActive])
      exact h.elim (fun h => h rfl) (fun h => h rfl)
    · have h := outside.1 2 (⟨i,hi⟩,PUnit.unit) (by simp [outputActive])
      exact h.elim (fun h => h rfl) (fun h => h rfl)
    · have h := outside.1 3 (⟨i,hi⟩,PUnit.unit) (by simp [outputActive])
      exact h.elim (fun h => h rfl) (fun h => h rfl)

end PublicData
namespace Logical
open Quoted
variable {α : Type} {mode : Mode} {s : Shape} {seed final : Bool} {x y : Inputs α}

theorem partialValue_eq (a : Arithmetic α) (plans : Plans) (agree : Agreement mode s seed final x y)
    (kind : PartialKind) (i : Nat) (inside : writtenRange mode s final (region kind) i) :
    partialValue a s plans x seed final kind i=partialValue a s plans y seed final kind i := by
  cases kind
  · cases mode <;> simp [writtenRange,forwardRange,backwardRange,region] at inside
    obtain ⟨_,hh,ht,_,_⟩ := unflatten_partial s s.NK s.DV i inside
    exact partialO_eq a plans agree _ hh _ ht _ _
  · cases mode <;> simp [writtenRange,forwardRange,backwardRange,region] at inside
    obtain ⟨_,hh,ht,_,_⟩ := unflatten_partial s s.NV s.DK i inside
    exact partialQ_eq a plans agree _ hh _ ht _ _
  · cases mode <;> simp [writtenRange,forwardRange,backwardRange,region] at inside
    obtain ⟨_,hh,ht,_,_⟩ := unflatten_partial s s.NV s.DK i inside
    exact partialK_eq a plans agree _ hh _ ht _ _
  · cases mode <;> simp [writtenRange,forwardRange,backwardRange,region] at inside
    obtain ⟨_,hh,ht,_,_⟩ := unflatten_partial s s.NK s.DV i inside
    exact partialV_eq a plans agree _ hh _ ht _ _
  · cases mode <;> simp [writtenRange,forwardRange,backwardRange,region] at inside
    obtain ⟨hh,_,_,_⟩ := unflatten_state s i inside.2
    exact state_eq a agree _ hh _ _ _ (by omega)

#axiomsClean partialValue_eq
end Logical
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
