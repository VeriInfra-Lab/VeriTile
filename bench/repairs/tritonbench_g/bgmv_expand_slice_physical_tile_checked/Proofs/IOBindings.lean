import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Proofs.IOGrid

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicIO
open VeriTile.Triton FP.Structural Direct Direct.FPValues
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

def IndexBinding (h : Call) (xs : Inputs A h) (s : State A) : Prop :=
  0<h.N → ∀ batch : Fin h.batches,
    (PublicExecution.data s).indices batch.val=index xs batch

theorem index_binding (M : Algebra A) (R : Reduction A) (project : A → A)
    (h : Call) (xs : Inputs A h) (s : State A)
    (hin : (sourceIO M R project h).InputBinding xs s) : IndexBinding h xs s := by
  intro positive batch
  exact hin ⟨3,by simp [sourceIO,inputs]⟩ (batch,PUnit.unit) positive

theorem x_binding (M : Algebra A) (R : Reduction A) (project : A → A)
    (h : Call) (xs : Inputs A h) (s : State A)
    (hin : (sourceIO M R project h).InputBinding xs s)
    (batch : Fin h.batches) (positive : 0<h.N) (active : index xs batch≠-1)
    (k : Nat) (hk : k<inputWidth h) :
    (PublicExecution.data s).input (batch.val*h.K+k)=input xs batch k := by
  have bind := hin ⟨0,by simp [sourceIO,inputs]⟩ (batch,⟨k,hk⟩,PUnit.unit) ⟨positive,active⟩
  change (s.mem _ _).read .real=_ at bind
  simpa only [input,hk,↓reduceDIte] using bind

theorem weight_binding (M : Algebra A) (R : Reduction A) (project : A → A)
    (h : Call) (xs : Inputs A h) (s : State A)
    (hin : (sourceIO M R project h).InputBinding xs s)
    (batch : Fin h.batches) (row : Fin h.N) (active : index xs batch≠-1)
    (k : Nat) (hk : k<h.K) :
    (PublicExecution.data s).weights ((index xs batch).toNat*(h.N*h.K)+row.val*h.K+k)=
      weight xs batch row k := by
  have bind := hin ⟨1,by simp [sourceIO,inputs]⟩ (batch,row,⟨k,hk⟩,PUnit.unit) active
  change (s.mem _ _).read .fp16=_ at bind
  simpa only [weight,hk,↓reduceDIte] using bind

theorem previous_binding (M : Algebra A) (R : Reduction A) (project : A → A)
    (h : Call) (xs : Inputs A h) (s : State A)
    (hin : (sourceIO M R project h).InputBinding xs s)
    (batch : Fin h.batches) (row : Fin h.N) (active : index xs batch≠-1)
    (adds : h.add=Bool.true) :
    (PublicExecution.data s).output (batch.val*h.columns+row.val+h.slice)=previous xs batch row :=
  hin ⟨2,by simp [sourceIO,inputs]⟩ (batch,row,PUnit.unit) ⟨active,adds⟩

#axiomsClean index_binding
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicIO
