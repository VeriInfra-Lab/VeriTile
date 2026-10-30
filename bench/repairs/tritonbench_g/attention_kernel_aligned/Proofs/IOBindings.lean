import bench.repairs.tritonbench_g.attention_kernel_aligned.IO

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Interface
open VeriTile Triton FP.Structural Full Whole
noncomputable section
set_option linter.unusedSimpArgs false
variable {α β : Type} [Inhabited β]

theorem point_binding (dtype : TileDType) (p : Parameters) (encode : α → β)
    (v : ChannelValues α (channels dtype p)) (s : State β)
    (bound : ∀ role j,
      (s.mem (channels dtype p)[role].buf (pointAddress p (point dtype p role j))).read (channels dtype p)[role].dtype=
        mapValue encode (channels dtype p)[role].dtype (v role j)) (i : Point p) :
    (s.mem (inputRegion p i.role) (pointAddress p i)).read dtype=
      mapValue encode dtype (pointValue dtype p v i) := by
  rcases i with ⟨role,b,h,n,d⟩
  fin_cases role
  · simpa [channels,point,pointValue,inputRegion] using bound (0 : Fin 4) (b,h,n,d,PUnit.unit)
  · simpa [channels,point,pointValue,inputRegion] using bound (1 : Fin 4) (b,h,n,d,PUnit.unit)
  · simpa [channels,point,pointValue,inputRegion] using bound (2 : Fin 4) (b,h,n,d,PUnit.unit)
  · simpa [channels,point,pointValue,inputRegion] using bound (3 : Fin 4) (b,h,n,d,PUnit.unit)

theorem readInputs_binding [Inhabited α] (dtype : TileDType) (p : Parameters) (encode : α → β)
    (v : ChannelValues α (channels dtype p)) (s : State β)
    (bound : ∀ i : Point p,(s.mem (inputRegion p i.role) (pointAddress p i)).read dtype=
      mapValue encode dtype (pointValue dtype p v i)) (i : Point p) :
    (s.mem (inputRegion p i.role) (pointAddress p i)).read dtype=
      mapValue encode dtype (readInputs dtype p v (inputRegion p i.role) (pointAddress p i)) := by
  classical
  have present : ∃ j : Point p,inputRegion p j.role=inputRegion p i.role ∧ pointAddress p j=pointAddress p i :=
    ⟨i,rfl,rfl⟩
  unfold readInputs
  rw [dif_pos present]
  simpa only [present.choose_spec.1,present.choose_spec.2] using bound present.choose

theorem point_bound (dtype : TileDType) (p : Parameters) (bounds : RegionName → Nat)
    (bound : ∀ role j,pointAddress p (point dtype p role j)<bounds (channels dtype p)[role].buf)
    (i : Point p) : pointAddress p i<bounds (inputRegion p i.role) := by
  rcases i with ⟨role,b,h,n,d⟩
  fin_cases role
  · simpa [channels,point,inputRegion] using bound (0 : Fin 4) (b,h,n,d,PUnit.unit)
  · simpa [channels,point,inputRegion] using bound (1 : Fin 4) (b,h,n,d,PUnit.unit)
  · simpa [channels,point,inputRegion] using bound (2 : Fin 4) (b,h,n,d,PUnit.unit)
  · simpa [channels,point,inputRegion] using bound (3 : Fin 4) (b,h,n,d,PUnit.unit)

/-- Agreement only on declared finite input coordinates, including aliases. -/
def Agree {γ : Type} (p : Parameters) (read₁ read₂ : RegionName → Nat → γ) : Prop :=
  ∀ i : Point p,read₁ (inputRegion p i.role) (pointAddress p i)=read₂ (inputRegion p i.role) (pointAddress p i)

end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Interface
