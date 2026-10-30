import bench.tritonbench_g.fused_recurrent_retention.APISource
import bench.tritonbench_g.fused_recurrent_retention.Proofs.Integer
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

/-- Torch contiguous strides include max(dim,1), including empty T. -/
def Tensor4.InBounds (x : Tensor4) (a h t c : Nat) : Prop :=
  x.offset+a*x.strides 0+h*x.strides 1+t*x.strides 2+c*x.strides 3<x.capacity

/-- Structural storage domain, with true aliases allowed among read-only inputs.
No condition requires finite intermediate values. -/
structure SourceDomain (s : Shape) (b : InputBuffers) (seed final : Bool) : Prop where
  shape : s.Admitted
  ints : IntegerDomain s b
  qDense : b.q.strides=contiguous4 s.B s.H s.T s.DK
  kDense : b.k.strides=contiguous4 s.B s.H s.T s.DK
  vDense : b.v.strides=contiguous4 s.B s.H s.T s.DV
  seedDense : seed=Bool.true → b.seed.strides=contiguous4 s.B s.H s.DK s.DV
  devices : b.k.device=b.q.device ∧ b.v.device=b.q.device ∧ b.do_.device=b.q.device ∧
    (seed=Bool.true → b.seed.device=b.q.device) ∧ (final=Bool.true → b.finalGrad.device=b.q.device)
  qAllocated : ∀ h<s.BH,∀ t<s.T,∀ k<s.DK,
    0≤(b.qAddress s h t k).2 ∧ (b.qAddress s h t k).2<(b.q.capacity : Int)
  kAllocated : ∀ h<s.BH,∀ t<s.T,∀ k<s.DK,
    0≤(b.kAddress s h t k).2 ∧ (b.kAddress s h t k).2<(b.k.capacity : Int)
  vAllocated : ∀ h<s.BH,∀ t<s.T,∀ v<s.DV,
    0≤(b.vAddress s h t v).2 ∧ (b.vAddress s h t v).2<(b.v.capacity : Int)
  doAllocated : ∀ h<s.BH,∀ t<s.T,∀ v<s.DV,b.do_.InBounds (h/s.H) (h%s.H) t v
  seedAllocated : seed=Bool.true → ∀ h<s.BH,∀ k<s.DK,∀ v<s.DV,
    0≤(b.seedAddress s h k v).2 ∧ (b.seedAddress s h k v).2<(b.seed.capacity : Int)
  finalAllocated : final=Bool.true → ∀ h<s.BH,∀ k<s.DK,∀ v<s.DV,
    b.finalGrad.InBounds (h/s.H) (h%s.H) k v

theorem head_positive (s : Shape) (h : Nat) (hh : h<s.BH) : 0<s.H := by
  unfold Shape.BH at hh
  by_contra hn
  have hz : s.H=0 := by omega
  simp [hz] at hh

theorem head_batch_bounds (s : Shape) (h : Nat) (hh : h<s.BH) :
    h/s.H<s.B ∧ h%s.H<s.H := by
  have hp := head_positive s h hh
  exact ⟨(Nat.div_lt_iff_lt_mul hp).mpr (by simpa [Shape.BH] using hh),Nat.mod_lt _ hp⟩

theorem source_head_stride (s : Shape) (width : Nat) (ht : 0<s.T) (hw : 0<width) :
    contiguous4 s.B s.H s.T width 1=s.T*width := by
  simp [contiguous4,Nat.max_eq_left ht,Nat.max_eq_left hw]

theorem source_partial_pointer (s : Shape) (width split h t c : Nat) (ht : 0<s.T) (hw : 0<width) :
    (h+split*s.B*s.H)*contiguous4 s.B s.H s.T width 1+c+t*width=
      partialOffset s width split h t c := by
  rw [source_head_stride s width ht hw]
  unfold partialOffset Shape.BH
  ring

theorem dense_source_address (s : Shape) (width h t c : Nat)
    (hh : h<s.BH) (ht : t<s.T) (hw : 0<width) :
    (h/s.H)*contiguous4 s.B s.H s.T width 0+
      (h%s.H)*contiguous4 s.B s.H s.T width 1+
      t*contiguous4 s.B s.H s.T width 2+c*contiguous4 s.B s.H s.T width 3=
      h*contiguous4 s.B s.H s.T width 1+t*width+c := by
  have hp := head_positive s h hh
  have hpT : 0<s.T := by omega
  simp [contiguous4,Nat.max_eq_left hp,Nat.max_eq_left hpT,Nat.max_eq_left hw]
  have he := Nat.mod_add_div h s.H
  calc
    _ = (h%s.H+s.H*(h/s.H))*(s.T*width) := by ring
    _ = _ := by rw [he]

#axiomsClean head_batch_bounds
#axiomsClean source_partial_pointer
#axiomsClean dense_source_address
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
