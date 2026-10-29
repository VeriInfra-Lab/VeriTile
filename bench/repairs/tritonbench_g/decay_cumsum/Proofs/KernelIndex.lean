import bench.repairs.tritonbench_g.decay_cumsum.Proofs.KernelTensor

/-! Fixed-width offsets from actual Tensor capacity. No separate allocation
extent cap or host dimension restriction is required: FP32 element capacity
inside the 64-bit storage window already bounds all signed element offsets. -/
namespace VeriTile.Bench.Repairs.DecayCumsum.KernelIndex
open VeriTile.Triton VeriTile.Torch.FP32 Geometry KernelTensor
noncomputable section

theorem nonempty (h : Call) (p : Phase) (hp : p∈phases h) (prog : Program h p) :
    0<h.B*h.H*h.T*h.DK := by
  have rows : 0<h.B*h.H := Nat.zero_lt_of_lt prog.rowInside
  have ph := phase_positive h p hp
  have time : 0<h.T := by
    have count := Nat.le_mul_of_pos_left p.span ph.1
    omega
  have width : 0<h.DK := by
    by_contra hn
    have hz : h.DK=0 := by omega
    have features : h.features=0 := by
      simp only [Call.features,hz,Nat.zero_add]
      exact Nat.div_eq_of_lt (by have pos:=h.featureTilePositive;omega)
    have inside:=prog.featureInside
    omega
  exact Nat.mul_pos (Nat.mul_pos rows time) width

theorem cells_signed (b : Buffers h kind) (r : RegionName) (hr : r∈argumentRoles kind) :
    h.cells<2^63 := by
  have cap := (b.view r).spanFits
  change (b.view r).base+4*(b.tensor r).metadata.numel≤2^64 at cap
  rw [b.count r hr] at cap
  norm_num at cap ⊢
  omega

theorem addresses (b : Buffers h kind) (p : Phase) (hp : p∈phases h)
    (contract : InputContract h kind observed scaleIsFloat) (prog : Program h p)
    (r : RegionName) (hr : r∈argumentRoles kind) (t : Fin p.span) (j : Fin h.BK) :
    let pid := KernelContract.pid contract (nonempty h p hp prog) p hp prog
    (NativeIndex.offsetWord h p pid t.val j.val).toInt=
      (NativeIndex.relative h p pid t.val j.val:Int) ∧
    (sourcePointer b p r+NativeIndex.offsetWord h p pid t.val j.val*BitVec.ofNat 64 4).toNat=
      (b.view r).address (Geometry.offset h p prog t.val j.val) ∧
    (b.view r).address (Geometry.offset h p prog t.val j.val)+4≤2^64 := by
  let pid := KernelContract.pid contract (nonempty h p hp prog) p hp prog
  have inside := offset_inside h p hp prog t j
  have eq := NativeIndex.shifted_offset h p pid t.val j.val
  change p.start*h.width+NativeIndex.relative h p pid t.val j.val=Geometry.offset h p prog t.val j.val at eq
  have lane : j.val<2147483648 := by have lim:=contract.blockLimit;have jl:=j.isLt;omega
  have bound : NativeIndex.relative h p pid t.val j.val<2^63 := by
    have signed := cells_signed b r hr
    omega
  have span := (b.view r).spanFits
  change (b.view r).base+4*(b.tensor r).metadata.numel≤2^64 at span
  rw [b.count r hr] at span
  have endBound : (b.view r).address (Geometry.offset h p prog t.val j.val)+4≤2^64 := by
    unfold DataView.address
    omega
  change (NativeIndex.offsetWord h p pid t.val j.val).toInt=_ ∧ _ ∧ _
  rw [NativeIndex.offset_word h p pid t.val j.val lane]
  refine ⟨NativeIndex.Shared.signed64 _ bound,?_,endBound⟩
  have word : sourcePointer b p r+BitVec.ofNat 64 (NativeIndex.relative h p pid t.val j.val)*BitVec.ofNat 64 4=
      BitVec.ofNat 64 ((b.view r).address (Geometry.offset h p prog t.val j.val)) := by
    simp only [sourcePointer,DataView.address,shift,←BitVec.ofNat_mul,←BitVec.ofNat_add]
    congr 1
    rw [←eq]
    ring
  rw [word,BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega)]

theorem loop_bounds (b : Buffers h kind) (p : Phase) (hp : p∈phases h)
    (prog : Program h p) (r : RegionName) (hr : r∈argumentRoles kind) :
    0<p.span ∧ p.span<2^63 ∧ (BitVec.ofNat 64 p.span).toInt=(p.span:Int) ∧
      (BitVec.ofNat 64 (p.span-1)).toInt=(p.span-1:Nat) := by
  have phase := phase_positive h p hp
  have rows := Nat.zero_lt_of_lt prog.rowInside
  have width : 0<h.width := by
    exact Nat.mul_pos (Nat.zero_lt_of_lt prog.featureInside) h.featureTilePositive
  have timeBound : p.span≤h.T := by
    have count := Nat.le_mul_of_pos_left p.span phase.1
    omega
  have total : h.T≤h.cells :=
    (Nat.le_mul_of_pos_right h.T width).trans (Nat.le_mul_of_pos_left _ rows)
  have bound := (timeBound.trans total).trans_lt (cells_signed b r hr)
  exact ⟨phase.2.1,bound,NativeIndex.Shared.signed64 _ bound,
    NativeIndex.Shared.signed64 _ ((Nat.sub_le _ _).trans_lt bound)⟩

#axiomsClean addresses
#axiomsClean loop_bounds
end
end VeriTile.Bench.Repairs.DecayCumsum.KernelIndex
