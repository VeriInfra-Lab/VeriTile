import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.KernelBinding

namespace VeriTile.Bench.Repairs.RmsnormFused.Index
open VeriTile Triton Binding
noncomputable section

def rowWord (row : Nat) : BitVec 64 := BitVec.ofInt 64 (BitVec.ofNat 32 row).toInt
def pointer (base : BitVec 64) (N row col : Nat) : BitVec 64 :=
  base+(rowWord row*BitVec.ofNat 64 N+BitVec.ofNat 64 col)*BitVec.ofNat 64 4

theorem signed_exact (n b : Nat) (bound : n<2^(b-1)) (positive : 0<b) :
    (BitVec.ofNat b n).toInt=(n:Int) := by
  have capacity : n<2^b := lt_of_lt_of_le bound (Nat.pow_le_pow_right (by decide) (by omega))
  have nat : (BitVec.ofNat b n).toNat=n := by rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt capacity]
  rw [BitVec.toInt_eq_toNat_of_lt,nat]
  rw [nat]
  have powers : 2^b=2*2^(b-1) := by
    calc
      2^b=2^((b-1)+1) := congrArg (fun n : Nat => 2^n) (by omega)
      _ = _ := by rw [Nat.pow_succ,Nat.mul_comm]
  rw [powers]
  omega

theorem pid (x w : Tensor) (eps : EpsilonInfo) (c : InputContract x.info w.info eps)
    (row : Nat) (active : row<(geometry x.info).rows) : rowWord row=BitVec.ofNat 64 row := by
  have count := geometry_rows x.info c.rank c.widthPositive
  have bound : row<2^31 := by rw [count] at active; have := c.grid; omega
  simp [rowWord,signed_exact row 32 bound (by decide),BitVec.ofInt_natCast]

theorem formed_index (x w : Tensor) (eps : EpsilonInfo) (c : InputContract x.info w.info eps)
    (row col : Nat) (hr : row<(geometry x.info).rows)
    (hc : col<RmsnormFused.block (width x.info)) :
    (rowWord row*BitVec.ofNat 64 (width x.info)+BitVec.ofNat 64 col).toInt=(row*width x.info+col:Nat) := by
  have bound := block_bound (width x.info)
  have rowBound : row*width x.info < x.size := by
    rw [(input_sizes x w eps c).1]
    simpa only [Shape.offset,Nat.add_zero] using offset_bounded (geometry x.info) row 0 hr c.widthPositive
  have size := c.byteExtent
  change x.size≤(2^63-1)/4 at size
  rw [pid x w eps c row hr,←BitVec.ofNat_mul,←BitVec.ofNat_add]
  exact signed_exact _ 64 (by omega) (by decide)

theorem view_pointer (t : Tensor) (row N col : Nat) (pidExact : rowWord row=BitVec.ofNat 64 row)
    (active : row*N+col<t.size) :
    (pointer t.actual.dataPtr N row col).toNat=t.view.address (row*N+col) ∧
      ∀ byte,byte<4 → t.view.address (row*N+col)+byte<2^64 := by
  have nonempty : 0<t.actual.metadata.numel := by change 0<t.size; omega
  have actual := VeriTile.Torch.FP32.actual_pointer t.actual t.laws t.contiguous nonempty
  change BitVec.ofNat 64 (t.view.address 0)=t.actual.dataPtr at actual
  have span := t.view.spanFits
  change t.view.base+4*t.size≤2^64 at span
  constructor
  · rw [←actual]
    simp only [pointer,pidExact,←BitVec.ofNat_mul,←BitVec.ofNat_add,BitVec.toNat_ofNat,
      VeriTile.Torch.FP32.DataView.address]
    rw [Nat.mod_eq_of_lt (by omega)]
    omega
  · intro byte hb
    unfold VeriTile.Torch.FP32.DataView.address
    omega

/-- Widened PID and all active pointer bytes refer to those same actual Tensor
storage views. Inactive masked pointers need no allocation/padding assumption. -/
theorem source_addresses (x w y : Tensor) (eps : EpsilonInfo)
    (c : InputContract x.info w.info eps) (out : Output x w y)
    (row col : Nat) (hr : row<(geometry x.info).rows) (hc : col<width x.info) :
    (pointer x.actual.dataPtr (width x.info) row col).toNat=x.view.address (row*width x.info+col) ∧
    (pointer y.actual.dataPtr (width x.info) row col).toNat=y.view.address (row*width x.info+col) ∧
    (w.actual.dataPtr+BitVec.ofNat 64 col*BitVec.ofNat 64 4).toNat=w.view.address col ∧
    (∀ byte,byte<4 → x.view.address (row*width x.info+col)+byte<2^64 ∧
      y.view.address (row*width x.info+col)+byte<2^64 ∧ w.view.address col+byte<2^64) := by
  have sizes := input_sizes x w eps c
  have active : row*width x.info+col<x.size := by
    rw [sizes.1]
    exact offset_bounded (geometry x.info) row col hr hc
  have ix := view_pointer x row (width x.info) col (pid x w eps c row hr) active
  have iy := view_pointer y row (width x.info) col (pid x w eps c row hr) (by rw [out.size]; exact active)
  have iw := view_pointer w 0 0 col (by decide) (by simpa [sizes.2] using hc)
  refine ⟨ix.1,iy.1,?_,fun byte hb => ⟨ix.2 byte hb,iy.2 byte hb,by simpa using iw.2 byte hb⟩⟩
  simpa [pointer,rowWord] using iw.1

theorem lane_mask (x w : Tensor) (eps : EpsilonInfo) (c : InputContract x.info w.info eps)
    (col : Nat) (inside : col<block (width x.info)) :
    (BitVec.ofNat 32 col).slt (BitVec.ofNat 32 (width x.info))=decide (col<width x.info) := by
  have bound := block_bound (width x.info)
  have covered := c.block
  rw [BitVec.slt_eq_decide,signed_exact col 32 (by omega) (by decide),
    signed_exact (width x.info) 32 (by omega) (by decide)]
  norm_cast

theorem active_alignment (x w y : Tensor) (eps : EpsilonInfo)
    (c : InputContract x.info w.info eps) (out : Output x w y)
    (row : Nat) (hr : row<(geometry x.info).rows) :
    x.actual.dataPtr.toNat%4=0 ∧ w.actual.dataPtr.toNat%4=0 ∧ y.actual.dataPtr.toNat%4=0 := by
  have nonempty : 0<x.size := by
    rw [(input_sizes x w eps c).1]
    exact Nat.mul_pos (by omega) c.widthPositive
  have align := c.alignment (by change x.size≠0; omega)
  exact ⟨align.1,align.2,out.aligned (by rw [out.size]; exact nonempty)⟩

def Correspondence (x w y : Tensor) : Prop :=
  (∀ row,row<(geometry x.info).rows → rowWord row=BitVec.ofNat 64 row) ∧
  (∀ col,col<block (width x.info) →
    BitVec.ofInt 64 (BitVec.ofNat 32 col).toInt=BitVec.ofNat 64 col ∧
    (BitVec.ofNat 32 col).slt (BitVec.ofNat 32 (width x.info))=decide (col<width x.info)) ∧
  (∀ row col,row<(geometry x.info).rows → col<block (width x.info) →
    (rowWord row*BitVec.ofNat 64 (width x.info)+BitVec.ofNat 64 col).toInt=(row*width x.info+col:Nat)) ∧
  (∀ row,row<(geometry x.info).rows →
    x.actual.dataPtr.toNat%4=0 ∧ w.actual.dataPtr.toNat%4=0 ∧ y.actual.dataPtr.toNat%4=0)

theorem correspondence (x w y : Tensor) (eps : EpsilonInfo)
    (c : InputContract x.info w.info eps) (out : Output x w y) : Correspondence x w y := by
  refine ⟨pid x w eps c,?_,formed_index x w eps c,active_alignment x w y eps c out⟩
  intro col hc
  have bound := block_bound (width x.info)
  refine ⟨?_,lane_mask x w eps c col hc⟩
  rw [signed_exact col 32 (by omega) (by decide),BitVec.ofInt_natCast]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFused.Index
