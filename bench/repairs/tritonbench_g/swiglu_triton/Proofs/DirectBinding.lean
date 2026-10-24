import bench.repairs.tritonbench_g.swiglu_triton.Contract
import bench.tritonbench_g.swiglu_fwd.Proofs.WholeIndex

namespace VeriTile.Bench.Repairs.SwigluTriton.DirectHost
open Memory DirectPhysical
noncomputable section
set_option maxHeartbeats 2000000

theorem check_iff (p : Prepared backward) : check p = Bool.true ↔ Admitted p := by
  simp only [check,Bool.and_eq_true,decide_eq_true_eq]
  exact ⟨fun ⟨⟨⟨⟨⟨⟨⟨⟨a,b⟩,c⟩,d⟩,e⟩,f⟩,g⟩,i⟩,j⟩ => ⟨a,b,c,d,e,f,g,i,j⟩,
    fun h => ⟨⟨⟨⟨⟨⟨⟨⟨h.payloads,h.counts⟩,h.forwardShape⟩,h.sameDevice⟩,h.positive⟩,h.rows32⟩,h.padded64⟩,h.alignment⟩,h.separate⟩⟩

theorem strided (p : Prepared backward) (h : Admitted p) (r : Role) :
    (p.tensor r).metadata.isStrided = Bool.true := by
  have hp := h.payloads r
  simp only [payload,Bool.and_eq_true,Bool.not_eq_true'] at hp
  exact hp.1.1.1.1

theorem fp32 (p : Prepared backward) (h : Admitted p) (r : Role) :
    (p.tensor r).metadata.isFP32 = Bool.true := by
  have hp := h.payloads r
  simp only [payload,Bool.and_eq_true,Bool.not_eq_true'] at hp
  exact hp.1.1.1.2

theorem contiguous (p : Prepared backward) (h : Admitted p) (r : Role) :
    (p.tensor r).metadata.isContiguous = Bool.true := by
  have hp := h.payloads r
  simp only [payload,Bool.and_eq_true,Bool.not_eq_true'] at hp
  exact hp.1.2

def allocation (p : Prepared backward) (h : Admitted p) (r : Role) : Allocation :=
  ⟨(p.tensor r).object,(p.tensor r).metadata.device.index,(p.tensor r).metadata.dataPtr.toNat,
    4*(p.tensor r).metadata.numel,(p.tensor r).contiguousSpan (strided p h r) (fp32 p h r) (contiguous p h r)⟩

def layout (p : Prepared backward) (h : Admitted p) : Layout :=
  ⟨allocation p h,p.rows,p.cols,blockSize p.cols⟩

theorem count (p : Prepared backward) (h : Admitted p) (r : Role) :
    (p.tensor r).metadata.numel = p.rows*p.cols := h.counts r

theorem valid (p : Prepared backward) (h : Admitted p) : Valid (layout p h) where
  cols := Nat.le_pow_clog (by decide) p.cols
  capacity := by intro r; simp [layout,allocation,Allocation.words,count p h]
  separate := by
    intro r s ne
    by_cases nonempty : 0 < p.rows*p.cols
    · exact Or.inr (h.separate nonempty r s ne)
    · have zero : p.rows*p.cols = 0 := by omega
      simp only [Allocation.separate,layout,allocation,count p h,zero,Nat.mul_zero,Nat.add_zero]
      exact Or.inr (Nat.le_total _ _)

theorem key_binding (p : Prepared backward) (h : Admitted p) (r : Role) (row col : Nat)
    (hr : row < p.rows) (hc : col < p.cols) :
    (layout p h).key r row col = (p.tensor r).key (row*p.cols+col) := by
  have inside : row*p.cols+col < (p.tensor r).metadata.numel := by
    rw [count p h]; exact index_bound _ _ _ _ hr hc
  rw [(p.tensor r).contiguousKeys (strided p h r) (fp32 p h r) (contiguous p h r) _ inside]
  rfl

/-- Exact finite row product and masked column, including padded lanes.
A masked pointer may wrap; every active address is separately proved inside
its actual contiguous Tensor span. -/
theorem indices (p : Prepared backward) (h : Admitted p) (small : blockSize p.cols ≤ 65536)
    (row lane : Nat) (hr : row < p.rows) (hl : lane < blockSize p.cols) :
    (BitVec.ofNat 64 row * BitVec.ofNat 64 p.cols + BitVec.ofNat 64 lane).toInt =
      (row*p.cols+lane : Nat) := by
  have scaled := Nat.mul_le_mul_right p.cols (Nat.le_of_lt hr)
  have padded := h.padded64
  have inside : row*p.cols+lane < 2^63 := by omega
  rw [←BitVec.ofNat_mul,←BitVec.ofNat_add]
  have value : (BitVec.ofNat 64 (row*p.cols+lane)).toNat = row*p.cols+lane := by
    rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega)]
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [value]; omega),value]

theorem pointer (p : Prepared backward) (h : Admitted p) (r : Role) (row col : Nat)
    (hr : row < p.rows) (hc : col < p.cols) :
    (BitVec.ofNat 64 (p.tensor r).metadata.dataPtr.toNat +
      BitVec.ofNat 64 (4*(row*p.cols+col))).toNat = ((p.tensor r).key (row*p.cols+col)).address ∧
    ((p.tensor r).key (row*p.cols+col)).address % 4 = 0 := by
  have inside := index_bound p.rows p.cols row col hr hc
  have bounds := (p.tensor r).contiguousSpan (strided p h r) (fp32 p h r) (contiguous p h r)
  rw [count p h] at bounds
  rw [(p.tensor r).contiguousKeys (strided p h r) (fp32 p h r) (contiguous p h r) _ (by rwa [count p h])]
  simp only [←BitVec.ofNat_add,BitVec.toNat_ofNat]
  have nonempty : 0 < p.rows*p.cols := Nat.zero_lt_of_lt inside
  have aligned := h.alignment nonempty r
  exact ⟨Nat.mod_eq_of_lt (by omega),by omega⟩

theorem mask (p : Prepared backward) (h : Admitted p) (small : blockSize p.cols ≤ 65536)
    (lane : Nat) (hl : lane < blockSize p.cols) :
    (BitVec.ofNat 32 lane).slt (BitVec.ofNat 32 p.cols) = decide (lane < p.cols) := by
  have columns := (valid p h).cols
  change decide ((BitVec.ofNat 32 lane).toInt < (BitVec.ofNat 32 p.cols).toInt) = _
  rw [VeriTile.Bench.TritonBenchG.SwigluFwd.Whole.signed_ofNat lane (by change lane < 2^31; omega),
    VeriTile.Bench.TritonBenchG.SwigluFwd.Whole.signed_ofNat p.cols (by change p.cols < 2^31; change p.cols ≤ blockSize p.cols at columns; omega)]
  simp only [Int.ofNat_lt]

#axiomsClean valid
#axiomsClean key_binding
#axiomsClean indices
#axiomsClean pointer
#axiomsClean mask
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton.DirectHost
