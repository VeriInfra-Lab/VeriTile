import bench.repairs.tritonbench_g.geglu_tanh_triton.Contract
import bench.repairs.tritonbench_g.geglu_tanh_triton.Proofs.Grid

namespace VeriTile.Bench.Repairs.GegluTanhTriton.DirectHost
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Physical
noncomputable section
set_option maxHeartbeats 2000000

theorem check_iff (p : Prepared backward) : check p = Bool.true ↔ Admitted p := by
  simp only [check,Bool.and_eq_true,decide_eq_true_eq]
  exact ⟨fun ⟨⟨⟨⟨⟨⟨⟨⟨a,b⟩,c⟩,d⟩,e⟩,f⟩,g⟩,i⟩,j⟩ => ⟨a,b,c,d,e,f,g,i,j⟩,
    fun h => ⟨⟨⟨⟨⟨⟨⟨⟨h.payloads,h.counts⟩,h.forwardShape⟩,h.sameDevice⟩,h.positive⟩,h.grid32⟩,h.padded64⟩,h.alignment⟩,h.separate⟩⟩

theorem strided (p : Prepared backward) (h : Admitted p) (r : Role) :
    (p.tensor r).metadata.isStrided = Bool.true := by
  have hp := h.payloads r
  simp only [payload,SwigluTriton.DirectHost.payload,Bool.and_eq_true,Bool.not_eq_true'] at hp
  exact hp.1.1.1.1

theorem fp32 (p : Prepared backward) (h : Admitted p) (r : Role) :
    (p.tensor r).metadata.isFP32 = Bool.true := by
  have hp := h.payloads r
  simp only [payload,SwigluTriton.DirectHost.payload,Bool.and_eq_true,Bool.not_eq_true'] at hp
  exact hp.1.1.1.2

theorem contiguous (p : Prepared backward) (h : Admitted p) (r : Role) :
    (p.tensor r).metadata.isContiguous = Bool.true := by
  have hp := h.payloads r
  simp only [payload,SwigluTriton.DirectHost.payload,Bool.and_eq_true,Bool.not_eq_true'] at hp
  exact hp.1.2

def allocation (p : Prepared backward) (h : Admitted p) (r : Role) : Allocation :=
  ⟨(p.tensor r).object,(p.tensor r).metadata.device.index,(p.tensor r).metadata.dataPtr.toNat,
    4*(p.tensor r).metadata.numel,(p.tensor r).contiguousSpan (strided p h r) (fp32 p h r) (contiguous p h r)⟩

def layout (p : Prepared backward) (h : Admitted p) : Layout :=
  ⟨allocation p h,p.rows*p.cols⟩

theorem valid (p : Prepared backward) (h : Admitted p) : Valid (layout p h) where
  capacity := by intro r; simp [layout,allocation,Allocation.words,h.counts]
  separate := by
    intro r s ne
    by_cases nonempty : 0 < p.rows*p.cols
    · exact Or.inr (h.separate nonempty r s ne)
    · have zero : p.rows*p.cols = 0 := by omega
      simp only [Allocation.separate,layout,allocation,h.counts,zero,Nat.mul_zero,Nat.add_zero]
      exact Or.inr (Nat.le_total _ _)

theorem key_binding (p : Prepared backward) (h : Admitted p) (r : Role) (i : Nat)
    (hi : i < p.rows*p.cols) : (layout p h).key r i = (p.tensor r).key i := by
  rw [(p.tensor r).contiguousKeys (strided p h r) (fp32 p h r) (contiguous p h r) _ (by rwa [h.counts])]
  rfl

theorem signed64 (n : Nat) (hn : n < 2^63) : (BitVec.ofNat 64 n).toInt = (n : Int) := by
  have value : (BitVec.ofNat 64 n).toNat = n := by rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega)]
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [value]; omega),value]

theorem indices (p : Prepared backward) (h : Admitted p) (pid lane : Nat)
    (hp : pid < Grid.grid (p.rows*p.cols)) (hl : lane < 128) :
    (BitVec.ofNat 64 pid * BitVec.ofNat 64 128 + BitVec.ofNat 64 lane).toInt = (pid*128+lane : Nat) ∧
    (BitVec.ofNat 64 pid * BitVec.ofNat 64 128 + BitVec.ofNat 64 lane).slt
      (BitVec.ofNat 64 (p.rows*p.cols)) = decide (pid*128+lane < p.rows*p.cols) := by
  have padded := h.padded64
  have count : p.rows*p.cols < 2^63 := by omega
  have laneBound : pid*128+lane < 2^63 := by dsimp [Grid.grid] at hp; omega
  have exactIndex : (BitVec.ofNat 64 pid * BitVec.ofNat 64 128 + BitVec.ofNat 64 lane).toInt = (pid*128+lane : Nat) := by
    rw [←BitVec.ofNat_mul,←BitVec.ofNat_add]
    exact signed64 _ laneBound
  refine ⟨exactIndex,?_⟩
  change decide (_ < (BitVec.ofNat 64 (p.rows*p.cols)).toInt) = _
  rw [exactIndex,signed64 _ count]
  simp only [Int.ofNat_lt]

theorem pointer (p : Prepared backward) (h : Admitted p) (r : Role) (i : Nat)
    (hi : i < p.rows*p.cols) :
    (BitVec.ofNat 64 (p.tensor r).metadata.dataPtr.toNat + BitVec.ofNat 64 (4*i)).toNat =
      ((p.tensor r).key i).address ∧ ((p.tensor r).key i).address % 4 = 0 := by
  have bounds := (p.tensor r).contiguousSpan (strided p h r) (fp32 p h r) (contiguous p h r)
  rw [h.counts] at bounds
  rw [(p.tensor r).contiguousKeys (strided p h r) (fp32 p h r) (contiguous p h r) _ (by rwa [h.counts])]
  simp only [←BitVec.ofNat_add,BitVec.toNat_ofNat]
  have aligned := h.alignment (Nat.zero_lt_of_lt hi) r
  exact ⟨Nat.mod_eq_of_lt (by omega),by omega⟩

#axiomsClean valid
#axiomsClean indices
#axiomsClean pointer
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.GegluTanhTriton.DirectHost
