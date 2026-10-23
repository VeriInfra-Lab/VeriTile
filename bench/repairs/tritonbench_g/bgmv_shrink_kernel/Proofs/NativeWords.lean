import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Storage

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.NativeWords
open VeriTile.Triton Geometry Storage

/-- Facts of emitted CUDA PID/arange values. Generic grid admission and PID
compatibility remain the separately recorded runtime obligation. -/
structure ABI (h : Call) (p : Program h) : Prop where
  batch : p.batch.val<2147483648
  lanes : h.BN≤2147483648

def rankWord (p : Program h) (c : Nat) (e : Fin 256) : BitVec 64 :=
  BitVec.ofNat 64 (c*16384)+(BitVec.ofInt 64 (BitVec.ofNat 32 e.val).toInt+
    BitVec.ofInt 64 (BitVec.ofNat 32 p.split.val).toInt*BitVec.ofNat 64 256)

def inputWord (h : Call) (p : Program h) (c : Nat) (e : Fin 256) : BitVec 64 :=
  BitVec.ofInt 64 (BitVec.ofNat 32 p.batch.val).toInt*BitVec.ofNat 64 h.paddedK+rankWord p c e

def weightWord (h : Call) (p : Program h) (slot : Nat) (c : Nat) (e : Fin 256) (n : Fin h.BN) : BitVec 64 :=
  BitVec.ofNat 64 (h.BN*h.paddedK)*BitVec.ofNat 64 slot+
    BitVec.ofInt 64 (BitVec.ofNat 32 n.val).toInt*BitVec.ofNat 64 h.paddedK+rankWord p c e

def outputWord (h : Call) (p : Program h) (n : Fin h.BN) : BitVec 64 :=
  BitVec.ofInt 64 (BitVec.ofNat 32 p.batch.val).toInt*BitVec.ofNat 64 h.BN+
    BitVec.ofInt 64 (BitVec.ofNat 32 n.val).toInt

theorem offsets (h : Call) (p : Program h) (abi : ABI h p) (slot c : Nat) (e : Fin 256) (n : Fin h.BN) :
    inputWord h p c e=BitVec.ofNat 64 (inputOffset h p.batch.val (rank p.split.val c e.val)) ∧
    weightWord h p slot c e n=BitVec.ofNat 64 (weightOffset h slot n.val (rank p.split.val c e.val)) ∧
    outputWord h p n=BitVec.ofNat 64 (outputOffset h p.batch.val n.val) := by
  have es : e.val<2147483648 := by have := e.isLt; omega
  have ps : p.split.val<2147483648 := by have := p.split.isLt; omega
  have ns := n.isLt.trans_le abi.lanes
  simp only [inputWord,weightWord,outputWord,rankWord,widen32 _ abi.batch,
    widen32 _ es,widen32 _ ps,widen32 _ ns,inputOffset,weightOffset,outputOffset,rank,
    BitVec.ofNat_add,BitVec.ofNat_mul]
  simp only [BitVec.mul_comm,true_and,and_true]

/-- Exact natural source expressions, signed64 values and allocation-backed
pointer words are linked for the same padded allocations and all lanes. -/
theorem linked (h : Call) (s : Placement h) (p : Program h) (abi : ABI h p)
    (slot : Fin h.adapters) (c : Fin h.cycles) (e : Fin 256) (n : Fin h.BN) :
    (inputWord h p c.val e).toInt=(inputOffset h p.batch.val (rank p.split.val c.val e.val):Int) ∧
    (weightWord h p slot.val c.val e n).toInt=(weightOffset h slot.val n.val (rank p.split.val c.val e.val):Int) ∧
    (outputWord h p n).toInt=(outputOffset h p.batch.val n.val:Int) := by
  obtain ⟨x,w,y⟩ := offsets h p abi slot.val c.val e n
  rw [x,w,y]
  obtain ⟨x,w,y,_⟩ := native_addresses h s p slot c e n
  exact ⟨x,w,y⟩

#axiomsClean offsets
#axiomsClean linked
end VeriTile.Bench.Repairs.BgmvShrinkKernel.NativeWords
