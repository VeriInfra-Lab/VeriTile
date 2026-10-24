import bench.repairs.tritonbench_g.parallel_attention.Proofs.CallContract
import bench.repairs.tritonbench_g.parallel_attention.Proofs.MachineClosure

namespace VeriTile.Bench.Repairs.ParallelAttention.Checked
set_option maxHeartbeats 1200000
open Machine

/-- Separate signed32 head, row and feature offsets, extended before each
64-bit pointer addition; this retains the source's staged base arithmetic. -/
def stagedPointer (base : Nat) (s : Shape) (width head time feature : Nat) : BitVec 64 :=
  BitVec.ofNat 64 base +
    BitVec.ofInt 64 ((BitVec.ofNat 32 head*BitVec.ofNat 32 (s.T*width)).toInt)*4 +
    BitVec.ofInt 64 ((BitVec.ofNat 32 time*BitVec.ofNat 32 width).toInt)*4 +
    BitVec.ofInt 64 ((BitVec.ofNat 32 feature).toInt)*4

theorem stagedPointer_eq (base : Nat) (s : Shape) (g : Guard s)
    (width head time feature : Nat) (hp : 0<width) (hw : width≤largestWidth s)
    (hh : head<shardHeads s) (ht : time<s.T+256) (hf : feature<width+128) :
    stagedPointer base s width head time feature=BitVec.ofNat 64 (base+4*formedOffset s width head time feature) := by
  obtain ⟨hhead,hheadWidth,htime,hfeature,hpartial,hfull⟩:=formed_intermediates s g width head time feature hp hw hh ht hf
  have headEq : Nonwrap32 (head*(s.T*width)) := by simpa [Nat.mul_assoc] using hheadWidth
  unfold stagedPointer
  rw [signed_mul head (s.T*width) headEq,signed_mul time width htime,signed32_ofNat feature hfeature]
  simp only [BitVec.ofInt_natCast]
  have eq : base+head*(s.T*width)*4+time*width*4+feature*4=base+4*formedOffset s width head time feature := by
    unfold formedOffset
    ring
  change BitVec.ofNat 64 base+BitVec.ofNat 64 (head*(s.T*width))*BitVec.ofNat 64 4+
    BitVec.ofNat 64 (time*width)*BitVec.ofNat 64 4+BitVec.ofNat 64 feature*BitVec.ofNat 64 4=_
  rw [←BitVec.ofNat_mul,←BitVec.ofNat_mul,←BitVec.ofNat_mul,
    ←BitVec.ofNat_add,←BitVec.ofNat_add,←BitVec.ofNat_add,eq]

theorem stagedPointer_actual (c : Call) (d : Domain c) (active : Active c.shape)
    (t : TensorView) (which : t=c.q ∨ t=c.k ∨ t=c.v)
    (width head time feature : Nat) (hp : 0<width) (hw : width≤largestWidth c.shape)
    (hh : head<shardHeads c.shape) (ht : time<c.shape.T+256) (hf : feature<width+128) :
    (stagedPointer t.pointer c.shape width head time feature).toNat=
      t.pointer+4*dense c.shape.T width head time feature := by
  rw [stagedPointer_eq t.pointer c.shape (admitted_active c.shape d.admission active)
    width head time feature hp hw hh ht hf]
  have limit:=pointer_room c d active t which (formedOffset c.shape width head time feature)
    (formed_offset c.shape width head time feature hw hh ht hf)
  rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega),graph_load_address]

/-- Actual active storage access, including an arbitrary storage offset. -/
theorem stored_lane (t : TensorView) (s : Shape) (width bh time feature : Nat)
    (bound : Stored t (s.B*s.H*s.T*width))
    (hb : bh<s.B*s.H) (ht : time<s.T) (he : feature<width) (byte : Fin 4) :
    t.storageBase≤4*(t.pointer/4+dense s.T width bh time feature)+byte.val ∧
    4*(t.pointer/4+dense s.T width bh time feature)+byte.val<t.storageBase+t.capacity := by
  have ⟨addr,lo,hi⟩:=stored_address t _ _ bound (dense_bound _ _ _ _ _ _ hb ht he)
  have hk:=byte.isLt
  rw [addr]
  omega

/-- Fresh Torch scratch/output allocations have ordinary signed-address
storage bounds. The padded formed offset still cannot wrap a 64-bit pointer,
even though inactive lanes need not belong to that allocation. -/
theorem allocation_pointer (base : Nat) (s : Shape) (g : Guard s) (baseBound : base≤2^63)
    (width head time feature : Nat) (hp : 0<width) (hw : width≤largestWidth s)
    (hh : head<shardHeads s) (ht : time<s.T+256) (hf : feature<width+128) :
    (stagedPointer base s width head time feature).toNat=base+4*formedOffset s width head time feature := by
  rw [stagedPointer_eq base s g width head time feature hp hw hh ht hf]
  have bound:=formed_machine s g width head time feature hw hh ht hf
  unfold Nonwrap32 at bound
  rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega)]

def storageOffset (t : TensorView) (b h row e : Nat) : Int :=
  match t.strides with
  | [sb,sh,st,se] => (b:Int)*sb+(h:Int)*sh+(row:Int)*st+(e:Int)*se
  | _ => 0

private theorem singleton_product (n i : Nat) (stride expected : Int)
    (hi : i<n) (bound : n≤1 ∨ stride=expected) : (i:Int)*stride=(i:Int)*expected := by
  rcases bound with h|rfl
  · have eq:i=0:=by omega
    simp [eq]
  · rfl

/-- The checked actual strides, including arbitrary singleton strides, give
exactly the canonical address passed to all four source kernels. -/
theorem dense_actual_strides (s : Shape) (width : Nat) (t : TensorView)
    (strides : DenseStrides s width t) (b h row e : Nat)
    (hb : b<s.B) (hh : h<s.H) (hr : row<s.T) (he : e<width) :
    storageOffset t b h row e=(dense s.T width (b*s.H+h) row e : Nat) := by
  have ne : ¬Machine.Empty s := by unfold Machine.Empty;omega
  simp only [DenseStrides,if_neg (show ¬(s.B=0 ∨ s.H=0 ∨ s.T=0) from ne)] at strides
  unfold storageOffset
  split at strides <;> simp_all only
  · obtain ⟨sb,sh,st,se⟩:=strides
    rw [singleton_product s.B b _ _ hb sb,singleton_product s.H h _ _ hh sh,
      singleton_product s.T row _ _ hr st,singleton_product width e _ _ he se]
    simp only [dense,Nat.cast_add,Nat.cast_mul]
    ring

#axiomsClean allocation_pointer
#axiomsClean stagedPointer_eq
#axiomsClean stagedPointer_actual
#axiomsClean stored_lane
#axiomsClean dense_actual_strides
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ParallelAttention.Checked
