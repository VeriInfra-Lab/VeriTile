import bench.repairs.tritonbench_g.rope_transform.Contract

namespace VeriTile.Bench.Repairs.RopeTransform.KernelContract
open VeriTile.Triton
set_option maxHeartbeats 3000000

def outputIndex (r : Request) (key side : Bool) (row head lane : Nat) : BitVec 32 :=
  BitVec.ofNat 32 row*BitVec.ofNat 32 (r.stride key)+BitVec.ofNat 32 head*BitVec.ofNat 32 r.dimension+
    BitVec.ofNat 32 lane+BitVec.ofNat 32 (if side then r.half else 0)
def tableIndex (r : Request) (row lane : Nat) : BitVec 32 :=
  BitVec.ofNat 32 (row%r.sequence)*BitVec.ofNat 32 r.half+BitVec.ofNat 32 lane

def Access (c : Call) : Prop :=
  ∀ row,row<c.geometry.rows →∀ k side head lane,head<c.geometry.paddedHeads k →lane<c.geometry.width →
    let q:=c.geometry.offset k row head (c.geometry.column side lane)
    let t:=row%c.geometry.sequence*c.geometry.half+lane
    q<2^31 ∧ t<2^31 ∧ (c.q k).dataPtr.toNat+4*q+4≤2^64 ∧
    (c.tables k).dataPtr.toNat+4*t+4≤2^64 ∧
    (outputIndex c.geometry k side row head lane).toInt=(q : Int) ∧
    (tableIndex c.geometry row lane).toInt=(t : Int)

theorem signed32 (n : Nat) (h : n<2^31) : (BitVec.ofNat 32 n).toInt=(n : Int) := by
  rw [BitVec.toInt_eq_toNat_cond]
  simp only [BitVec.toNat_ofNat]
  rw [Nat.mod_eq_of_lt (show n<2^32 by omega),if_pos (show 2*n<2^32 by omega)]

theorem access (c : Call) (admit : Admitted c) : Access c := by
  intro row hr k side head lane hh hl
  have a:=Nat.mul_le_mul_right (c.geometry.stride k) (show row≤c.geometry.rows-1 by omega)
  have b:=Nat.mul_le_mul_right c.geometry.dimension (show head≤c.geometry.paddedHeads k-1 by omega)
  have small : (if side then c.geometry.half else 0)≤c.geometry.half := by cases side <;> simp
  have qb : c.geometry.offset k row head (c.geometry.column side lane)≤outputLast c.geometry k := by
    unfold Request.offset Request.column outputLast; omega
  have seq : 0<c.geometry.sequence := by
    by_contra h
    have zero : c.geometry.sequence=0 := by omega
    simp [Request.rows,zero] at hr
  have hm:=Nat.mod_lt row seq
  have p:=Nat.mul_le_mul_right c.geometry.half (show row%c.geometry.sequence≤c.geometry.sequence-1 by omega)
  have tb : row%c.geometry.sequence*c.geometry.half+lane≤tableLast c.geometry := by unfold tableLast; omega
  obtain ⟨qo,tt,qp,tp⟩:=admit.formedBounds (by omega) k
  have qfit : c.geometry.offset k row head (c.geometry.column side lane)<2^31 := by omega
  have tfit : row%c.geometry.sequence*c.geometry.half+lane<2^31 := by omega
  refine ⟨qfit,tfit,by omega,by omega,?_,?_⟩
  · unfold outputIndex
    rw [←BitVec.ofNat_mul,←BitVec.ofNat_mul,←BitVec.ofNat_add,←BitVec.ofNat_add,←BitVec.ofNat_add]
    simpa [Request.offset,Request.column,Nat.add_assoc] using signed32 _ qfit
  · unfold tableIndex
    rw [←BitVec.ofNat_mul,←BitVec.ofNat_add]
    exact signed32 _ tfit

#axiomsClean access
end VeriTile.Bench.Repairs.RopeTransform.KernelContract
