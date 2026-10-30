import bench.repairs.tritonbench_g.rope_backward_transform.Contract

namespace VeriTile.Bench.Repairs.RopeBackwardTransform.KernelContract
open VeriTile.Triton
set_option maxHeartbeats 3000000

theorem output_bound (r : Request) (k : Bool) (row head lane : Nat)
    (hr : row<r.rows) (hh : head<r.paddedHeads k) (hl : lane<r.paddedDim/2) :
    row*(r.heads k*r.dimension)+head*r.dimension+lane+r.half≤outputLast r k := by
  have a:=Nat.mul_le_mul_right (r.heads k*r.dimension) (show row≤r.rows-1 by omega)
  have b:=Nat.mul_le_mul_right r.dimension (show head≤r.paddedHeads k-1 by omega)
  have d : lane≤r.paddedDim/2-1 := by omega
  unfold outputLast
  omega

theorem table_bound (r : Request) (k : Bool) (row lane : Nat)
    (hr : row<r.rows) (hl : lane<r.paddedDim/2) :
    row%r.sequence*r.rowStride k+lane*r.colStride k≤tableLast r k := by
  have nonzero : 0<r.sequence := by
    by_contra hn
    have zero : r.sequence=0 := by omega
    simp [Request.rows,zero] at hr
  have hmod:=Nat.mod_lt row nonzero
  have a:=Nat.mul_le_mul_right (r.rowStride k) (show row%r.sequence≤r.sequence-1 by omega)
  have b:=Nat.mul_le_mul_right (r.colStride k) (show lane≤r.paddedDim/2-1 by omega)
  unfold tableLast
  omega

def outputIndex (r : Request) (k second : Bool) (row head lane : Nat) : BitVec 32 :=
  BitVec.ofNat 32 row*BitVec.ofNat 32 (r.heads k*r.dimension)+
    BitVec.ofNat 32 head*BitVec.ofNat 32 r.dimension+BitVec.ofNat 32 lane+
      BitVec.ofNat 32 (if second then r.half else 0)

def tableIndex (r : Request) (k : Bool) (row lane : Nat) : BitVec 32 :=
  BitVec.ofNat 32 (row%r.sequence)*BitVec.ofNat 32 (r.rowStride k)+
    BitVec.ofNat 32 lane*BitVec.ofNat 32 (r.colStride k)

theorem signed32 (n : Nat) (h : n<2^31) : (BitVec.ofNat 32 n).toInt=(n : Int) := by
  rw [BitVec.toInt_eq_toNat_cond]
  simp only [BitVec.toNat_ofNat]
  rw [Nat.mod_eq_of_lt (show n<2^32 by omega),if_pos (show 2*n<2^32 by omega)]

def Access (c : Call) : Prop :=
  launched c.geometry →∀ row,row<c.geometry.rows →∀ k head lane,
    head<c.geometry.paddedHeads k →lane<c.geometry.paddedDim/2 →∀ second : Bool,
    let q:=row*(c.geometry.heads k*c.geometry.dimension)+head*c.geometry.dimension+lane+
      (if second then c.geometry.half else 0)
    let t:=row%c.geometry.sequence*c.geometry.rowStride k+lane*c.geometry.colStride k
    q<2^31 ∧ t<2^31 ∧
    (c.q k).dataPtr.toNat+4*q+4≤2^64 ∧ (c.tables k).dataPtr.toNat+4*t+4≤2^64 ∧
    (outputIndex c.geometry k second row head lane).toInt=(q : Int) ∧
    (tableIndex c.geometry k row lane).toInt=(t : Int)

theorem access (c : Call) (admit : Admitted c) : Access c := by
  intro on row hr k head lane hh hl second
  have high:=output_bound c.geometry k row head lane hr hh hl
  have side : (if second then c.geometry.half else 0)≤c.geometry.half := by
    cases second <;> simp
  have qb : row*(c.geometry.heads k*c.geometry.dimension)+head*c.geometry.dimension+lane+
      (if second then c.geometry.half else 0)≤outputLast c.geometry k := by omega
  have tb:=table_bound c.geometry k row lane hr hl
  obtain ⟨qo,tableOffset,qp,tp⟩:=admit.formedBounds on k
  have qfit : row*(c.geometry.heads k*c.geometry.dimension)+head*c.geometry.dimension+lane+
      (if second then c.geometry.half else 0)<2^31 := by omega
  have tfit : row%c.geometry.sequence*c.geometry.rowStride k+lane*c.geometry.colStride k<2^31 := by omega
  refine ⟨qfit,tfit,by omega,by omega,?_,?_⟩
  · unfold outputIndex
    rw [←BitVec.ofNat_mul,←BitVec.ofNat_mul,←BitVec.ofNat_add,←BitVec.ofNat_add,←BitVec.ofNat_add]
    exact signed32 _ qfit
  · unfold tableIndex
    rw [←BitVec.ofNat_mul,←BitVec.ofNat_mul,←BitVec.ofNat_add]
    exact signed32 _ tfit

#axiomsClean output_bound
#axiomsClean table_bound
#axiomsClean access
end VeriTile.Bench.Repairs.RopeBackwardTransform.KernelContract
