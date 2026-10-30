import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.Native

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Access
set_option maxHeartbeats 3000000

def address (t : TensorView) (row column : Nat) :=
  t.pointer+4*(row*t.rowStride.toNat+column*t.colStride.toNat)

theorem storage_access (t : TensorView) (valid : TensorBounds t)
    (r c : Nat) (hr : r<t.rows) (hc : c<t.columns) :
    t.storageBase≤address t r c ∧ address t r c+4≤t.storageBase+t.capacity ∧
    address t r c+4≤2^64 ∧ address t r c%4=0 := by
  obtain ⟨_,_,_,_,_,bounds⟩:=valid
  obtain ⟨aligned,lower,upper,word⟩:=bounds ⟨by omega,by omega⟩
  have rows:=Nat.mul_le_mul_right t.rowStride.toNat (show r≤t.rows-1 by omega)
  have cols:=Nat.mul_le_mul_right t.colStride.toNat (show c≤t.columns-1 by omega)
  refine ⟨?_,?_,?_,?_⟩
  · unfold address;omega
  · unfold address;omega
  · unfold address;omega
  · simp [address,Nat.add_mod,aligned]

theorem active_pointer (t : TensorView) (valid : TensorBounds t)
    (n b col pid lane c : Nat) (admitted : LaunchContract n t.columns b col) (hb : 0<b ∧ b≤1024)
    (hp : pid<Native.blocks n b) (hl : lane<b)
    (hr : pid*b+lane<t.rows) (hc : col+c<t.columns) :
    Native.pointer t.pointer b t.rowStride.toNat t.colStride.toNat col pid lane c=
      address t (pid*b+lane) (col+c) := by
  obtain ⟨ps,ls,_⟩:=Native.lane_bounds n b t.columns col pid lane admitted hb hp hl
  apply Native.pointer_exact _ _ _ _ _ _ _ _ ps ls
  have safe:=storage_access t valid _ _ hr hc
  unfold address at safe
  omega

theorem coordinate_address (x : TensorView) (valid : Coordinates x) (r c : Nat) :
    address x r c=x.pointer+4*(r*3+c) := by
  have strides:=valid.2.2.1
  simp only [address,TensorView.rowStride,TensorView.colStride,strides]
  simp

theorem destination_address (x y : TensorView) (col : Int) (valid : Destination x y col)
    (r c : Nat) : address y r c=y.pointer+4*(r*y.columns+c) := by
  have strides:=valid.2.2.2.2.1
  simp only [address,TensorView.rowStride,TensorView.colStride,strides]
  simp

theorem input_output_disjoint (x y : TensorView) (col : Int)
    (xvalid : Coordinates x) (yvalid : Destination x y col)
    (r s c j k l : Nat) (hr : r<x.rows) (hs : s<x.rows)
    (hc : c<3) (hj : j<y.columns) (hk : k<4) (hl : l<4) :
    address x r c+k≠address y s j+l := by
  have sep:=yvalid.2.2.2.2.2.2.2.2 (by omega)
  rw [coordinate_address x xvalid,destination_address x y col yvalid]
  have sx : r*3+c<x.rows*3 := by omega
  have sy : s*y.columns+j<x.rows*y.columns := by
    have h:=Nat.mul_le_mul_right y.columns (Nat.succ_le_of_lt hs)
    rw [Nat.succ_mul] at h
    omega
  rcases sep with sep|sep <;> omega

#axiomsClean storage_access
#axiomsClean active_pointer
#axiomsClean input_output_disjoint
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Access
