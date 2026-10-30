import bench.tritonbench_g.fused_recurrent_delta.GeometrySource
import bench.tritonbench_g.fused_recurrent_delta.Proofs.Settings
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Geometry
noncomputable section
set_option maxHeartbeats 5000000
theorem nextPower_ge (n : Nat) : n≤n.nextPowerOfTwo := Settings.nextPower_covers n
theorem nextPower_pos (n : Nat) : 0<n.nextPowerOfTwo :=
  Nat.pos_of_isPowerOfTwo (Nat.isPowerOfTwo_nextPowerOfTwo n)
theorem tile_pos (s : Shape) : 0<s.BK ∧ 0<s.FBV ∧ 0<s.BBV := by
  exact ⟨nextPower_pos _,lt_min (nextPower_pos _) (by decide),lt_min (nextPower_pos _) (by decide)⟩
theorem ceil_eq (V BV : Nat) (hV : 0<V) (hBV : 0<BV) : ceil V BV=(V-1)/BV+1 := by
  unfold ceil
  have hh : V+BV-1=(V-1)+BV := by omega
  rw [hh,Nat.add_div_right _ hBV]
theorem owner_exists (V BV v : Nat) (hBV : 0<BV) (hv : v<V) :
    v/BV<ceil V BV ∧ v%BV<BV ∧ (v/BV)*BV+v%BV=v := by
  rw [ceil_eq V BV (by omega) hBV]
  constructor
  · have hd : v/BV≤(V-1)/BV := Nat.div_le_div_right (show v≤V-1 by omega)
    omega
  · exact ⟨Nat.mod_lt _ hBV,by simpa [Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div v BV⟩
theorem owner_unique (BV iv v : Nat) (lo : iv*BV≤v) (hi : v<(iv+1)*BV) : iv=v/BV :=
  (Nat.div_eq_of_lt_le lo hi).symm
theorem masked_owner (V BV iv j : Nat) (hBV : 0<BV) (hj : j<BV) (hv : iv*BV+j<V) :
    iv<ceil V BV ∧ (iv*BV+j)/BV=iv ∧ (iv*BV+j)%BV=j := by
  have hdiv : (iv*BV+j)/BV=iv := Nat.div_eq_of_lt_le (by omega) (by rw [Nat.add_mul];omega)
  have hmod : (iv*BV+j)%BV=j := by simp [Nat.add_mod,Nat.mod_eq_of_lt hj]
  exact ⟨by simpa [hdiv] using (owner_exists V BV (iv*BV+j) hBV hv).1,hdiv,hmod⟩
theorem nk_one (s : Shape) (hK : 0<s.K) : ceil s.K s.BK=1 := by
  rw [ceil_eq s.K s.BK hK (tile_pos s).1]
  have hk:=nextPower_ge s.K
  rw [Nat.div_eq_of_lt (show s.K-1<s.BK by dsimp [Shape.BK];omega)]

theorem mixed_injective (C a b i j : Nat) (hi : i<C) (hj : j<C)
    (h : a*C+i=b*C+j) : a=b ∧ i=j := by
  have modEq:=congrArg (fun n => n%C) h
  have hij : i=j := by simpa [Nat.add_mod,Nat.mod_eq_of_lt hi,Nat.mod_eq_of_lt hj] using modEq
  subst j
  have mulEq : a*C=b*C := by omega
  exact ⟨Nat.eq_of_mul_eq_mul_right (by omega) mulEq,rfl⟩
theorem offset_injective (s : Shape) (f : Flags) (a b : Addr) (ha : Valid s f a)
    (hb : Valid s f b) (hr : a.region=b.region) (hoff : offset s f a=offset s f b) : a=b := by
  rcases a with ⟨r,ap,arow,aj⟩
  rcases b with ⟨r',bp,bt,bj⟩
  dsimp at hr
  subst r'
  obtain ⟨hrow,hj⟩:=mixed_injective _ _ _ _ _ ha.2.2 hb.2.2 hoff
  obtain ⟨hp,ht⟩:=mixed_injective _ _ _ _ _ ha.2.1 hb.2.1 hrow
  simp_all
theorem mixed_bound (P C a i : Nat) (ha : a<P) (hi : i<C) : a*C+i<P*C := by
  calc
    _ < a*C+C := Nat.add_lt_add_left hi _
    _ = (a+1)*C := by rw [Nat.add_mul];simp
    _ ≤ P*C := Nat.mul_le_mul_right _ (by omega)
theorem offset_bound (s : Shape) (f : Flags) (a : Addr) (h : Valid s f a) :
    offset s f a<capacity s f a.region :=
  mixed_bound _ _ _ _ (mixed_bound _ _ _ _ h.1 h.2.1) h.2.2

/-- Exact repaired vector-beta source expression, NK=1 and dense s_vo_h=T*V. -/
theorem vector_beta_pointer (s : Shape) (f : Flags) (hf : f.vectorBeta=Bool.true)
    (bh iv t j : Nat) :
    offset s f ⟨.dbeta,iv*s.BH+bh,t,iv*s.BBV+j⟩=
      (bh+0*s.B*s.H+iv*s.B*s.H*1)*(s.T*s.V)+iv*s.BBV+j+t*s.V := by
  simp [offset,rows,cols,hf,Shape.BH]
  ring
theorem key_scratch_pointer (s : Shape) (f : Flags) (bh iv t i : Nat) :
    offset s f ⟨.dk,iv*s.BH+bh,t,i⟩=(bh+iv*s.B*s.H)*(s.T*s.K)+i+t*s.K := by
  simp [offset,rows,cols,Shape.BH]
  ring
theorem state_pointer (s : Shape) (f : Flags) (bh iv BV i j : Nat) :
    offset s f ⟨.h0,bh,i,iv*BV+j⟩=bh*s.K*s.V+i*s.V+iv*BV+j := by
  simp [offset,rows,cols]
  ring

#axiomsClean nextPower_ge
#axiomsClean owner_exists
#axiomsClean masked_owner
#axiomsClean nk_one
#axiomsClean offset_injective
#axiomsClean offset_bound
#axiomsClean vector_beta_pointer
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Geometry
