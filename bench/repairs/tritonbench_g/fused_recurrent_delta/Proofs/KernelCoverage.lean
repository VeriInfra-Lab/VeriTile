import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelGrid
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelCoverage
open VeriTile.Triton KernelProgram KernelRegisters KernelBackwardRegisters
noncomputable section
set_option maxHeartbeats 10000000

theorem forward_values (s : Shape) (f : Flags) (bh : Fin s.BH) (t : Fin s.T) (v : Fin s.V) :
    ∃ p : Program .forward s,
      KernelGeometry.Writes .forward s f p "o" ((bh.val*s.T+t.val)*s.V+v.val) ∧
      KernelGeometry.Writes .forward s f p "v" ((bh.val*s.T+t.val)*s.V+v.val) := by
  obtain ⟨p,j,head,col⟩ := KernelGeometry.value_owner .forward s bh.val v.val bh.isLt v.isLt
  have active : p.value*s.FBV+j.val<s.V := by simpa only [←col] using v.isLt
  have address : (bh.val*s.T+t.val)*s.V+v.val=valAddress s p.head p.value s.FBV t.val j.val := by
    rw [←head,←col]
    simp only [blockV,ite_true,valAddress]
    ring
  exact ⟨p,Or.inl ⟨Or.inr rfl,t.val,j,t.isLt,active,address⟩,
    Or.inl ⟨Or.inl rfl,t.val,j,t.isLt,active,address⟩⟩
theorem forward_state (s : Shape) (f : Flags) (enabled : f.final=Bool.true)
    (bh : Fin s.BH) (i : Fin s.K) (v : Fin s.V) :
    ∃ p : Program .forward s,KernelGeometry.Writes .forward s f p "ht" ((bh.val*s.K+i.val)*s.V+v.val) := by
  obtain ⟨p,j,head,col⟩ := KernelGeometry.value_owner .forward s bh.val v.val bh.isLt v.isLt
  let wide : Fin s.BK := ⟨i.val,lt_of_lt_of_le i.isLt (Geometry.nextPower_ge s.K)⟩
  refine ⟨p,Or.inr ⟨enabled,rfl,wide,j,i.isLt,by simpa only [←col] using v.isLt,?_⟩⟩
  rw [←head,←col]
  simp only [blockV,ite_true,stateAddress,wide]
  ring
theorem backward_value (s : Shape) (f : Flags) (bh : Fin s.BH) (t : Fin s.T) (v : Fin s.V) :
    ∃ p : Program .backward s,KernelGeometry.Writes .backward s f p "dv" ((bh.val*s.T+t.val)*s.V+v.val) := by
  obtain ⟨p,j,head,col⟩ := KernelGeometry.value_owner .backward s bh.val v.val bh.isLt v.isLt
  refine ⟨p,Or.inr (Or.inl ⟨rfl,t.val,j,t.isLt,by simpa only [←col] using v.isLt,?_⟩)⟩
  rw [←head,←col]
  simp [blockV,valAddress]
  ring
theorem backward_state (s : Shape) (f : Flags) (enabled : f.initialGrad=Bool.true)
    (bh : Fin s.BH) (i : Fin s.K) (v : Fin s.V) :
    ∃ p : Program .backward s,KernelGeometry.Writes .backward s f p "dh0" ((bh.val*s.K+i.val)*s.V+v.val) := by
  obtain ⟨p,j,head,col⟩ := KernelGeometry.value_owner .backward s bh.val v.val bh.isLt v.isLt
  let wide : Fin s.BK := ⟨i.val,lt_of_lt_of_le i.isLt (Geometry.nextPower_ge s.K)⟩
  refine ⟨p,Or.inr (Or.inr (Or.inr ⟨enabled,rfl,wide,j,i.isLt,by simpa only [←col] using v.isLt,?_⟩))⟩
  rw [←head,←col]
  simp [blockV,stateAddress,wide]
  ring
theorem backward_keys (s : Shape) (f : Flags) (bh : Fin s.BH) (iv : Fin s.BNV)
    (t : Fin s.T) (i : Fin s.K) :
    ∃ p : Program .backward s,
      KernelGeometry.Writes .backward s f p "dq" (((iv.val*s.BH+bh.val)*s.T+t.val)*s.K+i.val) ∧
      KernelGeometry.Writes .backward s f p "dk" (((iv.val*s.BH+bh.val)*s.T+t.val)*s.K+i.val) := by
  let p : Program .backward s := ⟨bh.val,iv.val,bh.isLt,iv.isLt⟩
  let wide : Fin s.BK := ⟨i.val,lt_of_lt_of_le i.isLt (Geometry.nextPower_ge s.K)⟩
  have address : ((iv.val*s.BH+bh.val)*s.T+t.val)*s.K+i.val=keyAddress s (plane s p.head p.value) t.val wide.val := by
    simp only [keyAddress,plane,p,wide,Shape.BH]
    ring
  exact ⟨p,Or.inl ⟨Or.inr rfl,t.val,wide,t.isLt,i.isLt,address⟩,
    Or.inl ⟨Or.inl rfl,t.val,wide,t.isLt,i.isLt,address⟩⟩
theorem scalar_beta (s : Shape) (f : Flags) (scalar : f.vectorBeta=Bool.false)
    (bh : Fin s.BH) (iv : Fin s.BNV) (t : Fin s.T) :
    ∃ p : Program .backward s,KernelGeometry.Writes .backward s f p "dbeta" ((iv.val*s.BH+bh.val)*s.T+t.val) := by
  let p : Program .backward s := ⟨bh.val,iv.val,bh.isLt,iv.isLt⟩
  refine ⟨p,Or.inr (Or.inr (Or.inl ⟨rfl,?_⟩))⟩
  simp only [scalar,Bool.false_eq_true,ite_false]
  refine ⟨t.val,t.isLt,?_⟩
  simp only [plane,p,Shape.BH]
  ring
theorem vector_beta (s : Shape) (f : Flags) (vector : f.vectorBeta=Bool.true)
    (bh : Fin s.BH) (t : Fin s.T) (v : Fin s.V) :
    ∃ p : Program .backward s,KernelGeometry.Writes .backward s f p "dbeta"
      ((((v.val/s.BBV)*s.BH+bh.val)*s.T+t.val)*s.V+v.val) := by
  obtain ⟨inside,lane,coord⟩ := Geometry.owner_exists s.V s.BBV v.val (Geometry.tile_pos s).2.2 v.isLt
  let p : Program .backward s := ⟨bh.val,v.val/s.BBV,bh.isLt,inside⟩
  let j : Fin s.BBV := ⟨v.val%s.BBV,lane⟩
  refine ⟨p,Or.inr (Or.inr (Or.inl ⟨rfl,?_⟩))⟩
  simp only [vector,ite_true]
  refine ⟨t.val,j,t.isLt,by simpa only [p,j,coord] using v.isLt,?_⟩
  simp only [valAddress,plane,p,j,Shape.BH]
  calc
    _ = (((v.val/s.BBV)*(s.B*s.H)+bh.val)*s.T+t.val)*s.V+
        (v.val/s.BBV*s.BBV+v.val%s.BBV) := by omega
    _ = _ := by ring

#axiomsClean forward_values
#axiomsClean forward_state
#axiomsClean backward_value
#axiomsClean backward_state
#axiomsClean backward_keys
#axiomsClean scalar_beta
#axiomsClean vector_beta
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelCoverage
