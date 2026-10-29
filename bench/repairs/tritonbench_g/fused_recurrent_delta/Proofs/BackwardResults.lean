import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.BackwardGrid
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.BackwardResults
noncomputable section
set_option maxHeartbeats 8000000
def mid {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) (bh iv : Nat) :=
  if f.initialGrad then stateStore s p (Reverse.memory s f p g scale m bh iv s.T) .dh0 bh iv s.BBV
      (Reverse.h s f p g scale m bh iv s.T)
    else Reverse.memory s f p g scale m bh iv s.T
theorem dq_value {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees)
    (hg : (g.correction s.BBV).Legal s.BBV) (scale : α) (m : Memory α) (bh iv t k : Nat)
    (hb : bh<s.BH) (hi : iv<s.BNV) (ht : t<s.T) (hk : k<s.K) :
    backwardGrid s f p g scale m ⟨.dq,iv*s.BH+bh,t,k⟩=
      p.cut (Recompute.output s f p g scale (mid s f p g scale m bh iv) bh iv t k) := by
  have hprot : BackwardRelation.Protected s (bh,iv) ⟨.dq,iv*s.BH+bh,t,k⟩ := rfl
  rw [BackwardGrid.grid_readback s f p g hg scale m bh iv hb hi _ hprot,WholeSource.backward_program]
  simp [WholeSource.backwardReference,Recompute.memory,mid,ht,hk]
theorem dk_value {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees)
    (hg : (g.correction s.BBV).Legal s.BBV) (scale : α) (m : Memory α) (bh iv t k : Nat)
    (hb : bh<s.BH) (hi : iv<s.BNV) (ht : t<s.T) (hk : k<s.K) :
    backwardGrid s f p g scale m ⟨.dk,iv*s.BH+bh,t,k⟩=
      p.cut (Recompute.corrected s f p g (mid s f p g scale m bh iv) bh iv t k) := by
  have hprot : BackwardRelation.Protected s (bh,iv) ⟨.dk,iv*s.BH+bh,t,k⟩ := rfl
  rw [BackwardGrid.grid_readback s f p g hg scale m bh iv hb hi _ hprot,WholeSource.backward_program]
  simp [WholeSource.backwardReference,Recompute.memory,mid,ht,hk]
theorem dv_value {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees)
    (hg : (g.correction s.BBV).Legal s.BBV) (scale : α) (m : Memory α) (bh iv t j : Nat)
    (hb : bh<s.BH) (hi : iv<s.BNV) (ht : t<s.T) (hj : j<s.BBV) (hv : iv*s.BBV+j<s.V) :
    backwardGrid s f p g scale m ⟨.dv,bh,t,iv*s.BBV+j⟩=p.cut (Reverse.dv s f p g scale m bh iv t j) := by
  have hprot : BackwardRelation.Protected s (bh,iv) ⟨.dv,bh,t,iv*s.BBV+j⟩ :=
    ⟨rfl,by dsimp;omega,by dsimp;rw [Nat.add_mul];omega⟩
  rw [BackwardGrid.grid_readback s f p g hg scale m bh iv hb hi _ hprot,WholeSource.backward_program]
  have high : iv*s.BBV+j<(iv+1)*s.BBV := by rw [Nat.add_mul];omega
  simp [WholeSource.backwardReference,Recompute.memory,Reverse.memory,stateStore,write,ite_apply,ht,hv,high]
theorem dbeta_value {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees)
    (hg : (g.correction s.BBV).Legal s.BBV) (scale : α) (m : Memory α) (bh iv t v : Nat)
    (hb : bh<s.BH) (hi : iv<s.BNV) (ht : t<s.T) (hf : f.vectorBeta=Bool.true) :
    backwardGrid s f p g scale m ⟨.dbeta,iv*s.BH+bh,t,v⟩=
      if iv*s.BBV≤v ∧ v<min ((iv+1)*s.BBV) s.V then p.cut (Reverse.dbeta s f p g scale m bh iv t (v-iv*s.BBV))
      else m ⟨.dbeta,iv*s.BH+bh,t,v⟩ := by
  have hprot : BackwardRelation.Protected s (bh,iv) ⟨.dbeta,iv*s.BH+bh,t,v⟩ := rfl
  rw [BackwardGrid.grid_readback s f p g hg scale m bh iv hb hi _ hprot,WholeSource.backward_program]
  simp [WholeSource.backwardReference,Recompute.memory,Reverse.memory,stateStore,write,ite_apply,ht,hf]
theorem dbeta_scalar_value {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees)
    (hg : (g.correction s.BBV).Legal s.BBV) (scale : α) (m : Memory α) (bh iv t : Nat)
    (hb : bh<s.BH) (hi : iv<s.BNV) (ht : t<s.T) (hf : f.vectorBeta=Bool.false) :
    backwardGrid s f p g scale m ⟨.dbeta,iv*s.BH+bh,t,0⟩=
      p.cut ((g.beta s.BBV).eval p (fun j => Reverse.dbeta s f p g scale m bh iv t j)) := by
  have hprot : BackwardRelation.Protected s (bh,iv) ⟨.dbeta,iv*s.BH+bh,t,0⟩ := rfl
  rw [BackwardGrid.grid_readback s f p g hg scale m bh iv hb hi _ hprot,WholeSource.backward_program]
  simp [WholeSource.backwardReference,Recompute.memory,Reverse.memory,stateStore,write,ite_apply,ht,hf]
theorem dh0_value {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees)
    (hg : (g.correction s.BBV).Legal s.BBV) (scale : α) (m : Memory α) (bh iv k j : Nat)
    (hb : bh<s.BH) (hi : iv<s.BNV) (hk : k<s.K) (hj : j<s.BBV) (hv : iv*s.BBV+j<s.V)
    (hf : f.initialGrad=Bool.true) :
    backwardGrid s f p g scale m ⟨.dh0,bh,k,iv*s.BBV+j⟩=p.cut (Reverse.h s f p g scale m bh iv s.T k j) := by
  have hprot : BackwardRelation.Protected s (bh,iv) ⟨.dh0,bh,k,iv*s.BBV+j⟩ :=
    ⟨rfl,by dsimp;omega,by dsimp;rw [Nat.add_mul];omega⟩
  rw [BackwardGrid.grid_readback s f p g hg scale m bh iv hb hi _ hprot,WholeSource.backward_program]
  have high : iv*s.BBV+j<(iv+1)*s.BBV := by rw [Nat.add_mul];omega
  simp [WholeSource.backwardReference,Recompute.memory,Reverse.memory,stateStore,write,ite_apply,hk,hv,high,hf]

/-- Every off-tile term read by the vector-beta host reduction is initialized
to zero. This is why the pointer correction alone would be insufficient. -/
theorem dbeta_off_tile_zero {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees)
    (hg : (g.correction s.BBV).Legal s.BBV) (scale : α) (m : Memory α) (bh iv t v : Nat)
    (hb : bh<s.BH) (hi : iv<s.BNV) (ht : t<s.T) (hv : v<s.V) (hf : f.vectorBeta=Bool.true)
    (off : ¬(iv*s.BBV≤v ∧ v<min ((iv+1)*s.BBV) s.V)) :
    backwardGrid s f p g scale (prepareDbeta s f p m) ⟨.dbeta,iv*s.BH+bh,t,v⟩=p.zero := by
  rw [dbeta_value s f p g hg scale _ bh iv t v hb hi ht hf,if_neg off]
  have hp:=MemoryProof.scratch_plane s bh iv hb hi
  simp [prepareDbeta,write,hf,hp,ht,hv]

#axiomsClean dq_value
#axiomsClean dk_value
#axiomsClean dv_value
#axiomsClean dbeta_value
#axiomsClean dbeta_scalar_value
#axiomsClean dh0_value
#axiomsClean dbeta_off_tile_zero
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.BackwardResults
