import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.ForwardGrid
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.RealForward
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.ForwardResults
noncomputable section
set_option maxHeartbeats 5000000
theorem bounds (s : Shape) (v : Nat) (hv : v<s.V) :
    v/s.FBV<s.FNV ∧ v%s.FBV<s.FBV ∧
    (v/s.FBV)*s.FBV≤v ∧ v<min ((v/s.FBV+1)*s.FBV) s.V ∧ v-(v/s.FBV)*s.FBV=v%s.FBV := by
  obtain ⟨ho,hj,he⟩:=Geometry.owner_exists s.V s.FBV v (Geometry.tile_pos s).2.1 hv
  refine ⟨ho,hj,by omega,?_,by omega⟩
  apply lt_min _ hv
  rw [Nat.add_mul]
  omega
theorem output_value {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) (bh t v : Nat) (hb : bh<s.BH) (ht : t<s.T) (hv : v<s.V) :
    forwardGrid s f p g scale m ⟨.o,bh,t,v⟩=
      p.cut (Forward.output s f p g scale m bh (v/s.FBV) t (v%s.FBV)) := by
  obtain ⟨ho,hj,hlo,hhi,he⟩:=bounds s v hv
  have hprot : ForwardGrid.Protected s (bh,v/s.FBV) ⟨.o,bh,t,v⟩ :=
    ⟨rfl,hlo,lt_of_lt_of_le hhi (Nat.min_le_left _ _)⟩
  rw [ForwardGrid.grid_readback s f p g scale m bh (v/s.FBV) hb ho _ hprot,Forward.program_full]
  simp [Forward.memory,stateStore,write,ite_apply,ht,hlo,hhi,he]
theorem residual_value {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) (bh t v : Nat) (hb : bh<s.BH) (ht : t<s.T) (hv : v<s.V) :
    forwardGrid s f p g scale m ⟨.v,bh,t,v⟩=
      p.cut (Forward.residual s f p g m bh (v/s.FBV) t (v%s.FBV)) := by
  obtain ⟨ho,hj,hlo,hhi,he⟩:=bounds s v hv
  have hprot : ForwardGrid.Protected s (bh,v/s.FBV) ⟨.v,bh,t,v⟩ :=
    ⟨rfl,hlo,lt_of_lt_of_le hhi (Nat.min_le_left _ _)⟩
  rw [ForwardGrid.grid_readback s f p g scale m bh (v/s.FBV) hb ho _ hprot,Forward.program_full]
  simp [Forward.memory,stateStore,write,ite_apply,ht,hlo,hhi,he]
theorem final_value {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) (bh k v : Nat) (hb : bh<s.BH) (hk : k<s.K) (hv : v<s.V) (hf : f.final=Bool.true) :
    forwardGrid s f p g scale m ⟨.ht,bh,k,v⟩=
      p.cut (Forward.h s f p g m bh (v/s.FBV) s.T k (v%s.FBV)) := by
  obtain ⟨ho,hj,hlo,hhi,he⟩:=bounds s v hv
  have hprot : ForwardGrid.Protected s (bh,v/s.FBV) ⟨.ht,bh,k,v⟩ :=
    ⟨rfl,hlo,lt_of_lt_of_le hhi (Nat.min_le_left _ _)⟩
  rw [ForwardGrid.grid_readback s f p g scale m bh (v/s.FBV) hb ho _ hprot,Forward.program_full]
  simp [Forward.memory,stateStore,write,ite_apply,hf,hk,hlo,hhi,he]
theorem output_real (s : Shape) (f : Flags) (g : Trees) (hg : (g.key s.BK).Legal s.BK)
    (ho : (g.readout s.BK).Legal s.BK)
    (scale : ℝ) (m : Memory ℝ) (bh t v : Nat) (hb : bh<s.BH) (ht : t<s.T) (hv : v<s.V) :
    forwardGrid s f Real.ops g scale m ⟨.o,bh,t,v⟩=Math.output scale (Real.column s f m bh v) t := by
  rw [output_value s f Real.ops g scale m bh t v hb ht hv]
  change Forward.output s f Real.ops g scale m bh (v/s.FBV) t (v%s.FBV)=_
  have he:=(Geometry.owner_exists s.V s.FBV v (Geometry.tile_pos s).2.1 hv).2.2
  rw [Real.output_math s f g hg ho scale m bh (v/s.FBV) (v%s.FBV) t (by simpa [he] using hv),he]
theorem residual_real (s : Shape) (f : Flags) (g : Trees) (hg : (g.key s.BK).Legal s.BK)
    (scale : ℝ) (m : Memory ℝ) (bh t v : Nat) (hb : bh<s.BH) (ht : t<s.T) (hv : v<s.V) :
    forwardGrid s f Real.ops g scale m ⟨.v,bh,t,v⟩=
      Math.residual (Math.state (Real.column s f m bh v) t) ((Real.column s f m bh v).k t) ((Real.column s f m bh v).x t) := by
  rw [residual_value s f Real.ops g scale m bh t v hb ht hv]
  change Forward.residual s f Real.ops g m bh (v/s.FBV) t (v%s.FBV)=_
  have he:=(Geometry.owner_exists s.V s.FBV v (Geometry.tile_pos s).2.1 hv).2.2
  rw [Real.residual_math s f g hg m bh (v/s.FBV) (v%s.FBV) t (by simpa [he] using hv),he]
theorem final_real (s : Shape) (f : Flags) (g : Trees) (hg : (g.key s.BK).Legal s.BK)
    (scale : ℝ) (m : Memory ℝ) (bh v : Nat) (i : Fin s.K) (hb : bh<s.BH) (hv : v<s.V) (hf : f.final=Bool.true) :
    forwardGrid s f Real.ops g scale m ⟨.ht,bh,i.val,v⟩=Math.state (Real.column s f m bh v) s.T i := by
  rw [final_value s f Real.ops g scale m bh i.val v hb i.isLt hv hf]
  change Forward.h s f Real.ops g m bh (v/s.FBV) s.T i.val (v%s.FBV)=_
  have he:=(Geometry.owner_exists s.V s.FBV v (Geometry.tile_pos s).2.1 hv).2.2
  rw [Real.h_math s f g hg m bh (v/s.FBV) (v%s.FBV) (by simpa [he] using hv) s.T i,he]

#axiomsClean output_value
#axiomsClean residual_value
#axiomsClean final_value
#axiomsClean output_real
#axiomsClean residual_real
#axiomsClean final_real
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.ForwardResults
