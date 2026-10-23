import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.BackwardRelation
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Geometry
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.BackwardGrid
open GridPrinciple BackwardRelation
noncomputable section
set_option maxHeartbeats 10000000
theorem other_program {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (target job : Nat × Nat) (htarget : target.1<s.BH) (hjob : job.1<s.BH)
    (hne : job≠target) (m : Memory α) :
    Agree (Protected s target) (backwardProgram s f p g scale job.1 job.2 m) m := by
  intro a ha
  rcases target with ⟨tb,tv⟩
  rcases job with ⟨bh,iv⟩
  rcases a with ⟨r,plane,t,col⟩
  have hplane : iv*s.BH+bh≠tv*s.BH+tb := by
    intro he
    obtain ⟨hi,hb⟩:=Geometry.mixed_injective s.BH iv tv bh tb hjob htarget he
    apply hne
    simp [hi,hb]
  have outside (hp : plane=tb) (hlo : tv*s.BBV≤col) (hhi : col<(tv+1)*s.BBV) :
      ¬(plane=bh ∧ iv*s.BBV≤col ∧ col<min ((iv+1)*s.BBV) s.V) := by
    rintro ⟨he,lo,hi⟩
    have same:= (Geometry.owner_unique s.BBV iv col lo (lt_of_lt_of_le hi (Nat.min_le_left _ _))).trans
      (Geometry.owner_unique s.BBV tv col hlo hhi).symm
    apply hne
    have hb : bh=tb := by omega
    simp [hb,same]
  cases r <;> simp only [Protected] at ha <;> try contradiction
  all_goals simp only [WholeSource.backward_program,WholeSource.backwardReference]
  all_goals try (simp [Recompute.memory,Reverse.memory,stateStore,write,ite_apply];done)
  all_goals try
    have hp : plane≠iv*s.BH+bh := by intro he;apply hplane;omega
    simp [Recompute.memory,Reverse.memory,stateStore,write,ite_apply,hp]
    done
  all_goals
    obtain ⟨hp,hlo,hhi⟩:=ha
    have hout:=outside hp hlo hhi
    have hout' : ¬(plane=bh ∧ iv*s.BBV≤col ∧ col<(iv+1)*s.BBV ∧ col<s.V) := by simpa using hout
    cases hf:f.initialGrad <;> simp [Recompute.memory,Reverse.memory,stateStore,write,ite_apply,hf]
      <;> aesop

/-- All scratch readbacks come from their unique program, even though the
actual execution visits every program against live current memory. -/
theorem grid_readback {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees)
    (hg : (g.correction s.BBV).Legal s.BBV) (scale : α) (m : Memory α)
    (bh iv : Nat) (hb : bh<s.BH) (hv : iv<s.BNV) (a : Addr) (ha : Protected s (bh,iv) a) :
    backwardGrid s f p g scale m a=backwardProgram s f p g scale bh iv m a := by
  exact unique_writer_valid (fun job mem => backwardProgram s f p g scale job.1 job.2 mem)
    (Protected s) (fun job => job.1<s.BH) (bh,iv) a ha
    (fun job hj hne mem => other_program s f p g scale (bh,iv) job hb hj hne mem)
    (fun x y hc => program_agree s f p g hg scale bh iv x y hc a ha)
    (jobs s.BH s.BNV) (jobs_nodup _ _) ((MemoryProof.job_member _ _ _ _).mpr ⟨hb,hv⟩)
    (fun job hj => ((MemoryProof.job_member _ _ _ _).mp hj).1) m

#axiomsClean other_program
#axiomsClean grid_readback
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.BackwardGrid
