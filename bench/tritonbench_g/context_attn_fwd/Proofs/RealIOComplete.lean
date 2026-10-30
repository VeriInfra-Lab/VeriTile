import bench.tritonbench_g.context_attn_fwd.Proofs.RealBindings
import bench.tritonbench_g.context_attn_fwd.Proofs.RealRegional

namespace VeriTile.Bench.Repairs.ContextAttnFwd.RealIOComplete
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnFwd
open RealInterface
noncomputable section

theorem complete (p : Layout) (r : Regions) (extent : RegionBounds) (g : Grid) (scale : ℝ)
    (positive : 0 < p.BN) :
    (io p r extent g scale).ImplementsOn (Domain p r extent g) (formula p r extent g scale) := by
  intro v domain s bound
  have metadata := RealBindings.metadata p r extent g scale v s domain bound
  have activeEq (e : Event p g) : active p g (arrays r extent v) e=active p g (Program.inputs s r) e :=
    (metadata e.1).active p _ _ _ e.2
  have addressEq (e : Event p g) : address p g (arrays r extent v) e=address p g (Program.inputs s r) e :=
    (metadata e.1).address p _ _ _ e.2
  have unique : ∀ e f : Event p g,active p g (Program.inputs s r) e → active p g (Program.inputs s r) f →
      address p g (Program.inputs s r) e=address p g (Program.inputs s r) f → e=f := by
    intro e f he hf equal
    rw [←activeEq e] at he
    rw [←activeEq f] at hf
    rw [←addressEq e,←addressEq f] at equal
    exact domain.2 e f he hf equal
  obtain ⟨final,ran,values,frame⟩ := RealRegional.complete p r g scale s positive unique
  refine ⟨final,AlgorithmExecution.cons ran (.nil final),?_,?_,?_⟩
  · intro caps windows
    refine ⟨?_,trivial⟩
    intro idx
    exact WholeSafety.complete p r .fp32 Bool.false scale (s.withGridIndex idx) positive caps
      (RealBindings.capacity p r extent g scale v s domain bound caps windows idx)
  · intro o j hj
    change active p g (arrays r extent v) (decode p g j) at hj
    change final.readMem r.out (address p g (arrays r extent v) (decode p g j))=formula p r extent g scale v o j
    rw [addressEq]
    rw [activeEq] at hj
    rw [values (decode p g j) hj]
    exact (RealLocality.expected_eq p r extent (launch (decode p g j).1) _ _ positive
      (RealBindings.agreement p r extent g scale v s bound) (domain.1 (decode p g j).1)
      scale (decode p g j).2).symm
  · intro rg off outside _
    apply frame
    intro e he
    have ha : active p g (arrays r extent v) e := by rwa [activeEq]
    have miss := outside (0 : Fin 1) (encode p g e) (by simpa only [io,decode_encode] using ha)
    change rg≠r.out ∨ off≠address p g (arrays r extent v) (decode p g (encode p g e)) at miss
    simpa only [decode_encode,addressEq] using miss

#axiomsClean complete
end
end VeriTile.Bench.Repairs.ContextAttnFwd.RealIOComplete
