import bench.repairs.tritonbench_g.decay_cumsum.Proofs.FPForward
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.FPPrepare
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.FPBackward

namespace VeriTile.Bench.Repairs.DecayCumsum.FPTile
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FPOps FPPrefix Geometry
noncomputable section
set_option maxHeartbeats 5000000
variable {α : Type} [Inhabited α]

def inputs : Kind → List RegionName
  | .forward => ["g"]
  | .prepare => ["q","k","g"]
  | .backward => FPBackwardBase.inputRegions

def outputs : Kind → List RegionName
  | .forward => ["go"]
  | .prepare => ["qg","kg"]
  | .backward => FPBackwardBase.outputRegions

def expected (M : Algebra α) (kind : Kind) (h : Call) (p : Phase) (prog : Program h p)
    (scale : ℝ) (u : State α) (r : RegionName) (t : Nat) (j : Fin h.BK) : Cell α :=
  match kind with
  | .forward => FPReference.forward M (fun n => raw h.BK "g" (off h p prog n) u (j,PUnit.unit)) t
  | .prepare => FPReference.stored M (if r="qg" then FPPrepare.qValue M h p prog scale u t (j,PUnit.unit)
      else FPPrepare.kValue M h p prog u t (j,PUnit.unit))
  | .backward => FPReference.backwardStored M r (FPBackwardBase.output M h p prog u r t (j,PUnit.unit))

theorem complete (M : Algebra α) (kind : Kind) (h : Call) (p : Phase) (positive : 0<p.span)
    (prog : Program h p) (scale : ℝ) (u : State α)
    (hf : u.pids 0=prog.feature) (hc : u.pids 1=prog.chunk) (hr : u.pids 2=prog.row) :
    ∃ v, FP.Structural.exec (engine M) (Kernels.source kind h p scale) u=some v ∧
      (∀ r∈outputs kind, ∀ (t : Fin p.span) (j : Fin h.BK),
        v.mem r (off h p prog t.val j)=expected M kind h p prog scale u r t.val j) ∧
      FPMemory.Frame h p prog (outputs kind) u v := by
  cases kind
  · obtain ⟨v,he,out,frame⟩ := FPForward.source_correct M h p prog scale u hf hc hr
    refine ⟨v,he,?_,frame⟩
    intro r member t j
    have eq : r="go" := by simpa [outputs] using member
    subst r
    exact out t j
  · obtain ⟨v,he,qo,ko,frame⟩ := FPPrepare.source_correct M h p positive prog scale u hf hc hr
    refine ⟨v,he,?_,frame⟩
    intro r member t j
    simp only [outputs,List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl
    · simpa [expected] using qo t j
    · simpa [expected] using ko t j
  · exact FPBackward.source_correct M h p positive prog scale u hf hc hr

#axiomsClean complete
end
end VeriTile.Bench.Repairs.DecayCumsum.FPTile
