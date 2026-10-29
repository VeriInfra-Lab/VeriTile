import bench.repairs.tritonbench_g.swiglu_backward.Contract

namespace VeriTile.Bench.Repairs.SwigluBackward
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole

theorem checkLaunch_iff (p : Prepared α call entry preexisting) (b : Nat → Nat) :
    checkLaunch p b = Bool.true ↔ LaunchContract p b := by
  simp only [checkLaunch,Bool.and_eq_true,decide_eq_true_eq]
  constructor
  · rintro ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨a,b⟩,c⟩,d⟩,e⟩,f⟩,g⟩,h⟩,i⟩,j⟩,k⟩
    exact ⟨a,b,c,d,e,f,g,h,i,j,k⟩
  · intro h
    exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨h.positive,h.sameDevice⟩,h.rows32⟩,h.columns32⟩,h.stride32⟩,
      h.padded32⟩,h.gradientUnique⟩,h.outputUnique⟩,h.alignment⟩,h.gradientSeparate⟩,h.outputSeparate⟩

#axiomsClean checkLaunch_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SwigluBackward
