import bench.repairs.tritonbench_g.token_attn_reduceV.Real
import bench.repairs.tritonbench_g.token_attn_reduceV.Stores

namespace VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.RawReal
open VeriTile.Triton FP.Structural TokenAttnReduceV
noncomputable section

/-- Every program in the complete raw grid executes the quoted kernel. Each
output equals the independent gathered weighted sum, with the full raw frame. -/
theorem grid_real (p : Parameters) (batches heads : Nat) (work : List Grid.PID)
    (order : (Grid.fullGrid batches heads).Perm work) (memory : Grid.Memory ℝ)
    (positive : 0<p.block) (separate : Grid.Separate p) (unique : Grid.Unique p work) :
    ∃ after,Grid.run Real.algebra Real.sum p work memory=some after ∧
      (∀ b<batches,∀ h<heads,∀ d : Fin p.width,
        (after p.output (Grid.offset p (b,h) (d,PUnit.unit))).read .fp32=
          Real.weighted p (Grid.state memory (b,h)) d.val) ∧
      (∀ r o,(r≠p.output ∨ ∀ e∈Grid.events p work,o≠Grid.offset p e.1 e.2) →
        after r o=memory r o) := by
  obtain ⟨after,run,_,values,frame⟩ := Grid.whole Real.algebra Real.sum p work memory separate unique
  refine ⟨after,run,?_,frame⟩
  intro b hb h hh d
  let e : Grid.Event p := ((b,h),(d,PUnit.unit))
  have live : e∈Grid.events p work :=
    (Grid.mem_events _ _ _).mpr (order.mem_iff.mp ((Grid.mem_fullGrid _ _ _).mpr ⟨hb,hh⟩))
  rw [values e live,Cell.read_same]
  exact Real.output_weighted p (Grid.state memory (b,h)) positive (d,PUnit.unit)

#axiomsClean grid_real
#specNonCircular Real.weighted avoiding [kernel,FP.Structural.exec,Grid.run,outputValues]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.RawReal
