import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.Execution
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.CausalFormula

/-! Successful execution of the complete candidate with the independent
causal-attention readback and the all-cell logical-region memory frame. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.CausalExecution
open VeriTile.Triton Memory
noncomputable section

theorem executes (s : BlockState) (p : Parameters) (BD : Nat) (hD : p.headDim ≤ BD)
    (hstage : p.stage = 3) (reciprocal : Bool) :
    let c := Coordinates.configuration s p
    ∃ t, stepStmts (Verified.code 128 64 BD p reciprocal) s = some t ∧
      (∀ i, Output.active c 128 BD i = Bool.true →
        (∀ x y, Output.active c 128 BD x = Bool.true → Output.active c 128 BD y = Bool.true →
          Output.address c 128 BD x = Output.address c 128 BD y → x = y) →
        t.readMem p.Out (Output.address c 128 BD i) =
          RealProofs.expected p.heads p.context p.headDim c.z c.h (c.startM*128+i.1.val) i.2.1.val
            (LogicalInputs.inputs s p)) ∧
      (∀ region offset, (region ≠ p.Out ∨ ∀ i, Output.active c 128 BD i = Bool.true → offset ≠ Output.address c 128 BD i) →
        t.mem region offset = s.mem region offset) := by
  dsimp only
  obtain ⟨t,ht,hr,hf⟩ := Execution.executes s p 128 64 BD (by omega) reciprocal
  refine ⟨t,ht,?_,hf⟩
  intro i hi hinj
  rw [hr i hi hinj]
  rcases i with ⟨row,lane,⟨⟩⟩
  have hh : (Coordinates.configuration s p).startM*128+row.val < p.context ∧ lane.val < p.headDim := by
    simpa [Output.active,Output.outputConfig,VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealEpilogue.active]
      using hi
  simpa [Execution.result,hstage] using
    CausalFormula.normalized s (Coordinates.configuration s p) BD row lane hD hh.1 hh.2

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.CausalExecution
