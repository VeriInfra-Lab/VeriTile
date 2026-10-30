import bench.tritonbench_g.attn_fwd_triton.Proofs.FlatExecution
import bench.tritonbench_g.attn_fwd_triton.Proofs.CausalFormula

/-! The actual complete flat translation reaches the independent causal
formula from one shared initial memory, allowing read/read aliases. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.FlatCausal
open VeriTile.Triton Memory Relocation
noncomputable section

theorem executes (A : FlatAlloc) (s : BlockState) (p : Parameters) (BD : Nat) (hD : p.headDim ≤ BD)
    (hstage : p.stage = 3) (reciprocal : Bool) :
    let c := config A (Coordinates.configuration s p)
    ∃ t, stepStmts (A.flattenStmts (Verified.code 128 64 BD p reciprocal)) s = some t ∧
      (∀ i, Output.active c 128 BD i = Bool.true →
        (∀ x y, Output.active c 128 BD x = Bool.true → Output.active c 128 BD y = Bool.true →
          Output.address c 128 BD x = Output.address c 128 BD y → x = y) →
        t.readMem A.flat (Output.address c 128 BD i) =
          RealProofs.expected p.heads p.context p.headDim c.z c.h (c.startM*128+i.1.val) i.2.1.val
            (LogicalInputs.inputs s (parameters p A.flat) (offsets A p))) ∧
      (∀ region offset, (region ≠ A.flat ∨ ∀ i, Output.active c 128 BD i = Bool.true → offset ≠ Output.address c 128 BD i) →
        t.mem region offset = s.mem region offset) := by
  dsimp only
  obtain ⟨t,ht,hr,hf⟩ := FlatExecution.executes A s p 128 64 BD (by omega) reciprocal
  refine ⟨t,ht,?_,hf⟩
  intro i hi hinj
  rw [hr i hi hinj]
  rcases i with ⟨row,lane,⟨⟩⟩
  have hh : (config A (Coordinates.configuration s p)).startM*128+row.val < p.context ∧ lane.val < p.headDim := by
    simpa [Output.active,Output.outputConfig,VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealEpilogue.active]
      using hi
  simpa [FlatExecution.result,hstage,config,parameters,offsets] using
    CausalFormula.normalized s (config A (Coordinates.configuration s p)) BD row lane hD hh.1 hh.2

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.FlatCausal
