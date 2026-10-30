import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.FlatPrologue
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.FlatSource
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.CausalFormula
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.TwoLoops
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.Output

/-! Execute the complete translated source in one shared memory. Read roles
may overlap at independent offsets; no disjoint-region premise is used. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.FlatExecution
open VeriTile.Triton Memory Relocation
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def result (A : FlatAlloc) (mem : BlockState) (p : Parameters) (BM BN BD : Nat) :=
  TwoLoops.bothResult mem (config A (Coordinates.configuration mem p)) BM BN BD (decide (p.stage=2))
    (Initialization.initial BM BD)

theorem executes (A : FlatAlloc) (s : BlockState) (p : Parameters) (BM BN BD : Nat) (hN : 0 < BN) (reciprocal : Bool) :
    let c := config A (Coordinates.configuration s p)
    let carry := result A s p BM BN BD
    ∃ t, stepStmts (A.flattenStmts (Verified.code BM BN BD p reciprocal)) s = some t ∧
      (∀ i, Output.active c BM BD i = Bool.true →
        (∀ x y, Output.active c BM BD x = Bool.true → Output.active c BM BD y = Bool.true →
          Output.address c BM BD x = Output.address c BM BD y → x = y) →
        t.readMem A.flat (Output.address c BM BD i) = carry.numerator i/carry.denominator i.1) ∧
      (∀ region offset, (region ≠ A.flat ∨ ∀ i, Output.active c BM BD i = Bool.true → offset ≠ Output.address c BM BD i) →
        t.mem region offset = s.mem region offset) := by
  dsimp only
  let c := config A (Coordinates.configuration s p)
  obtain ⟨sp,hpstep,hfixed,hcarry,hcaller,hout,hpf⟩ := FlatPrologue.executes A s p BM BN BD
  obtain ⟨sl,hlstep,hlfixed,hlcarry,hlf⟩ := TwoLoops.executes s sp c BM BN BD (decide (p.stage=2))
    (Initialization.initial BM BD) hN hfixed hcarry hcaller
  have hop : sl.regs .ptr [BM,BD] "O_block_ptr" = some (outputPointer c BM BD) :=
    (hlf.regs _ _ "O_block_ptr" (by decide)).trans hout
  obtain ⟨t,htstep,htread,htframe⟩ := Output.executes sl c BM BD reciprocal
    (result A s p BM BN BD).numerator (result A s p BM BN BD).denominator
    hlcarry.numerator hlcarry.denominator hop hlfixed.rows hlfixed.context
  refine ⟨t,?_,htread,?_⟩
  · rw [FlatSource.body]
    exact (stepStmts.append_some ((stepStmts.append_some hpstep).trans hlstep)).trans htstep
  · intro region offset h
    rw [htframe region offset h,hlf.mem,hpf.mem]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.FlatExecution
