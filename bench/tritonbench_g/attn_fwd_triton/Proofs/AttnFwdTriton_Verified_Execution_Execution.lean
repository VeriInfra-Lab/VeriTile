import bench.tritonbench_g.attn_fwd_triton.Proofs.Prologue
import bench.tritonbench_g.attn_fwd_triton.Proofs.TwoLoops
import bench.tritonbench_g.attn_fwd_triton.Proofs.Output

/-! Whole changed-source Real execution, from arbitrary initial registers to
all active output lanes and an all-cell memory frame. The next input-formula
bridge removes the block decomposition from the readback expression. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Execution
open VeriTile.Triton Memory
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem body (p : Parameters) (BM BN BD : Nat) (reciprocal : Bool) :
    Verified.code BM BN BD p reciprocal =
      (coordinates BM BN BD p ++ pointers BM BN BD p ++ initialization BM BN BD p) ++
      TwoLoops.code BM BN BD p.headDim (decide (p.stage=2)) ++ epilogue BM BD p.headDim reciprocal := by
  rw [decomposition,SetupSource.first,SetupSource.second]
  simp only [TwoLoops.code,List.append_assoc]

def result (mem : BlockState) (p : Parameters) (BM BN BD : Nat) :=
  TwoLoops.bothResult mem (Coordinates.configuration mem p) BM BN BD (decide (p.stage=2))
    (Initialization.initial BM BD)

theorem executes (s : BlockState) (p : Parameters) (BM BN BD : Nat) (hN : 0 < BN) (reciprocal : Bool) :
    let c := Coordinates.configuration s p
    let carry := result s p BM BN BD
    ∃ t, stepStmts (Verified.code BM BN BD p reciprocal) s = some t ∧
      (∀ i, Output.active c BM BD i = Bool.true →
        (∀ x y, Output.active c BM BD x = Bool.true → Output.active c BM BD y = Bool.true →
          Output.address c BM BD x = Output.address c BM BD y → x = y) →
        t.readMem p.Out (Output.address c BM BD i) = carry.numerator i/carry.denominator i.1) ∧
      (∀ region offset, (region ≠ p.Out ∨ ∀ i, Output.active c BM BD i = Bool.true → offset ≠ Output.address c BM BD i) →
        t.mem region offset = s.mem region offset) := by
  dsimp only
  let c := Coordinates.configuration s p
  obtain ⟨sp,hpstep,hfixed,hcarry,hcaller,hout,hpf⟩ := Prologue.executes s p BM BN BD
  obtain ⟨sl,hlstep,hlfixed,hlcarry,hlf⟩ := TwoLoops.executes s sp c BM BN BD (decide (p.stage=2))
    (Initialization.initial BM BD) hN hfixed hcarry hcaller
  have hop : sl.regs .ptr [BM,BD] "O_block_ptr" = some (outputPointer c BM BD) :=
    (hlf.regs _ _ "O_block_ptr" (by decide)).trans hout
  obtain ⟨t,htstep,htread,htframe⟩ := Output.executes sl c BM BD reciprocal
    (result s p BM BN BD).numerator (result s p BM BN BD).denominator
    hlcarry.numerator hlcarry.denominator hop hlfixed.rows hlfixed.context
  refine ⟨t,?_,htread,?_⟩
  · rw [body]
    exact (stepStmts.append_some ((stepStmts.append_some hpstep).trans hlstep)).trans htstep
  · intro region offset h
    rw [htframe region offset h,hlf.mem,hpf.mem]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Execution
