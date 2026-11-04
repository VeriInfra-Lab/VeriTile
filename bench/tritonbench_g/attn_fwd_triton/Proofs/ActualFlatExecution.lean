import bench.tritonbench_g.attn_fwd_triton.Proofs.ActualAssembly
import bench.tritonbench_g.attn_fwd_triton.Proofs.ActualMiddle

/-! Complete source execution in the shared backing memory, including both
helper invocations and all actual caller-side loads, casts and stores. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualFlatExecution
open VeriTile.Triton Memory Relocation
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

def result (A : FlatAlloc) (mem : BlockState) (p : Parameters) (BD : Nat) :=
  ActualMiddle.result mem (config A (Coordinates.configuration mem p)) BD
    (Initialization.initial 128 BD)

theorem executes (A : FlatAlloc) (s : BlockState) (p : Parameters) (BD : Nat) :
    let c := config A (Coordinates.configuration s p)
    let carry := result A s p BD
    ∃ t, stepStmts (A.flattenStmts (Verified.code 128 64 BD p Bool.false)) s=some t ∧
      (∀ i, Output.active c 128 BD i=Bool.true →
        (∀ x y, Output.active c 128 BD x=Bool.true → Output.active c 128 BD y=Bool.true →
          Output.address c 128 BD x=Output.address c 128 BD y → x=y) →
        t.readMem A.flat (Output.address c 128 BD i)=carry.numerator i/carry.denominator i.1) ∧
      (∀ region offset,
        (region≠A.flat ∨ ∀ i, Output.active c 128 BD i=Bool.true → offset≠Output.address c 128 BD i) →
        t.mem region offset=s.mem region offset) := by
  dsimp only
  let c := config A (Coordinates.configuration s p)
  obtain ⟨sp,prologueRan,pfixed,pcarry,pcaller,pout,pframe⟩ := ActualPrologue.executes A s p BD
  obtain ⟨sl,middleRan,lfixed,lcarry,lframe⟩ := ActualMiddle.executes s sp c BD
    (Initialization.initial 128 BD) pfixed pcarry pcaller
  have out : sl.regs .ptr [128,BD] "O_block_ptr"=some (outputPointer c 128 BD) :=
    (lframe.regs _ _ "O_block_ptr" (by decide)).trans pout
  obtain ⟨t,tailRan,read,frame⟩ := Output.executes sl c 128 BD Bool.false
    (result A s p BD).numerator (result A s p BD).denominator
    lcarry.numerator lcarry.denominator out lfixed.rows lfixed.context
  refine ⟨t,?_,read,?_⟩
  · rw [ActualAssembly.flat]
    have ran := (stepStmts.append_some ((stepStmts.append_some prologueRan).trans middleRan)).trans tailRan
    simpa only [ActualMiddle.code,ActualTail.source,List.append_assoc] using ran
  · intro region offset outside
    rw [frame region offset outside,lframe.mem,pframe.mem]

#axiomsClean executes
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualFlatExecution
