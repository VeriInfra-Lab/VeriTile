import bench.tritonbench_g.swiglu_triton.Proofs.DirectTile
import bench.tritonbench_g.swiglu_triton.Proofs.NativeProfile

namespace VeriTile.Bench.Repairs.SwigluTriton.KernelNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Observed (Term graphAlgebra)
open Native
noncomputable section

/-- Loads are raw word leaves. Graph contraction cannot cross a memory load. -/
def rawState (ports : List RegionName) (read : RegionName → Nat → Word)
    (scratch : State (Term Word)) : State (Term Word) :=
  { scratch with mem := fun r o =>
      if r ∈ ports then .mk .real (.input (read r o)) else scratch.mem r o }

theorem rawState_read (ports : List RegionName) (read : RegionName → Nat → Word)
    (scratch : State (Term Word)) (r : RegionName) (o : Nat) (hr : r ∈ ports) :
    ((rawState ports read scratch).mem r o).read .real = .input (read r o) := by
  simp only [rawState, hr, if_true]
  rfl

theorem scalar_lowering (instructions : InstructionSemantics) (ch : Channel)
    (a b dc : Word) :
    interpret (compiledPorts instructions)
      (DirectTile.scalar graphAlgebra ch (.input a) (.input b) (.input dc)) =
      compiledReference instructions ch a b dc :=
  compiled_lowering instructions ch a b dc

/-- Successful full original source, actual ordered native expression at each
active output, and complete exterior typed-memory frame. No primitive identity
or numerical transformation law is assumed. -/
theorem correct (instructions : InstructionSemantics) (backward : Bool)
    (block cols row : Nat) (covers : cols ≤ block) (read : RegionName → Nat → Word)
    (scratch : State (Term Word)) :
    ∃ final, exec graphAlgebra (directSource backward block cols)
        (DirectTile.rowState row (rawState ["A", "B", "DC"] read scratch)) = some final ∧
      (∀ ch, enabled backward ch → ∀ col, col < cols →
        interpret (compiledPorts instructions)
          ((final.mem (outputRegion ch) (row*cols+col)).read .real) =
          compiledReference instructions ch (read "A" (row*cols+col))
            (read "B" (row*cols+col)) (read "DC" (row*cols+col))) ∧
      (∀ r o, DirectTile.Outside backward block cols row r o →
        final.mem r o = (rawState ["A", "B", "DC"] read scratch).mem r o) := by
  refine ⟨(DirectTile.sourceRun graphAlgebra backward block cols row (rawState ["A", "B", "DC"] read scratch)).val,
    (DirectTile.sourceRun graphAlgebra backward block cols row (rawState ["A", "B", "DC"] read scratch)).property, ?_, ?_⟩
  · intro ch on col hc
    rw [DirectTile.readback graphAlgebra backward block cols row (rawState ["A", "B", "DC"] read scratch) ch on
      ⟨⟨col, lt_of_lt_of_le hc covers⟩, PUnit.unit⟩ hc]
    simpa [rawState, Cell.read, ofFloat] using scalar_lowering instructions ch
      (read "A" (row*cols+col)) (read "B" (row*cols+col)) (read "DC" (row*cols+col))
  · exact DirectTile.frame graphAlgebra backward block cols row (rawState ["A", "B", "DC"] read scratch)

#axiomsClean correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton.KernelNative
