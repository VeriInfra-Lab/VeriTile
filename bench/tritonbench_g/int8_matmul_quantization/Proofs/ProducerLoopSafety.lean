import bench.tritonbench_g.int8_matmul_quantization.Proofs.ProducerPrelude
import VeriTile.Triton.Float.StructuralSafety
namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer.SourceSafety
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 12000

theorem loop_from_invariant {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName→Nat) (body : List ComputeStmt) (invariant : Nat→State α→Prop)
    (safe : ∀ k state,invariant k state →
      Safety.traceSafeList M bounds body (state.setReg "k" .int [] (fun _=>(k : Int))))
    (advance : ∀ k state,invariant k state → ∃ next,
      run M body (state.setReg "k" .int [] (fun _=>(k : Int)))=some next ∧ invariant (k+1) next)
    (start stop : Nat) (state : State α) (valid : invariant start state) :
    Safety.rangeIntSafe M bounds "k" (start : Int) (stop : Int) 1 body state := by
  by_cases active : start<stop
  · obtain ⟨next,ran,inv⟩:=advance start state valid
    rw [Safety.rangeIntSafe_unfold,if_neg (by decide),if_pos (by exact_mod_cast active),ran]
    refine ⟨safe start state valid,?_⟩
    simpa only [Int.natCast_add,Int.natCast_one] using
      loop_from_invariant M bounds body invariant safe advance (start+1) stop next inv
  · rw [Safety.rangeIntSafe_unfold,if_neg (by decide),if_neg (by exact_mod_cast active)]
    trivial
termination_by stop-start

#axiomsClean loop_from_invariant
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer.SourceSafety
