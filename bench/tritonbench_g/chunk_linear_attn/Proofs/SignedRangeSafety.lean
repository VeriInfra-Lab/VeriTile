import bench.tritonbench_g.chunk_linear_attn.Proofs.SignedRange
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.SignedRange
open VeriTile Triton FP.Structural
noncomputable section

/-- A finite natural proof clock for the actual signed counter. Both execution
and access safety retain the signed register and the original loop body. -/
theorem clock_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (index : RegName) (body : List ComputeStmt)
    (offset : Int) (stop : Nat) (I : Nat → State α → Prop)
    (bodySafe : ∀ n, n<stop → ∀ s, I n s →
      Safety.traceSafeList M bounds body
        (s.setReg index .int [] (fun _ => offset+(n : Int))))
    (preserved : ∀ n, n<stop → ∀ s, I n s → ∀ t,
      run M body (s.setReg index .int [] (fun _ => offset+(n : Int)))=some t → I (n+1) t)
    (start : Nat) (bound : start≤stop) (s : State α) (initial : I start s) :
    Safety.rangeIntSafe M bounds index (offset+(start : Int)) (offset+(stop : Int)) 1 body s := by
  let P := fun q t => ∃ n : Nat, n≤stop ∧ q=offset+(n : Int) ∧ I n t
  apply Safety.rangeIntSafe_of_invariant M bounds index (offset+(stop : Int)) 1 body P
  · intro q active t reached
    obtain ⟨n, hn, clock, state⟩ := reached
    have live : n<stop := by omega
    subst q
    exact bodySafe n live t state
  · intro q active t reached u ran
    obtain ⟨n, hn, clock, state⟩ := reached
    have live : n<stop := by omega
    subst q
    refine ⟨n+1, by omega, ?_, preserved n live t state u ran⟩
    push_cast
    omega
  · exact ⟨start,bound,rfl,initial⟩

#axiomsClean clock_safe
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.SignedRange
