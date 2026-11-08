import bench.tritonbench_g.chunk_linear_attn.Proofs.Support.StateProgram
import bench.tritonbench_g.chunk_linear_attn.Proofs.SignedRangeSafety

namespace VeriTile.Tests.ChunkLinearPastedBody
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkLinearAttn
noncomputable section

/-- Empty Python reverse ranges must stay empty: natural subtraction would
incorrectly create one iteration when NT is zero. -/
theorem empty_reverse {α : Type} [Inhabited α] (M : Algebra α)
    (body : List ComputeStmt) (s : State α) :
    rangeInt M "__rev_i_t" (0-((0 : Int)-1)) 1 1 body s=some s := by
  rw [rangeInt]
  simp

/-- The source has signed reverse bounds for every NT, including zero. -/
theorem reverse_bounds (p : Parameters) (scale : ℝ) (seed final : Bool) :
    stateLoopStmt p scale Bool.true seed final =
      .forRangeInt "__rev_i_t"
        (.sub .int .nil (.constInt 0) (.sub .int .nil (.constInt (Int.ofNat p.NT)) (.constInt 1)))
        (.constInt 1) (.constNat 1) (stateIntParts p scale seed final).body := rfl

/-- Three live signed counter positions map to Python's exact 2,1,0 order. -/
theorem descending_three :
    ([0,1,2] : List Nat).map (fun n => (0 : Int)-((n : Int)+1-3))=[2,1,0] := by decide

/-- The nested source load materializes FP32 before the explicit cast.
This statement holds for arbitrary load/cast semantics; neither is erased. -/
theorem seed_keeps_load_and_cast {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (pid : PID) (memory : Memory α) (rev : Bool)
    (i : TileIndex [p.BK,p.BV]) :
    stateInitial M p pid memory rev Bool.true i =
      M.cast (some .fp32) .real .real
        (M.fp32Load (rawLoad M memory (seedName rev) (pid.z*p.K*p.V)
          p.K p.V p.V 1 (pid.x*p.BK) (pid.y*p.BV) i)) := by
  cases rev <;> rfl

#axiomsClean empty_reverse
#axiomsClean reverse_bounds
#axiomsClean descending_three
#axiomsClean seed_keeps_load_and_cast
end
end VeriTile.Tests.ChunkLinearPastedBody
