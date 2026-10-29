import bench.tritonbench_g.layer_norm_fwd.Proofs.Scale
import bench.tritonbench_g.layer_norm_fwd.Proofs.Memory

/-! Actual seven-round source dropout operations, including the mask store.
The source-expression identity is audited separately in SeedSource. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Execution
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def randomOp (seeds : RegionName) (block : Nat) : Op .real [block] :=
  .rand Domain.randomProfile
    (.castNatToInt (Op.intToUnsigned 32
      (.load .int (.region seeds.cast (.ref .nat [] "row")) .none)))
    (.castNatToInt (.ref .nat [block] "cols"))

def keepOp (seeds : RegionName) (p : ℝ) (block : Nat) : Op .bool [block] :=
  .gt .real .scalarR (randomOp seeds block) (.const p)

def dropoutPhase (enabled storeMask : Bool) (seeds mask : RegionName)
    (N block : Nat) (p : ℝ) : List Stmt :=
  [.ifThen (.constBool enabled)
    [.assign .bool [block] "keep_mask" (keepOp seeds p block),
     .assign .real [block] "x" (.where (.ref .bool [block] "keep_mask")
       (.div .real .scalarR (.ref .real [block] "x")
         (.sub .real .nil (.const 1.0) (.const p)))
       (.broadcast (.const 0.0) [block])),
     .ifThen (.constBool storeMask)
       [.store .bool [block]
         (.region mask.cast (.add .nat .scalarL
           (.mul .nat .nil (.ref .nat [] "row") (.constNat N))
           (.ref .nat [block] "cols")))
         (.ref .bool [block] "keep_mask")
         (.mask (.lt .nat .scalarR (.ref .nat [block] "cols") (.constNat N)))]]]

def keepTile (seed : Int) (p : ℝ) (block : Nat) : Tile .bool [block] :=
  ⟨fun j => Formula.keep seed p j.1.val⟩

theorem randomOp_eval (seeds : RegionName) (row block : Nat) (s : BlockState)
    (hBlock : Random.ArangeSizeDomain block)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val))) :
    evalOp (randomOp seeds block) s =
      some ⟨fun j => some (Random.randReal
        (Formula.seedU32 (s.readMemValue .int seeds row)) (j.1.val : Int) .bits32 7)⟩ := by
  unfold randomOp
  apply Domain.eval_rand _ hBlock
  · apply Domain.eval_seed_cast
    simp [evalOp.eq_def, hrow, Region.cast]
  · simp [evalOp.eq_def, hcols]

theorem keepOp_eval (seeds : RegionName) (row block : Nat) (p : ℝ) (s : BlockState)
    (hBlock : Random.ArangeSizeDomain block)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val))) :
    evalOp (keepOp seeds p block) s =
      some (keepTile (s.readMemValue .int seeds row) p block) := by
  unfold keepOp
  rw [evalOp_gt, randomOp_eval seeds row block s hBlock hrow hcols]
  simp only [evalOp_const, bind, Option.bind, pure]
  congr 1
  ext j
  simp only [keepTile, Tile.cop, ComparableDType.gt, Formula.keep,
    Tile.scalar, Tile.vec, Broadcast.leftIndex, Broadcast.rightIndex,
    WithBot.some_eq_coe, WithBot.coe_lt_coe]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Execution
