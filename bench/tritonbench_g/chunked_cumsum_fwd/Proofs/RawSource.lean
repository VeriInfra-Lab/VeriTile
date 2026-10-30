import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholeBackendTile

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawSource
open VeriTile.Triton FP.Structural FP.ProfiledRewrite Math Program
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

/-- The original runtime clamp is a max followed by a min. Both arguments
are separately supplied FP32 words, never Real literals. The local minimum
node is necessary because the historical DSL expands min to compare/where,
which cannot encode a canonical NaN result distinct from either operand. -/
structure Code where
  beforeClamp : List ComputeStmt
  afterClamp : List ComputeStmt

def source (c : Whole.Host) (r : Regions) (hasBias useSoftplus : Bool) : Code :=
  let original := WholeFP.typed32 c r Bool.false hasBias useSoftplus 0 (.finite 0)
  let before := Source.prefixBeforeStore original.surfaceBody
  ⟨before.take (before.length-2), before.drop (before.length-1) ++ original.surfaceBody.drop before.length⟩

/-- Only the clamp operand materialization changes. The source context on both
sides is exactly the historical complete original source for every flag pair. -/
theorem source_context (c : Whole.Host) (r : Regions) (hasBias useSoftplus : Bool)
    (lower upper : ℝ) :
    let original := WholeFP.typed32 c r Bool.false hasBias useSoftplus lower (.finite upper)
    let before := Source.prefixBeforeStore original.surfaceBody
    original.surfaceBody = (source c r hasBias useSoftplus).beforeClamp ++
      (before.drop (before.length-2)).take 1 ++ (source c r hasBias useSoftplus).afterClamp := by
  cases hasBias <;> cases useSoftplus <;>
    simp [source,WholeFP.typed32,Program.kernel,VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.source,finiteSource,ComputeKernel.surfaceBody,Source.prefixBeforeStore]

def exec {α : Type} [Inhabited α] (M : Algebra α) (minimum : α → α → α)
    (c : Whole.Host) (code : Code) (lower upper : α) (l : Launch) (s : State α) : Option (State α) :=
  (run (engine M) code.beforeClamp (WholeFP.tileState l s)).bind fun middle =>
    (middle.regs .real [c.BH,c.BC] "dt").bind fun dt =>
      run (engine M) code.afterClamp (middle.setReg "dt" .real [c.BH,c.BC]
        (fun i => minimum (M.binary (some .fp32) .real .max (dt i) lower) upper))

def runTiles {α : Type} [Inhabited α] (M : Algebra α) (minimum : α → α → α)
    (c : Whole.Host) (code : Code) (lower upper : α) : List Launch → State α → Option (State α)
  | [],s => some s
  | l::rest,s => (exec M minimum c code lower upper l s).bind (runTiles M minimum c code lower upper rest)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawSource
