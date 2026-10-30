import bench.tritonbench_g.rotary_emb_nopad.Proofs.TypedDefinitions

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RotaryEmbNopad
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem cached_memory {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (p : Kernels.CachedLayout) (s : State A) (token : s.pids 1<p.q_total_tokens) :
    (cachedExecution rest ops p s).val.mem=(cReference ops p s).mem := by
  by_cases head : s.pids 0<p.Q_HEAD_NUM
  all_goals
    simp only [cachedExecution,cReference,head,↓reduceDIte,↓reduceIte,cBranch,
      Pair.written,Pair.halfScatter,cValue,cOffset,cInput,cCoefficient,cDim,cActive,cRegion,
      cBlock,cPast,algebra,VeriTile.Bench.Repairs.RotaryEmb.Typed.algebra,bop,
      Broadcast.leftIndex,Broadcast.rightIndex,State.setReg,masked_scatter_memory]
    try dsimp only [TileShape.insertAxis,TileShape.dropInsertedIndex]
    try simp only [int_half,←Int.natCast_add,←Int.natCast_mul,←Int.natCast_ediv,
      Int.toNat_natCast,Int.ofNat_lt]
    all_goals simp (config := { maxSteps := 3000000 }) +contextual
      [and_assoc,token,←IntegerExpr.tdiv_formula,Int.tmod_def]
    try dsimp only [TileShape.insertAxis,TileShape.dropInsertedIndex]
    try simp only [int_half,←Int.natCast_add,←Int.natCast_mul,←Int.natCast_ediv,
      Int.toNat_natCast,Int.ofNat_lt]
    all_goals simp (config := { maxSteps := 3000000 }) +contextual [and_assoc,token]
    try rfl

#axiomsClean cached_memory
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.Typed
