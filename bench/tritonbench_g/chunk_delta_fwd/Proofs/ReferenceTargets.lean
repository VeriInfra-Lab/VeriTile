import bench.tritonbench_g.chunk_delta_fwd.IO

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.Reference
open VeriTile Triton FP.Structural ChunkDeltaFwd Interface
open VeriTile.Bench.TritonBenchG.ChunkDeltaFwd
open VeriTile.Bench.Repairs.ChunkGlaSimple.ScalarGraph (WordPrimitives)
noncomputable section

/-- Independent mathematical window recurrence, including rounded overlapping
windows when BC does not divide BT and the last visiting corrected-row value. -/
def realFormula (p : Parameters) (xs : Inputs ℝ p) : Outputs ℝ p :=
  let state := fun head => Real.state (realInputs p xs head)
    p.T p.K p.BT (((p.BT+p.BC-1)/p.BC)*p.BC) p.useSeed
  result (fun head c k v => state head.val c.val k.val v.val)
    (fun head t v => RealMath.corrected (realInputs p xs head.val) p.K
      (state head.val (t.val/p.BT)) t.val v.val)
    (fun head k v => state head.val p.NT k.val v.val)

/-- The existing ordered native graph: padded reduction lanes, casts, scalar
FMA expansion and the one-subblock carry-fusion policy remain unchanged. -/
def nativeFormula (I : WordPrimitives) (p : Parameters) (shape : Geometry.ShapeValid p)
    (xs : Inputs (BitVec 32) p) : Outputs (BitVec 32) p :=
  result
    (fun head c k v =>
      (NativeReference.history p shape (nativeInputs p xs) head.val c.val k.val v.val k.isLt).eval I.isa)
    (fun head t v =>
      (NativeReference.corrected p shape (nativeInputs p xs) head.val t.val v.val).eval I.isa)
    (fun head k v =>
      (NativeReference.history p shape (nativeInputs p xs) head.val p.NT k.val v.val k.isLt).eval I.isa)

#specNonCircular realFormula avoiding [FP.Structural.run,FP.Structural.exec,runGrid,gridMemory,
  ChunkDeltaFwd.states,Real.arrays,Real.target]
#specNonCircular nativeFormula avoiding [FP.Structural.run,FP.Structural.exec,runGrid,gridMemory,
  ChunkDeltaFwd.states,Native.inputs]
end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.Reference
