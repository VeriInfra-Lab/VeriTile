import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import bench.tritonbench_g.chunk_cumsum_vector.Sources.Mathematical_LegacyReal
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.chunk_cumsum_vector.Sources.Mathematical_RealMath

/-! The complete erased source, exposed as statements for the loop proof.
`original_body` checks the connection to Kernels; none of these fragments is
substituted for the full source in a public specification. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealProgram
open VeriTile.Triton
set_option maxRecDepth 12000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
structure Layout where
  headStride : Nat
  timeStride : Nat
  channelStride : Nat
  T : Nat
  S : Nat
  BT : Nat
  BS : Nat

structure Launch where
  feature : Nat
  head : Nat

def numChunks (p : Layout) : Nat := (p.T + p.BT - 1) / p.BT

def stopOp (p : Layout) : Op .nat [] :=
  .div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat p.T) (.constNat p.BT))
    (.constNat 1)) (.constNat p.BT)

def pointerOp (p : Layout) (r : RegionName) : Op .blockPtr [p.BT, p.BS] :=
  .makeBlockPtrDynOffsets r (.mul .nat .nil (.ref .nat [] "i_bh") (.constNat p.headStride))
    [p.T, p.S] [p.BT, p.BS] [p.timeStride, p.channelStride]
    [.mul .nat .nil (.ref .nat [] "i_t") (.constNat p.BT),
     .mul .nat .nil (.ref .nat [] "i_s") (.constNat p.BS)]

def carryOp (p : Layout) (swapped : Bool) : Op .real [p.BS] :=
  let a := Op.ref .real [p.BS] "b_z"
  let b := Op.reduceSum ⟨0, by simp⟩ Bool.false (.ref .real [p.BT, p.BS] "b_s")
  if swapped then .add .real (.consSame .nil) b a
  else .add .real (.consSame .nil) a b

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealProgram
