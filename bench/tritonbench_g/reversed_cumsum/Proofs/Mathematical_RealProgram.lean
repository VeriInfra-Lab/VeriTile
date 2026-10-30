import bench.tritonbench_g.reversed_cumsum.Proofs.Mathematical_Source
import bench.tritonbench_g.reversed_cumsum.Proofs.Mathematical_RealMath

/-! Exact per-iteration expressions and initialization, exposed for staged
proofs. The complete reverse-loop source connection is proved separately;
these fragments are never a replacement public kernel. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.RealProgram
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
    [.castIntToNat (.mul .int .nil (.ref .int [] "i_t") (.constInt (p.BT : Int))),
     .mul .nat .nil (.ref .nat [] "i_s") (.constNat p.BS)]

def carryOp (p : Layout) : Op .real [p.BS] :=
  let a := Op.ref .real [p.BS] "b_z"
  let b := Op.reduceSum ⟨0, by simp⟩ Bool.false (.ref .real [p.BT, p.BS] "b_s")
  .add .real (.consSame .nil) a b

def loopBody (p : Layout) (s o : RegionName) : List Stmt :=
  [.assign .blockPtr [p.BT, p.BS] "p_s" (pointerOp p s),
   .assign .blockPtr [p.BT, p.BS] "p_z" (pointerOp p o),
   .assign .real [p.BT, p.BS] "b_s"
     (.load .real (.blockPtr (.ref .blockPtr [p.BT, p.BS] "p_s") [0, 1]) .none .unspecified),
   .assign .real [p.BT, p.BS] "b_c"
     (.add .real (.consL (.consSame .nil))
       (.expandDim ⟨0, by simp⟩ (.ref .real [p.BS] "b_z"))
       (.dot (batch := []) (.ref .real [p.BT, p.BT] "m_s") (.ref .real [p.BT, p.BS] "b_s")
         {allowTF32 := some Bool.false})),
   .store .real [p.BT, p.BS] (.blockPtr (.ref .blockPtr [p.BT, p.BS] "p_z") [0, 1])
     (.ref .real [p.BT, p.BS] "b_c") .none,
   .ifThen (.ge .int .nil (.ref .int [] "i_t") (.castNatToInt (.constNat 0)))
     [.assign .real [p.BS] "b_z" (carryOp p)]]

def maskOp (p : Layout) (swapped : Bool) : Op .bool [p.BT,p.BT] :=
  if swapped then
    .ge .nat (.consL (.consR .nil))
      (.expandDim ⟨0, by simp⟩ (.ref .nat [p.BT] "o_i"))
      (.expandDim ⟨1, by simp⟩ (.ref .nat [p.BT] "o_i"))
  else
    .le .nat (.consR (.consL .nil))
      (.expandDim ⟨1, by simp⟩ (.ref .nat [p.BT] "o_i"))
      (.expandDim ⟨0, by simp⟩ (.ref .nat [p.BT] "o_i"))

def initCode (p : Layout) (swapped : Bool) : List Stmt :=
  [.assign .nat [] "i_s" (.programId 0), .assign .nat [] "i_bh" (.programId 1),
   .assign .nat [p.BT] "o_i" (.arange p.BT),
   .assign .real [p.BT, p.BT] "m_s"
     (.where (maskOp p swapped)
       (.broadcast (.const 1.0) [p.BT, p.BT]) (.broadcast (.const 0.0) [p.BT, p.BT])),
   .assign .real [p.BS] "b_z" (.full [p.BS] (.const 0))]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReversedCumsum.RealProgram
