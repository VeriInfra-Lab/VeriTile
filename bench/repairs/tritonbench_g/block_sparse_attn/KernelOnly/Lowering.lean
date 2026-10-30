import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Source

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

def same1 : Broadcast [16] [16] [16] := .consSame .nil
def same2 : Broadcast [16,16] [16,16] [16,16] := .consSame (.consSame .nil)
def ref1 (name : RegName) : Op .real [16] := .ref .real [16] name
def ref2 (name : RegName) : Op .real [16,16] := .ref .real [16,16] name
def fp {shape : TileShape} (e : Op .real shape) : ComputeExpr .real shape :=
  .compute (.alg .fp32 e)

def sourceDotNode (acc left right : RegName) : ComputeStmt :=
  .assign .real [16,16] acc (fp (.add .real same2 (ref2 acc)
    (.dot (batch := []) (ref2 left) (ref2 right))))
def carryNode (acc left right : RegName) : ComputeStmt :=
  .assign .real [16,16] acc (fp (.dotAcc .real .real .real (batch := [])
    (ref2 left) (ref2 right) (ref2 acc) {}))

def causalMask : Op .real [16,16] :=
  .where (.ge .int (.consR (.consL .nil))
    (.castNatToInt (.expandDim ⟨1,by decide⟩ (.ref .nat [16] "offs_m")))
    (.add .int .scalarL (.ref .int [] "start_n")
      (.castNatToInt (.expandDim ⟨0,by decide⟩ (.ref .nat [16] "offs_n")))))
    (.broadcast (.const 0) [16,16]) (.broadcast .negInf [16,16])
def sourceScaleNode (p : Parameters) : ComputeStmt :=
  .assign .real [16,16] "qk" (fp (.mul .real .scalarR (ref2 "qk") (.ref .real [] "softmax_scale")))
def sourceMaskNode : ComputeStmt :=
  .assign .real [16,16] "qk" (fp (.add .real same2 (ref2 "qk") causalMask))
def scoreFmaNode (p : Parameters) : ComputeStmt :=
  .assign .real [16,16] "qk"
    (.fma .fp32 .scalarL same2 (fp (.ref .real [] "softmax_scale")) (fp (ref2 "qk")) (fp causalMask))

def sourceDenominatorNode : ComputeStmt :=
  .assign .real [16] "l_i_new" (fp (.add .real same1
    (.mul .real same1 (ref1 "alpha") (ref1 "l_i"))
    (.mul .real same1 (ref1 "beta") (ref1 "l_ij"))))
def denominatorFmaNode : ComputeStmt :=
  .assign .real [16] "l_i_new" (.fma .fp32 same1 same1
    (fp (ref1 "l_i")) (fp (ref1 "alpha"))
    (fp (.mul .real same1 (ref1 "l_ij") (ref1 "beta"))))
def sourceAccScaleNode : ComputeStmt :=
  .assign .real [16] "acc_scale" (fp (.mul .real same1
    (.div .real same1 (ref1 "l_i") (ref1 "l_i_new")) (ref1 "alpha")))
def accScaleNode : ComputeStmt :=
  .assign .real [16] "acc_scale" (fp (.mul .real same1 (ref1 "alpha")
    (.div .real same1 (ref1 "l_i") (ref1 "l_i_new"))))

def compiledScoreBody (p : Parameters) :=
  (scoreBody p).take 2 ++ [carryNode "qk" "q" "k"]

/-- Selected local lowering. The PTX uses scale*score+mask, oldDen*alpha+
blockSum*beta, alpha*(oldDen/newDen), and seeded QK/PV fixed-K8 MMA.
These are observed expression-site realizations, not FP algebraic identities. -/
def compiledBody (p : Parameters) :=
  (body p).take 3 ++ [.forLoop "score_block" p.blocks (compiledScoreBody p)] ++
  [scoreFmaNode p] ++ ((body p).drop 6).take 6 ++ [denominatorFmaNode] ++
  ((body p).drop 13).take 2 ++ [accScaleNode] ++ ((body p).drop 16).take 3 ++
  [carryNode "acc" "p" "v"] ++ (body p).drop 20

theorem source_sites (p : Parameters) :
    scoreBody p = (scoreBody p).take 2 ++ [sourceDotNode "qk" "q" "k"] ∧
    (body p)[4]? = some (sourceScaleNode p) ∧
    (body p)[5]? = some sourceMaskNode ∧
    (body p)[12]? = some sourceDenominatorNode ∧
    (body p)[15]? = some sourceAccScaleNode ∧
    (body p)[19]? = some (sourceDotNode "acc" "p" "v") := by
  exact ⟨rfl,rfl,rfl,rfl,rfl,rfl⟩

theorem source_body_partition (p : Parameters) :
    body p = (body p).take 3 ++ [.forLoop "score_block" p.blocks (scoreBody p)] ++
      [sourceScaleNode p,sourceMaskNode] ++ ((body p).drop 6).take 6 ++
      [sourceDenominatorNode] ++ ((body p).drop 13).take 2 ++
      [sourceAccScaleNode] ++ ((body p).drop 16).take 3 ++
      [sourceDotNode "acc" "p" "v"] ++ (body p).drop 20 := by
  rfl

def loopBody (native : Bool) (p : Parameters) := if native then compiledBody p else body p
def channelBody (native : Bool) (p : Parameters) :=
  if native then compiledScoreBody p else scoreBody p
def selectedSource (p : Parameters) := prelude p ++
  [.forRangeInt "col_idx_idx" (.ref .int [] "start_l")
    (.ref .int [] "end_l") (.constNat 1) (compiledBody p)] ++ epilogue p

#axiomsClean source_sites
#axiomsClean source_body_partition
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
