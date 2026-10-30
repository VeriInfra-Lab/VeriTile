import bench.tritonbench_g.llama_ff_triton.Proofs.Source

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

def carryNode (acc weight : RegName) : ComputeStmt :=
  .assign .real [16,16] acc (.compute (.alg .fp32
    (.dotAcc .fp16 .fp16 .real (batch := [])
      (.ref .fp16 [16,64] "a") (.ref .fp16 [64,16] weight)
      (.ref .real [16,16] acc) {})))
def sourceDotNode (acc weight : RegName) : ComputeStmt :=
  .assign .real [16,16] acc (.compute (.alg .fp32
    (.add .real (.consSame (.consSame .nil)) (.ref .real [16,16] acc)
      (.castFloat .fp32 .real (.dotTyped .fp16 .fp16 .fp32 (batch := [])
        (.ref .fp16 [16,64] "a") (.ref .fp16 [64,16] weight))))))
def dot1Index (p : Parameters) : Nat := if p.byte then 10 else 5
def dot3Index (p : Parameters) : Nat := if p.byte then 15 else 7
def directWeightNode (weight : RegName) : ComputeStmt :=
  .assign .fp16 [64,16] weight (.alg
    (.castFloat .f8e5 .fp16 (.withPrecision .f8e5 .fp32 (.ref .f8e5 [64,16] weight))))
def compiledBody (p : Parameters) : List ComputeStmt :=
  if p.byte then
    (body p).take 8 ++ [directWeightNode "b",carryNode "acc1" "b"] ++
    ((body p).drop 11).take 2 ++ [directWeightNode "c",carryNode "acc2" "c"] ++
    (body p).drop 16
  else
    (body p).take (dot1Index p) ++ [carryNode "acc1" "b"] ++
    ((body p).drop (dot1Index p+1)).take (dot3Index p-dot1Index p-1) ++
    [carryNode "acc2" "c"] ++ (body p).drop (dot3Index p+1)

/-- The two local source dot-plus-add sites are identified in the exact shared
quotation. The native byte branch separately contracts the two weight casts
to the observed direct E5M2-to-half opcode, without a numerical equality claim. -/
theorem source_dot_sites (p : Parameters) :
    (body p)[dot1Index p]? = some (sourceDotNode "acc1" "b") ∧
    (body p)[dot3Index p]? = some (sourceDotNode "acc2" "c") ∧
    body p = (body p).take (dot1Index p) ++ [sourceDotNode "acc1" "b"] ++
      ((body p).drop (dot1Index p+1)).take (dot3Index p-dot1Index p-1) ++
      [sourceDotNode "acc2" "c"] ++ (body p).drop (dot3Index p+1) := by
  rcases p with ⟨M,N,K,fp8,eps⟩
  cases fp8 <;> exact ⟨rfl,rfl,rfl⟩

def weightSingleNode (weight : RegName) : ComputeStmt :=
  .assign .real [64,16] weight (.compute (.alg .fp32
    (.castFloat .f8e5 .real (.ref .f8e5 [64,16] weight))))
def weightNarrowNode (weight : RegName) : ComputeStmt :=
  .assign .fp16 [64,16] weight (.alg
    (.castFloat .real .fp16 (.withPrecision .real .fp32 (.ref .real [64,16] weight))))

/-- Exact source locations contracted by the selected native compiler. The
source/Real interpretation retains both casts. Native interpretation uses
`directWeightNode` and does not assume exceptional-value cast equivalence. -/
theorem source_weight_cast_sites (p : Parameters) (byte : p.byte=Bool.true) :
    ((body p).drop 8).take 2=[weightSingleNode "b",weightNarrowNode "b"] ∧
    ((body p).drop 13).take 2=[weightSingleNode "c",weightNarrowNode "c"] := by
  rcases p with ⟨M,N,K,fp8,eps⟩
  cases fp8
  · cases byte
  · exact ⟨rfl,rfl⟩

#axiomsClean source_dot_sites
#axiomsClean source_weight_cast_sites
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
