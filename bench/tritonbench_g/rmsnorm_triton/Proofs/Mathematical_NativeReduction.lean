import bench.tritonbench_g.rmsnorm_triton.Sources.Mathematical_WholeSource
import VeriTile.Meta.StatementAudit
import Mathlib.Data.List.Sort

namespace VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

/-- The four-warp observed profiles, with sizePerThread=1 or 4.
The scalar layout owns tid+128*q; the vector layout owns
4*tid+(q%4)+512*(q/4), for q=0..7. Operands of the native adds
are retained exactly; a multi-iteration profile places each later register
on the left of the running intra-thread reduction. -/
def nativeLeaf (n : Nat) : SumTree := .leaf ⟨n%1024,Nat.mod_lt _ (by decide)⟩

def nativeThreadIndex (vector : Bool) (tid q : Nat) : Nat :=
  if vector then tid*4+q%4+(q/4)*512 else tid+q*128

def nativeThreadTree (vector multi : Bool) (tid : Nat) : SumTree :=
  (List.range 6).foldl (fun acc q =>
    if multi then .add (nativeLeaf (nativeThreadIndex vector tid (q+2))) acc
    else .add acc (nativeLeaf (nativeThreadIndex vector tid (q+2))))
    (.add (nativeLeaf (nativeThreadIndex vector tid 0)) (nativeLeaf (nativeThreadIndex vector tid 1)))

/-- Exact butterfly sequence 16,8,4,2,1. Only lane zero's root is stored
for each warp; no equality between different rounded lane roots is asserted. -/
def nativeWarpTree (vector multi : Bool) (warp : Nat) : Nat → Nat → SumTree
  | 0,lane => nativeThreadTree vector multi (warp*32+lane)
  | step+1,lane => .add (nativeWarpTree vector multi warp step lane)
      (nativeWarpTree vector multi warp step (Nat.xor lane (2^(4-step))))

/-- Inter-warp butterfly 2,1, lane zero store, then shared[0] broadcast.
This is one explicit instruction tree for each of the four observed profiles. -/
def nativeTree (vector multi : Bool) : SumTree :=
  .add (.add (nativeWarpTree vector multi 0 5 0) (nativeWarpTree vector multi 2 5 0))
       (.add (nativeWarpTree vector multi 1 5 0) (nativeWarpTree vector multi 3 5 0))

theorem nativeTree_sorted (vector multi : Bool) :
    List.insertionSort (fun a b : Fin 1024 => a.val ≤ b.val) (nativeTree vector multi).leaves =
      List.finRange 1024 := by
  cases vector <;> cases multi <;> decide +kernel

theorem nativeTree_complete (vector multi : Bool) :
    (nativeTree vector multi).leaves.Perm (List.finRange 1024) := by
  rw [← nativeTree_sorted vector multi]
  exact (List.perm_insertionSort _ _).symm

/-- The initial-loop add order is a separate observed structural choice.
It cannot be inferred solely from the width or the local reduction order. -/
def nativeReduction (vector multi initialRightZero : Bool) : Reduction :=
  ⟨nativeTree vector multi,nativeTree_complete _ _,initialRightZero⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
