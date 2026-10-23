import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholeBackendTile

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.NativePlan2
open Backend
set_option maxHeartbeats 20000000
set_option maxRecDepth 100000
-- Ordered native row graph SHA256 845d687e9c0aded18572c6ee7654840ff1d82a0c48b6b7386def202d8ac18250
def node0 : SumTree 16 := .leaf ⟨0,by decide⟩
def node1 : SumTree 16 := .leaf ⟨1,by decide⟩
def node2 : SumTree 16 := .leaf ⟨2,by decide⟩
def node3 : SumTree 16 := .leaf ⟨3,by decide⟩
def node4 : SumTree 16 := .leaf ⟨4,by decide⟩
def node5 : SumTree 16 := .leaf ⟨5,by decide⟩
def node6 : SumTree 16 := .leaf ⟨6,by decide⟩
def node7 : SumTree 16 := .leaf ⟨7,by decide⟩
def node8 : SumTree 16 := .leaf ⟨8,by decide⟩
def node9 : SumTree 16 := .leaf ⟨9,by decide⟩
def node10 : SumTree 16 := .leaf ⟨10,by decide⟩
def node11 : SumTree 16 := .leaf ⟨11,by decide⟩
def node12 : SumTree 16 := .leaf ⟨12,by decide⟩
def node13 : SumTree 16 := .leaf ⟨13,by decide⟩
def node14 : SumTree 16 := .leaf ⟨14,by decide⟩
def node15 : SumTree 16 := .fused ⟨1,by decide⟩ node0
def node16 : SumTree 16 := .fused ⟨2,by decide⟩ node1
def node17 : SumTree 16 := .fused ⟨3,by decide⟩ node2
def node18 : SumTree 16 := .fused ⟨4,by decide⟩ node3
def node19 : SumTree 16 := .fused ⟨5,by decide⟩ node4
def node20 : SumTree 16 := .fused ⟨6,by decide⟩ node5
def node21 : SumTree 16 := .fused ⟨7,by decide⟩ node6
def node22 : SumTree 16 := .fused ⟨8,by decide⟩ node7
def node23 : SumTree 16 := .fused ⟨9,by decide⟩ node8
def node24 : SumTree 16 := .fused ⟨10,by decide⟩ node9
def node25 : SumTree 16 := .fused ⟨11,by decide⟩ node10
def node26 : SumTree 16 := .fused ⟨12,by decide⟩ node11
def node27 : SumTree 16 := .fused ⟨13,by decide⟩ node12
def node28 : SumTree 16 := .fused ⟨14,by decide⟩ node13
def node29 : SumTree 16 := .fused ⟨15,by decide⟩ node14
def node30 : SumTree 16 := .add node16 node0
def node31 : SumTree 16 := .add node17 node15
def node32 : SumTree 16 := .add node18 node16
def node33 : SumTree 16 := .add node19 node17
def node34 : SumTree 16 := .add node20 node18
def node35 : SumTree 16 := .add node21 node19
def node36 : SumTree 16 := .add node22 node20
def node37 : SumTree 16 := .add node23 node21
def node38 : SumTree 16 := .add node24 node22
def node39 : SumTree 16 := .add node25 node23
def node40 : SumTree 16 := .add node26 node24
def node41 : SumTree 16 := .add node27 node25
def node42 : SumTree 16 := .add node28 node26
def node43 : SumTree 16 := .add node29 node27
def node44 : SumTree 16 := .add node32 node0
def node45 : SumTree 16 := .add node33 node15
def node46 : SumTree 16 := .add node34 node30
def node47 : SumTree 16 := .add node35 node31
def node48 : SumTree 16 := .add node36 node32
def node49 : SumTree 16 := .add node37 node33
def node50 : SumTree 16 := .add node38 node34
def node51 : SumTree 16 := .add node39 node35
def node52 : SumTree 16 := .add node40 node36
def node53 : SumTree 16 := .add node41 node37
def node54 : SumTree 16 := .add node42 node38
def node55 : SumTree 16 := .add node43 node39
def node56 : SumTree 16 := .add node48 node0
def node57 : SumTree 16 := .add node49 node15
def node58 : SumTree 16 := .add node50 node30
def node59 : SumTree 16 := .add node51 node31
def node60 : SumTree 16 := .add node52 node44
def node61 : SumTree 16 := .add node53 node45
def node62 : SumTree 16 := .add node54 node46
def node63 : SumTree 16 := .add node55 node47

def row : Fin 16 → SumTree 16 := ![node0,node15,node30,node31,node44,node45,node46,node47,node56,node57,node58,node59,node60,node61,node62,node63]
theorem prefix_correct : ∀ j : Fin 16, (row j).leaves.Perm (List.range (j.val+1)) := by decide
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.NativePlan2
