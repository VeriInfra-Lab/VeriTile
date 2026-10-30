import bench.repairs.tritonbench_g.cross_entropy1.Proofs.ScalarGraph

namespace VeriTile.Bench.Repairs.CrossEntropy1.Floating
noncomputable section

/-- Schedule data has no access to floating values. Its leaves are actual
input slots; each internal node is one ordered binary operation or word move. -/
inductive ReductionTree (width : Nat) where
  | input (index : Fin width)
  | identity (bits : Word)
  | combine (left right : ReductionTree width)
  | transfer (edge : Transfer) (value : ReductionTree width)

def ReductionTree.graph (op : ScalarBinary) : ReductionTree width → ScalarGraph width
  | .input i => .input i
  | .identity x => .raw x
  | .combine a b => .binary op (a.graph op) (b.graph op)
  | .transfer edge a => .transfer edge (a.graph op)

def ReductionTree.at (width index : Nat) (padding : Word) : ReductionTree width :=
  if h : index<width then .input ⟨index,h⟩ else .identity padding

def orderedFold (xs : List (ReductionTree width)) (identity : Word) : ReductionTree width :=
  xs.foldl ReductionTree.combine (.identity identity)

/-- Triton's register accumulation starts at the first loaded register,
without an inserted zero. Registers are accumulated in increasing slot order. -/
def registerFold (width threads lane count : Nat) (padding : Word) : ReductionTree width :=
  ((List.range (count-1)).map (fun i => ReductionTree.at width (lane+(i+1)*threads) padding)).foldl
    ReductionTree.combine (ReductionTree.at width lane padding)

def butterfly (offsets : List Nat) (xs : Nat → ReductionTree width) : Nat → ReductionTree width :=
  offsets.foldl (fun current offset lane => .combine (current lane)
    (.transfer (.shuffle (Nat.xor lane offset) lane offset) (current (Nat.xor lane offset)))) xs

/-- Blocked layout [sizePerThread=1, threadsPerWarp=32, warpsPerCTA=w].
Register accumulation precedes decreasing-offset butterfly; warp lane zero
writes one word, and the final warp reads these words before its butterfly.
Other layouts must supply their own checked structural schedule. -/
def tritonBlocked (width warps : Nat) (padding : Word) : ReductionTree width :=
  let threads := 32*warps
  let registers := fun lane => registerFold width threads lane ((width+threads-1)/threads) padding
  let lanes := butterfly ([16,8,4,2,1].filter (fun offset => offset<width)) registers
  if width≤32 then lanes 0
  else
    let activeWarps := min warps ((width+31)/32)
    let shared := fun lane => ReductionTree.transfer (.shared 0 1 (4*lane) (32*lane) lane) (lanes (32*lane))
    (butterfly ([16,8,4,2,1].filter (fun offset => offset<activeWarps)) shared) 0

/-- All selection-dependent reduction order is explicit structural data.
Host plans are selected using full TensorIterator/device descriptors, not
merely their reduction length. The selected profile is shared by the two
scalar interpretations; it is never a numerical equality assumption. -/
structure ReductionProfile where
  laneSum : (width : Nat) → ReductionTree width
  laneMaximum : (width : Nat) → ReductionTree width
  hostSum : (width : Nat) → ReductionTree width
  hostMaximum : (width : Nat) → ReductionTree width
  hostExponentialSum : (width : Nat) → ReductionTree width
  rankSum : (width : Nat) → ReductionTree width

def ReductionProfile.sumGraph (p : ReductionProfile) (width : Nat) : ScalarGraph width :=
  (p.laneSum width).graph .add
def ReductionProfile.maximumGraph (p : ReductionProfile) (width : Nat) : ScalarGraph width :=
  (p.laneMaximum width).graph .maximum
def ReductionProfile.hostSumGraph (p : ReductionProfile) (width : Nat) : ScalarGraph width :=
  (p.hostSum width).graph .hostAdd
def ReductionProfile.rankSumGraph (p : ReductionProfile) (width : Nat) : ScalarGraph width :=
  (p.rankSum width).graph .rankAdd

/-- Complete Torch logsumexp graph: amax, abs-infinity mask, broadcast
subtraction, elementwise exp, its independently selected sum plan, log, add.
Every tensor boundary is already a scalar graph boundary; no FMA is inserted. -/
def ReductionProfile.hostLSEGraph (p : ReductionProfile) (width : Nat) : ScalarGraph width :=
  if width=0 then .unary .hostLog (.raw 0)
  else
    let maximum := ScalarGraph.unary .infinityToZero ((p.hostMaximum width).graph .hostMaximum)
    let exponentials := fun i => ScalarGraph.unary .hostExp (.binary .hostSub (.input i) maximum)
    let total := ((p.hostExponentialSum width).graph .hostAdd).bind exponentials
    .binary .hostAdd (.unary .hostLog total) maximum

theorem reduction_equiv (h : Obligations n r) (t : ReductionTree width) (op : ScalarBinary)
    (xs ys : Fin width → Word) (inputs : ∀ i, xs i=ys i) :
    (t.graph op).eval n xs=(t.graph op).eval r ys := ScalarGraph.equiv h xs ys inputs _

theorem host_lse_equiv (h : Obligations n r) (p : ReductionProfile)
    (xs ys : Fin width → Word) (inputs : ∀ i, xs i=ys i) :
    (p.hostLSEGraph width).eval n xs=(p.hostLSEGraph width).eval r ys :=
  ScalarGraph.equiv h xs ys inputs _

#axiomsClean reduction_equiv
#axiomsClean host_lse_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Floating
