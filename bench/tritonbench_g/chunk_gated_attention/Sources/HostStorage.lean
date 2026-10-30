import bench.tritonbench_g.chunk_gated_attention.Sources.RawStorage

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HostStorage
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
open ByteMemory RawStorage
noncomputable section

def innerCount (r : Admission.Inner) := r.B*r.H*r.config.NT*r.K*r.V
def finalCount (r : Admission.Inner) := r.B*r.H*r.K*r.V
def prefixCount (r : Admission.Prefix) := r.B*r.H*r.T*r.S
def historyCapacity (r : Admission.Inner) := r.config.heads*r.config.NT*r.config.PK*r.config.PV
def finalCapacity (r : Admission.Inner) := r.config.heads*r.config.PK*r.config.PV
def keyCapacity (r : Admission.Inner) := r.config.heads*r.config.PT*r.config.PK
def valueCapacity (r : Admission.Inner) := r.config.heads*r.config.PT*r.config.PV
def gateCapacity (r : Admission.Inner) := r.config.heads*r.config.PT*(if r.gateK then r.config.PK else r.config.PV)
def prefixCapacity (r : Admission.Prefix) := r.B*r.H*HostFP.padded r.T r.BT*HostFP.padded r.S 64
def copyWrites (original actual : Admission.Tensor) : List Admission.Tensor := if original=actual then [] else [actual]
def innerCopies (r : Admission.Inner) (key value gate seed : Admission.Tensor) : List Admission.Tensor :=
  copyWrites r.key key++copyWrites r.value value++copyWrites r.gate gate++
    (match r.seed with | none=>[] | some original=>copyWrites original seed)

/-- Ordinary fresh return placement, carrying no source or numerical result. -/
structure Place (device : Nat) (shape : List Nat) (caller : List Admission.Tensor) where
  objectId : Nat
  tensor : Admission.Tensor
  valid : tensor.Valid device shape
  fresh : ∀ t∈caller,Disjoint tensor.byteBase shape.prod t.byteBase t.shape.prod

abbrev InnerPlace (r : Admission.Inner) := Place r.query.device [r.B,r.H,r.config.NT*r.K,r.V] r.tensors
abbrev PrefixPlace (r : Admission.Prefix) := Place r.input.device [r.B,r.H,r.T,r.S] [r.input]

/-- Successful padding/zero-fill/copy results expose local coordinate laws.
Unpadded inputs retain their exact tensors. Read-only inputs may alias; only
new history/final allocations require freshness. These laws contain no kernel
execution, gate arithmetic, dot result, history value, or final-state value. -/
structure InnerPrepared (r : Admission.Inner) (before : Memory) where
  key : Admission.Tensor
  value : Admission.Tensor
  gate : Admission.Tensor
  seed : Admission.Tensor
  history : Admission.Tensor
  final : Admission.Tensor
  keyValid : key.Valid r.query.device [r.B,r.H,r.config.PT,r.config.PK]
  valueValid : value.Valid r.query.device [r.B,r.H,r.config.PT,r.config.PV]
  gateValid : gate.Valid r.query.device [r.B,r.H,r.config.PT,if r.gateK then r.config.PK else r.config.PV]
  seedValid : r.seed.isSome=Bool.true → seed.Valid r.query.device [r.B,r.H,r.config.PK,r.config.PV]
  historyValid : history.Valid r.query.device [r.B,r.H,r.config.NT*r.config.PK,r.config.PV]
  finalValid : r.final.isSome=Bool.true → final.Valid r.query.device [r.B,r.H,r.config.PK,r.config.PV]
  keyDirect : r.key.shape=[r.B,r.H,r.config.PT,r.config.PK] → key=r.key
  valueDirect : r.value.shape=[r.B,r.H,r.config.PT,r.config.PV] → value=r.value
  gateDirect : r.gate.shape=[r.B,r.H,r.config.PT,if r.gateK then r.config.PK else r.config.PV] → gate=r.gate
  seedDirect : ∀ t,r.seed=some t → t.shape=[r.B,r.H,r.config.PK,r.config.PV] → seed=t
  memory : Memory
  keyRead : ∀ i,i<keyCapacity r → read memory (key.byteBase+4*i)=HostData.key r before i
  valueRead : ∀ i,i<valueCapacity r → read memory (value.byteBase+4*i)=HostData.value r before i
  gateRead : ∀ i,i<gateCapacity r → read memory (gate.byteBase+4*i)=HostData.gate r before i
  seedRead : r.seed.isSome=Bool.true → ∀ i,i<finalCapacity r → read memory (seed.byteBase+4*i)=HostData.seed r before i
  callerFrame : ∀ t∈r.tensors,∀ j,j<4*t.shape.prod → memory (t.byteBase+j)=before (t.byteBase+j)
  frame : ∀ address,(∀ t∈innerCopies r key value gate seed,Outside t.byteBase t.shape.prod address) →
    memory address=before address
  historyFresh : ∀ t∈r.tensors++[key,value,gate,seed],
    Disjoint history.byteBase (historyCapacity r) t.byteBase t.shape.prod
  finalFresh : r.final.isSome=Bool.true → ∀ t∈r.tensors++[key,value,gate,seed,history],
    Disjoint final.byteBase (finalCapacity r) t.byteBase t.shape.prod

def InnerPrepared.tensors {r before} (p : InnerPrepared r before) : List Admission.Tensor :=
  [p.key,p.value,p.gate,p.history]++(if r.seed.isSome then [p.seed] else [])++(if r.final.isSome then [p.final] else [])
def InnerPrepared.check {r before} (p : InnerPrepared r before) : Bool := decide (∀ t∈p.tensors,Admission.Address t)

structure InnerFinish (r : Admission.Inner) (before : Memory) (p : InnerPrepared r before) where
  place : InnerPlace r
  direct : place.tensor.byteBase=p.history.byteBase → ∀ i,i < innerCount r → HostData.historyOffset r i=i
  other : place.tensor.byteBase≠p.history.byteBase →
    Disjoint place.tensor.byteBase (innerCount r) p.history.byteBase (historyCapacity r)
  finalFresh : r.final.isSome=Bool.true →
    Disjoint place.tensor.byteBase (innerCount r) p.final.byteBase (finalCapacity r)

structure PrefixPrepared (r : Admission.Prefix) (before : Memory) where
  input : Admission.Tensor
  output : Admission.Tensor
  inputValid : input.Valid r.input.device [r.B,r.H,HostFP.padded r.T r.BT,HostFP.padded r.S 64]
  outputValid : output.Valid r.input.device [r.B,r.H,HostFP.padded r.T r.BT,HostFP.padded r.S 64]
  inputDirect : r.input.shape=[r.B,r.H,HostFP.padded r.T r.BT,HostFP.padded r.S 64] → input=r.input
  memory : Memory
  inputRead : ∀ i,i<prefixCapacity r → read memory (input.byteBase+4*i)=HostData.prefixValues r before i
  callerFrame : ∀ j,j<4*r.input.shape.prod → memory (r.input.byteBase+j)=before (r.input.byteBase+j)
  frame : ∀ address,(∀ t∈copyWrites r.input input,Outside t.byteBase t.shape.prod address) →
    memory address=before address
  fresh : ∀ t∈[r.input,input],Disjoint output.byteBase (prefixCapacity r) t.byteBase t.shape.prod

def PrefixPrepared.check {r before} (p : PrefixPrepared r before) : Bool :=
  decide (Admission.Address p.input ∧ Admission.Address p.output)

structure PrefixFinish (r : Admission.Prefix) (before : Memory) (p : PrefixPrepared r before) where
  place : PrefixPlace r
  direct : place.tensor.byteBase=p.output.byteBase → ∀ i,i<prefixCount r → HostData.prefixOffset r i=i
  other : place.tensor.byteBase≠p.output.byteBase →
    Disjoint place.tensor.byteBase (prefixCount r) p.output.byteBase (prefixCapacity r)

theorem inner_place_count (r : Admission.Inner) (p : InnerPlace r) : p.tensor.shape.prod=innerCount r := by
  rw [p.valid.2.2.2.2.2.2.2.2]
  simp [innerCount,Nat.mul_assoc]
theorem prefix_place_count (r : Admission.Prefix) (p : PrefixPlace r) : p.tensor.shape.prod=prefixCount r := by
  rw [p.valid.2.2.2.2.2.2.2.2]
  simp [prefixCount,Nat.mul_assoc]

/-- Local copy/read laws are consumed by the exact DSL input views. -/
theorem key_load (r : Admission.Inner) (before : Memory) (p : InnerPrepared r before)
    (scratch : State NativeGraph.Expr) (i : Nat) (hi : i<keyCapacity r) :
    ((HostData.innerInput r before scratch).mem "k" i).read .real=
      .leaf (read p.memory (p.key.byteBase+4*i)) := by
  rw [p.keyRead i hi]
  simp [HostData.innerInput,Cell.read_same]
theorem value_load (r : Admission.Inner) (before : Memory) (p : InnerPrepared r before)
    (scratch : State NativeGraph.Expr) (i : Nat) (hi : i<valueCapacity r) :
    ((HostData.innerInput r before scratch).mem "v" i).read .real=
      .leaf (read p.memory (p.value.byteBase+4*i)) := by
  rw [p.valueRead i hi]
  simp [HostData.innerInput,Cell.read_same]
theorem gate_load (r : Admission.Inner) (before : Memory) (p : InnerPrepared r before)
    (scratch : State NativeGraph.Expr) (i : Nat) (hi : i<gateCapacity r) :
    ((HostData.innerInput r before scratch).mem "g" i).read .real=
      .leaf (read p.memory (p.gate.byteBase+4*i)) := by
  rw [p.gateRead i hi]
  simp [HostData.innerInput,Cell.read_same]
theorem seed_load (r : Admission.Inner) (before : Memory) (p : InnerPrepared r before)
    (scratch : State NativeGraph.Expr) (enabled : r.seed.isSome=Bool.true) (i : Nat) (hi : i<finalCapacity r) :
    ((HostData.innerInput r before scratch).mem "h0" i).read .fp32=
      .leaf (read p.memory (p.seed.byteBase+4*i)) := by
  rw [p.seedRead enabled i hi]
  simp [HostData.innerInput,Cell.read_same]
theorem prefix_load (r : Admission.Prefix) (before : Memory) (p : PrefixPrepared r before)
    (scratch : State NativeGraph.Expr) (i : Nat) (hi : i<prefixCapacity r) :
    ((HostData.prefixInput r before scratch).mem "s" i).read .fp32=
      .leaf (read p.memory (p.input.byteBase+4*i)) := by
  rw [p.inputRead i hi]
  simp [HostData.prefixInput,Cell.read_same]

#axiomsClean key_load
#axiomsClean value_load
#axiomsClean gate_load
#axiomsClean seed_load
#axiomsClean prefix_load
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HostStorage
