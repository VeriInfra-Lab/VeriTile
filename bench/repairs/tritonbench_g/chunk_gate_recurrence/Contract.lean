import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.SourceBounds
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.StorageGeometry
import bench.repairs.tritonbench_g.rotary_emb.Contract

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Checked
open Machine
abbrev Tensor := VeriTile.Bench.Repairs.RotaryEmb.Checked.Tensor

def tensorShape (c : Shape) (r : Region) : List Nat := match r with
  | .x | .output | .incoming | .dx => [c.B,c.H,c.N,c.K,c.V]
  | .gate | .dgate => [c.B,c.H,c.N]
  | .seed | .dseed => [c.B,c.H,c.K,c.V]
  | .tileGate => [c.heads,c.N,c.KT,c.VT]
  | .caller => []

def present (c : Shape) (seed : Bool) (r : Region) : Bool := match r with
  | .seed => seed
  | .dseed => decide (0 < c.N) || seed
  | .tileGate => decide (0 < c.N)
  | .caller => Bool.false
  | _ => Bool.true

def span (t : Tensor) : Nat :=
  if t.shape.prod=0 then 0 else 1+((t.shape.zip t.strides).map (fun p => (p.1-1)*p.2)).sum

def allocation (t : Tensor) : Allocation := ⟨t.allocationBase,t.allocationBytes,t.base⟩

/-- Ordinary contiguous/view/allocation metadata, separate from all source
execution. The fresh-allocation laws do not contain values or output formulas. -/
structure Runtime (c : Shape) (seed : Bool) where
  tensor : Region → Tensor
  seedStrides : SeedStrides
  device : Nat
  currentDevice : Nat
  callerLive : List Allocation
  /-- Absent allocation roles use an empty ordinary descriptor. -/
  absent : ∀ r,present c seed r=Bool.false →
    (allocation (tensor r)).byteLength=0 ∧ (allocation (tensor r)).storageBase=0
  freshCaller : ∀ r,Written r → ∀ a∈callerLive,(allocation (tensor r)).Disjoint a
  freshOutput : ∀ r q,Written r → Written q → r≠q → (allocation (tensor r)).Disjoint (allocation (tensor q))
  /-- Successful allocation is fresh against the live inputs of its own phase.
  Incoming backward gradients may share the earlier forward output. -/
  freshForward : ∀ r,r=.x ∨ r=.gate ∨ r=.seed →
    (allocation (tensor .output)).Disjoint (allocation (tensor r))
  freshBackward : ∀ r,r=.dx ∨ r=.tileGate ∨ r=.dseed ∨ r=.dgate →
    ∀ q,q=.incoming ∨ q=.gate ∨ q=.output →
      (allocation (tensor r)).Disjoint (allocation (tensor q))

def Runtime.abi (runtime : Runtime c seed) : ABI :=
  ⟨fun r => allocation (runtime.tensor r),runtime.seedStrides,runtime.callerLive⟩

def tensorCheck (c : Shape) (seed : Bool) (runtime : Runtime c seed) : Bool :=
  allRegions.all fun r =>
    if present c seed r then
      let t := runtime.tensor r
      decide (t.shape=tensorShape c r ∧ t.strided=Bool.true ∧ t.cuda=Bool.true ∧
        t.device=runtime.device ∧ t.dtype="float32" ∧ t.elementBytes=4 ∧
        t.negative=Bool.false ∧ t.conjugate=Bool.false ∧ span t=extent c runtime.seedStrides r ∧
        (r≠.seed → t.contiguous=Bool.true) ∧
        (r=.seed → t.strides=[runtime.seedStrides.batch,runtime.seedStrides.head,runtime.seedStrides.key,runtime.seedStrides.value]) ∧
        (extent c runtime.seedStrides r=0 ∨
          (t.base=t.allocationBase+4*t.storageOffset ∧ t.base%4=0 ∧
            4*t.storageOffset+4*extent c runtime.seedStrides r ≤ t.allocationBytes ∧
            t.allocationBase+t.allocationBytes ≤ 2^64)))
    else Bool.true

def check (c : Shape) (seed : Bool) (runtime : Runtime c seed) : Bool :=
  tensorCheck c seed runtime && (decide (c.N=0) ||
    (decide (runtime.currentDevice=runtime.device ∧ c.heads < 2^31 ∧ c.KT ≤ 65535 ∧ c.VT ≤ 65535) &&
      rangeCheck c runtime.seedStrides && pointerCheck c runtime.seedStrides seed runtime.abi.allocation))

end VeriTile.Bench.Repairs.ChunkGateRecurrence.Checked
