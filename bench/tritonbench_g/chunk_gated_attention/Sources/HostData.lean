import bench.tritonbench_g.chunk_gated_attention.Sources.Admission
import bench.tritonbench_g.chunk_gated_attention.Sources.ScalarNative

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HostData
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
open ByteMemory
noncomputable section

def tensorRead (before : Memory) (tensor : Admission.Tensor) (i : Nat) : Word :=
  read before (tensor.byteBase+4*i)
def dense {α : Type} (times features : Nat) (f : Nat→Nat→Nat→α) (off : Nat) : α :=
  f (off/(times*features)) ((off/features)%times) (off%features)
def logical (before : Memory) (tensor : Admission.Tensor) (times features h t f : Nat) : Word :=
  tensorRead before tensor (h*(times*features)+t*features+f)

def key (r : Admission.Inner) (before : Memory) : Nat→Word :=
  dense r.config.PT r.config.PK (fun h t k=>
    if t<r.T ∧ k<r.K then logical before r.key r.T r.K h t k else 0)
def value (r : Admission.Inner) (before : Memory) : Nat→Word :=
  dense r.config.PT r.config.PV (fun h t v=>
    if t<r.T ∧ v<r.V then logical before r.value r.T r.V h t v else 0)
def gate (r : Admission.Inner) (before : Memory) : Nat→Word :=
  dense r.config.PT (if r.gateK then r.config.PK else r.config.PV) (fun h t f=>
    if 0<r.T ∧ f<(if r.gateK then r.K else r.V) then
      logical before r.gate r.T (if r.gateK then r.K else r.V) h (min t (r.T-1)) f else 0)
def seed (r : Admission.Inner) (before : Memory) : Nat→Word :=
  dense r.config.PK r.config.PV (fun h k v=>match r.seed with
    | none=>0
    | some tensor=>if k<r.K ∧ v<r.V then logical before tensor r.K r.V h k v else 0)

/-- Source K/V/G and history have the FP32-profile DSL real view; the seed
and final pointer dtypes are explicit FP32. This preserves their actual loads. -/
def innerInput (r : Admission.Inner) (before : Memory) (scratch : State NativeGraph.Expr) : State NativeGraph.Expr :=
  {scratch with mem:=fun region i=>
    if region="k" then .mk .real (.leaf (key r before i))
    else if region="v" then .mk .real (.leaf (value r before i))
    else if region="g" then .mk .real (.leaf (gate r before i))
    else if region="h0" then .mk .fp32 (.leaf (seed r before i)) else scratch.mem region i}

def prefixValues (r : Admission.Prefix) (before : Memory) : Nat→Word :=
  dense (HostFP.padded r.T r.BT) (HostFP.padded r.S 64) (fun h t f=>
    if t<r.T ∧ f<r.S then logical before r.input r.T r.S h t f else 0)
def prefixInput (r : Admission.Prefix) (before : Memory) (scratch : State NativeGraph.Expr) : State NativeGraph.Expr :=
  {scratch with mem:=fun region i=>if region="s" then .mk .fp32 (.leaf (prefixValues r before i)) else scratch.mem region i}

def historyOffset (r : Admission.Inner) (i : Nat) : Nat :=
  let c:=r.config
  i/(c.NT*r.K*r.V)*c.NT*c.PK*c.PV+(i/(r.K*r.V))%c.NT*c.PK*c.PV+(i/r.V)%r.K*c.PV+i%r.V
def finalOffset (r : Admission.Inner) (i : Nat) : Nat :=
  let c:=r.config
  i/(r.K*r.V)*c.PK*c.PV+(i/r.V)%r.K*c.PV+i%r.V
def prefixOffset (r : Admission.Prefix) (i : Nat) : Nat :=
  i/(r.T*r.S)*HostFP.padded r.T r.BT*HostFP.padded r.S 64+(i/r.S)%r.T*HostFP.padded r.S 64+i%r.S

def innerResult (r : Admission.Inner) (before : Memory) (scratch : State NativeGraph.Expr) :=
  NativeGrid.innerResult r.config (innerInput r before scratch)
def prefixResult (r : Admission.Prefix) (BS : Nat) (before : Memory) (scratch : State NativeGraph.Expr) :=
  NativeGrid.prefixResult (r.config BS) (prefixInput r before scratch)

def historyWord (I : ScalarNative.Primitives) (r : Admission.Inner) (before : Memory)
    (scratch : State NativeGraph.Expr) (i : Nat) : Word :=
  ScalarNative.eval I (((innerResult r before scratch).mem "h" (historyOffset r i)).read .real)
def finalWord (I : ScalarNative.Primitives) (r : Admission.Inner) (before : Memory)
    (scratch : State NativeGraph.Expr) (i : Nat) : Word :=
  ScalarNative.eval I (((innerResult r before scratch).mem "ht" (finalOffset r i)).read .fp32)
def prefixWord (I : ScalarNative.Primitives) (r : Admission.Prefix) (BS : Nat) (before : Memory)
    (scratch : State NativeGraph.Expr) (i : Nat) : Word :=
  ScalarNative.eval I (((prefixResult r BS before scratch).mem "o" (prefixOffset r i)).read .fp32)

/-- The empty path is the actual PyTorch FP32 zero-plus-seed operation,
including the addition that determines the sign of an initial zero. -/
def emptyFinal (I : ScalarNative.Primitives) (r : Admission.Inner) (before : Memory) (i : Nat) : Word :=
  match r.seed with
  | none=>0
  | some tensor=>I.binary none .fp32 .add 0 (tensorRead before tensor i)

theorem inner_source (r : Admission.Inner) (before : Memory) (scratch : State NativeGraph.Expr) :
    HostFP.batches NativeValue.algebra (NativeGrid.program (NativeGrid.innerSource r.config))
      (NativeGrid.innerOwns r.config) r.config.plans (innerInput r before scratch)=
      some (innerResult r before scratch) := NativeGrid.inner_complete _ _
theorem prefix_source (r : Admission.Prefix) (BS : Nat) (before : Memory) (scratch : State NativeGraph.Expr) :
    HostFP.batches NativeValue.algebra (NativeGrid.program (NativeGrid.prefixSource (r.config BS)))
      (NativeGrid.prefixOwns (r.config BS)) (r.config BS).plans (prefixInput r before scratch)=
      some (prefixResult r BS before scratch) := NativeGrid.prefix_complete _ _

#specNonCircular historyWord avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.range,HostFP.batches]
#specNonCircular finalWord avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.range,HostFP.batches]
#specNonCircular prefixWord avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.range,HostFP.batches]
#axiomsClean inner_source
#axiomsClean prefix_source
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HostData
