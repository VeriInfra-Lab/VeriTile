import bench.tritonbench_g.context_attn_llama.Proofs.HostReadback
import bench.repairs.tritonbench_g.context_attn_nopad.Contract

namespace VeriTile.Bench.Repairs.ContextAttnLlama.Checked
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Formula Program
set_option maxRecDepth 24000
set_option synthInstance.maxSize 512

abbrev View := VeriTile.Bench.Repairs.ContextAttnFwd.Checked.View
abbrev separate := VeriTile.Bench.Repairs.ContextAttnFwd.Checked.separate
structure Views where
  (q k v o start sequence prompt request pages : View)
structure Dimensions where
  (queryRows cacheRows requests tokens : Nat)
def all (v : Views) := [v.q,v.k,v.v,v.o,v.start,v.sequence,v.prompt,v.request,v.pages]
def reads (v : Views) := [v.q,v.k,v.v,v.start,v.sequence,v.prompt,v.request,v.pages]
def metadata (v : Views) := [v.start,v.sequence,v.prompt,v.request,v.pages]

def sourceStrides (c : Call) : Prop :=
  c.layout.qRow=c.layout.heads*c.layout.D ∧ c.layout.qHead=c.layout.D ∧ c.layout.qDim=1 ∧
  c.layout.outRow=c.layout.heads*c.layout.D ∧ c.layout.outHead=c.layout.D ∧ c.layout.outDim=1 ∧
  c.layout.kRow=c.kvHeads*c.layout.D ∧ c.layout.kHead=c.layout.D ∧ c.layout.kDim=1 ∧
  c.layout.vRow=c.kvHeads*c.layout.D ∧ c.layout.vHead=c.layout.D ∧ c.layout.vDim=1

def tensorChecks (c : Call) (d : Dimensions) (v : Views) : Prop :=
  (∀ x∈[v.q,v.o],x.metadata.shape=[d.queryRows,c.layout.heads,c.layout.D] ∧
    x.metadata.dtype=.floating .fp32 ∧ x.metadata.isContiguous=Bool.true ∧
    x.strides=[c.layout.heads*c.layout.D,c.layout.D,1]) ∧
  (∀ x∈[v.k,v.v],x.metadata.shape=[d.cacheRows,c.kvHeads,c.layout.D] ∧
    x.metadata.dtype=.floating .fp32 ∧ x.metadata.isContiguous=Bool.true ∧
    x.strides=[c.kvHeads*c.layout.D,c.layout.D,1]) ∧
  (∀ x∈[v.start,v.sequence,v.prompt,v.request],x.metadata.shape=[c.batches] ∧
    x.metadata.dtype=.other "int32" ∧ x.metadata.isContiguous=Bool.true) ∧
  (v.pages.metadata.shape=[d.requests,d.tokens] ∧ v.pages.metadata.dtype=.other "int32" ∧
    v.pages.metadata.isContiguous=Bool.true ∧ v.pages.strides=[d.tokens,1]) ∧
  v.q.metadata.device.kind="cuda" ∧
  (∀ x∈all v,x.metadata.layout=.strided ∧ x.metadata.device=v.q.metadata.device ∧
    x.metadata.isNeg=Bool.false ∧ x.conjugate=Bool.false ∧ x.storageBytes%4=0 ∧ x.storageBytes<2^63 ∧
    (x.metadata.numel=0 ∨ x.metadata.dataPtr.toNat%4=0)) ∧
  (c.batches=0 ∨ ∀ x∈reads v,separate v.o x)

def geometryChecks (c : Call) (d : Dimensions) (extent : RegionBounds) : Prop :=
  sourceStrides c ∧ c.layout.reqRow=d.tokens ∧ c.layout.reqToken=1 ∧
  0 < c.kvHeads ∧ 0 < c.layout.kvGroup ∧ c.layout.heads=c.kvHeads*c.layout.kvGroup ∧
  (∀ b : Fin c.batches,c.start b+c.length b≤d.queryRows ∧ c.length b≤c.maxInput ∧
    c.length b+c.prompt b≤d.tokens ∧ c.request b < d.requests ∧
    ∀ j : Fin (c.length b+c.prompt b),c.page b j < d.cacheRows) ∧
  (∀ b z : Fin c.batches,0 < c.length b → 0 < c.length z → b≠z →
    c.start b+c.length b≤c.start z ∨ c.start z+c.length z≤c.start b) ∧
  (∀ rg∈[c.regions.query,c.regions.out],d.queryRows*c.layout.heads*c.layout.D≤extent rg) ∧
  (∀ rg∈[c.regions.key,c.regions.value],d.cacheRows*c.kvHeads*c.layout.D≤extent rg) ∧
  (∀ rg∈[c.regions.start,c.regions.sequence,c.regions.prompt,c.regions.request],c.batches≤extent rg) ∧
  d.requests*d.tokens≤extent c.regions.pages

def payloadBudget (rows heads width : Nat) (v : View) : Prop :=
  v.storageOffset+(rows-1)*(heads*width)+(heads-1)*width+(width-1)<2^61 ∧
  heads*width<2^61 ∧ width<2^61

def nativeChecks (c : Call) (d : Dimensions) (v : Views) (requestedMax : Nat) : Prop :=
  requestedMax<2^31 ∧ c.maxInput≤requestedMax ∧ c.batches<2^31 ∧ c.batches*c.layout.heads≤65535 ∧
  (∀ x∈[v.q,v.o],payloadBudget (d.queryRows+requestedMax+128) c.layout.heads c.layout.D x) ∧
  (∀ x∈[v.k,v.v],payloadBudget (max d.cacheRows 1) c.kvHeads c.layout.D x) ∧
  (∀ x∈[v.start,v.sequence,v.prompt,v.request],
    VeriTile.Bench.Repairs.ContextAttnFwd.Checked.vectorBudget x c.batches) ∧
  v.pages.storageOffset+(d.requests-1)*d.tokens+(d.tokens+128-1)<2^61 ∧ d.tokens<2^61

def metadataChecks (c : Call) (start sequence prompt request pages : Nat → Int) : Prop :=
  (∀ b : Fin c.batches,start b=(c.start b : Int) ∧
    sequence b=((c.length b+c.prompt b : Nat) : Int) ∧ prompt b=(c.prompt b : Int) ∧
    request b=(c.request b : Int) ∧
    ∀ j : Fin (c.length b+c.prompt b),pages (c.layout.reqRow*c.request b+c.layout.reqToken*j)=(c.page b j : Int)) ∧
  c.maxInput=(Finset.univ.sup fun b : Fin c.batches => c.length b)

instance (c : Call) : Decidable (sourceStrides c) := by unfold sourceStrides; infer_instance
instance (c : Call) (d : Dimensions) (v : Views) : Decidable (tensorChecks c d v) := by
  unfold tensorChecks all reads; infer_instance
instance (c : Call) (d : Dimensions) (extent : RegionBounds) : Decidable (geometryChecks c d extent) := by
  unfold geometryChecks; infer_instance
instance (rows heads width : Nat) (v : View) : Decidable (payloadBudget rows heads width v) := by
  unfold payloadBudget; infer_instance
instance (c : Call) (d : Dimensions) (v : Views) (requestedMax : Nat) : Decidable (nativeChecks c d v requestedMax) := by
  unfold nativeChecks; infer_instance
instance (c : Call) (start sequence prompt request pages : Nat → Int) :
    Decidable (metadataChecks c start sequence prompt request pages) := by unfold metadataChecks; infer_instance

def InputContract (c : Call) (d : Dimensions) (v : Views) (extent : RegionBounds) (requestedMax : Nat)
    (start sequence prompt request pages : Nat → Int) : Prop :=
  tensorChecks c d v ∧ geometryChecks c d extent ∧ nativeChecks c d v requestedMax ∧
    metadataChecks c start sequence prompt request pages

def checkInputs (c : Call) (d : Dimensions) (v : Views) (extent : RegionBounds) (requestedMax : Nat)
    (start sequence prompt request pages : Nat → Int) : Bool :=
  decide (tensorChecks c d v) && decide (geometryChecks c d extent) && decide (nativeChecks c d v requestedMax) &&
    decide (metadataChecks c start sequence prompt request pages)

theorem checkInputs_iff (c : Call) (d : Dimensions) (v : Views) (extent : RegionBounds) (requestedMax : Nat)
    (start sequence prompt request pages : Nat → Int) :
    checkInputs c d v extent requestedMax start sequence prompt request pages=Bool.true ↔
      InputContract c d v extent requestedMax start sequence prompt request pages := by
  simp only [checkInputs,InputContract,Bool.and_eq_true,decide_eq_true_eq,and_assoc]

#axiomsClean checkInputs_iff
end VeriTile.Bench.Repairs.ContextAttnLlama.Checked
