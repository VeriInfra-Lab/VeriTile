import bench.tritonbench_g.context_attn_bloom.Proofs.Correct
import bench.tritonbench_g.context_attn_llama.Proofs.CheckedBridge

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Checked
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host (View3)
set_option maxRecDepth 24000

abbrev View := VeriTile.Bench.Repairs.ContextAttnFwd.Checked.View
abbrev separate := VeriTile.Bench.Repairs.ContextAttnFwd.Checked.separate
structure Views where
  (q k v o start sequence prompt request pages : View)
def all (v : Views) := [v.q,v.k,v.v,v.o,v.start,v.sequence,v.prompt,v.request,v.pages]
def reads (v : Views) := [v.q,v.k,v.v,v.start,v.sequence,v.prompt,v.request,v.pages]
def readRegions (r : Regions) := [r.query,r.key,r.value,r.start,r.sequence,r.prompt,r.request,r.tokens]
def metadataRegions (r : Regions) := [r.start,r.sequence,r.prompt,r.request,r.tokens]
def Labels (r : Regions) : Prop := ∀ rg∈readRegions r,r.out≠rg

def canonical (v : View3) : Prop := v.rowStride=v.heads*v.dims ∧ v.headStride=v.dims ∧ v.dimStride=1

def sourceStrides (a : Host.Arguments) : Prop :=
  canonical a.q ∧ canonical a.k ∧ canonical a.v ∧ canonical a.o ∧
  a.requestRowStride=a.requestColumns ∧ a.requestTokenStride=1

def tensorChecks (a : Host.Arguments) (v : Views) : Prop :=
  (∀ pair∈[(v.q,a.q),(v.k,a.k),(v.v,a.v),(v.o,a.o)],
    pair.1.metadata.shape=[pair.2.rows,pair.2.heads,pair.2.dims] ∧
    pair.1.strides=[pair.2.rowStride,pair.2.headStride,pair.2.dimStride] ∧
    pair.1.metadata.dtype=.floating .fp32 ∧ pair.1.metadata.isContiguous=Bool.true ∧ pair.2.negative=Bool.false) ∧
  (∀ x∈[v.start,v.sequence,v.prompt,v.request],x.metadata.shape=[a.batch] ∧
    x.metadata.dtype=.other "int32" ∧ x.metadata.isContiguous=Bool.true) ∧
  (v.pages.metadata.shape=[a.requestRows,a.requestColumns] ∧ v.pages.metadata.dtype=.other "int32" ∧
    v.pages.metadata.isContiguous=Bool.true ∧ v.pages.strides=[a.requestColumns,1]) ∧
  v.q.metadata.device.kind="cuda" ∧
  (∀ x∈all v,x.metadata.layout=.strided ∧ x.metadata.device=v.q.metadata.device ∧
    x.metadata.isNeg=Bool.false ∧ x.conjugate=Bool.false ∧ x.storageBytes%4=0 ∧ x.storageBytes<2^63 ∧
    (x.metadata.numel=0 ∨ x.metadata.dataPtr.toNat%4=0)) ∧
  (a.batch=0 ∨ ∀ x∈reads v,separate v.o x)

def geometryChecks (a : Host.Arguments) (r : Regions) (extent : RegionBounds) : Prop :=
  sourceStrides a ∧ 0 < a.q.heads ∧ 0 < a.k.heads ∧ a.k.heads ∣ a.q.heads ∧
  a.q.dims=a.k.dims ∧ a.k.dims=a.v.dims ∧ a.o.dims=a.k.dims ∧
  a.o.heads=a.q.heads ∧ a.v.heads=a.k.heads ∧ a.o.rows=a.q.rows ∧ a.v.rows=a.k.rows ∧
  16≤a.q.dims ∧ a.q.dims≤256 ∧
  a.q.rows*a.q.heads*a.q.dims≤extent r.query ∧
  a.k.rows*a.k.heads*a.k.dims≤extent r.key ∧
  a.v.rows*a.v.heads*a.v.dims≤extent r.value ∧
  a.o.rows*a.o.heads*a.o.dims≤extent r.out ∧
  (∀ rg∈[r.start,r.sequence,r.prompt,r.request],a.batch≤extent rg) ∧
  a.requestRows*a.requestColumns≤extent r.tokens

def metadataChecks (a : Host.Arguments) (u : Inputs) : Prop :=
  (∀ b : Fin a.batch,
    0≤u.start b ∧ 0≤u.prompt b ∧ u.prompt b≤u.sequence b ∧
    u.sequence b-u.prompt b≤a.maxInput ∧
    u.start b+(u.sequence b-u.prompt b)≤a.q.rows ∧
    u.sequence b≤a.requestColumns ∧ 0≤u.request b ∧ u.request b<a.requestRows ∧
    ∀ j : Fin (u.sequence b).toNat,
      0≤u.tokens (a.requestColumns*(u.request b).toNat+j) ∧
      u.tokens (a.requestColumns*(u.request b).toNat+j)<a.k.rows) ∧
  (∀ b c : Fin a.batch,0<u.sequence b-u.prompt b → 0<u.sequence c-u.prompt c → b≠c →
    u.start b+(u.sequence b-u.prompt b)≤u.start c ∨
    u.start c+(u.sequence c-u.prompt c)≤u.start b) ∧
  a.maxInput=Finset.univ.sup (fun b : Fin a.batch => (u.sequence b-u.prompt b).toNat)

def payloadBudget (rows heads width padded : Nat) (v : View) : Prop :=
  v.storageOffset+(rows-1)*(heads*width)+(heads-1)*width+(padded-1)<2^61 ∧
  heads*width<2^61 ∧ width<2^61

def nativeChecks (a : Host.Arguments) (v : Views) (requestedMax : Nat) : Prop :=
  requestedMax<2^31 ∧ a.maxInput≤requestedMax ∧ a.batch<2^31 ∧ a.q.heads≤65535 ∧
  Bounds.ceilBlocks a.maxInput (Host.block a)≤65535 ∧
  (∀ x∈[v.q,v.o],payloadBudget (a.q.rows+requestedMax+128) a.q.heads a.q.dims (Host.paddedDim a) x) ∧
  (∀ x∈[v.k,v.v],payloadBudget (max a.k.rows 1) a.k.heads a.k.dims (Host.paddedDim a) x) ∧
  (∀ x∈[v.start,v.sequence,v.prompt,v.request],
    VeriTile.Bench.Repairs.ContextAttnFwd.Checked.vectorBudget x a.batch) ∧
  v.pages.storageOffset+(a.requestRows-1)*a.requestColumns+(a.requestColumns+128-1)<2^61 ∧
  a.requestColumns<2^61

instance (v : View3) : Decidable (canonical v) := by unfold canonical; infer_instance
instance (a : Host.Arguments) : Decidable (sourceStrides a) := by unfold sourceStrides; infer_instance
instance (a : Host.Arguments) (v : Views) : Decidable (tensorChecks a v) := by
  unfold tensorChecks all reads; infer_instance
instance (a : Host.Arguments) (r : Regions) (extent : RegionBounds) : Decidable (geometryChecks a r extent) := by
  unfold geometryChecks; infer_instance
instance (a : Host.Arguments) (u : Inputs) : Decidable (metadataChecks a u) := by
  unfold metadataChecks; infer_instance
instance (rows heads width padded : Nat) (v : View) : Decidable (payloadBudget rows heads width padded v) := by
  unfold payloadBudget; infer_instance
instance (a : Host.Arguments) (v : Views) (requestedMax : Nat) : Decidable (nativeChecks a v requestedMax) := by
  unfold nativeChecks; infer_instance

def InputContract (a : Host.Arguments) (r : Regions) (v : Views) (extent : RegionBounds)
    (u : Inputs) (requestedMax : Nat) : Prop :=
  tensorChecks a v ∧ geometryChecks a r extent ∧ metadataChecks a u ∧ nativeChecks a v requestedMax

def checkInputs (a : Host.Arguments) (r : Regions) (v : Views) (extent : RegionBounds)
    (u : Inputs) (requestedMax : Nat) : Bool :=
  decide (tensorChecks a v) && decide (geometryChecks a r extent) &&
    decide (metadataChecks a u) && decide (nativeChecks a v requestedMax)

theorem checkInputs_iff (a : Host.Arguments) (r : Regions) (v : Views) (extent : RegionBounds)
    (u : Inputs) (requestedMax : Nat) :
    checkInputs a r v extent u requestedMax=Bool.true ↔ InputContract a r v extent u requestedMax := by
  simp only [checkInputs,InputContract,Bool.and_eq_true,decide_eq_true_eq,and_assoc]

#axiomsClean checkInputs_iff
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Checked
