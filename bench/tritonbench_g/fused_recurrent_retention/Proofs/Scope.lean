import bench.tritonbench_g.fused_recurrent_retention.Proofs.SourceBounds
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section

structure ForwardScope (s : Shape) (args : ForwardArguments) (current : Nat) : Prop where
  shape : s.Admitted
  qRaw : args.q.RawFP32 current
  kRaw : args.k.RawFP32 current
  vRaw : args.v.RawFP32 current
  qShape : args.q.dimensions=![s.B,s.H,s.T,s.DK]
  kShape : args.k.dimensions=![s.B,s.H,s.T,s.DK]
  vShape : args.v.dimensions=![s.B,s.H,s.T,s.DV]
  qDense : args.q.view.strides=contiguous4 s.B s.H s.T s.DK
  kDense : args.k.view.strides=contiguous4 s.B s.H s.T s.DK
  vDense : args.v.view.strides=contiguous4 s.B s.H s.T s.DV
  seedMeta : ∀ x,args.seed=some x → x.RawFP32 current ∧ x.dimensions=![s.B,s.H,s.DK,s.DV] ∧
    x.view.strides=contiguous4 s.B s.H s.DK s.DV
  bounds : SourceAddress.Bounds .forward args.seed.isSome args.outputFinal s args.buffers
  qRead : ∀ h<s.BH,∀ t<s.T,∀ k<s.DK,
    0≤(args.buffers.qAddress s h t k).2 ∧ (args.buffers.qAddress s h t k).2<(args.q.view.capacity : Int)
  kRead : ∀ h<s.BH,∀ t<s.T,∀ k<s.DK,
    0≤(args.buffers.kAddress s h t k).2 ∧ (args.buffers.kAddress s h t k).2<(args.k.view.capacity : Int)
  vRead : ∀ h<s.BH,∀ t<s.T,∀ v<s.DV,
    0≤(args.buffers.vAddress s h t v).2 ∧ (args.buffers.vAddress s h t v).2<(args.v.view.capacity : Int)
  seedRead : args.seed.isSome=Bool.true → ∀ h<s.BH,∀ k<s.DK,∀ v<s.DV,
    0≤(args.buffers.seedAddress s h k v).2 ∧
      (args.buffers.seedAddress s h k v).2<(args.buffers.seed.capacity : Int)

structure BackwardScope (s : Shape) (ctx : SavedContext) (do_ : TensorObject)
    (final : Option TensorObject) (current : Nat) : Prop where
  doRaw : do_.RawFP32 current
  doShape : do_.dimensions=![s.B,s.H,s.T,s.DV]
  finalMeta : ∀ f,final=some f → ctx.outputFinal=Bool.true ∧ f.RawFP32 current ∧
    f.dimensions=![s.B,s.H,s.DK,s.DV]
  bounds : SourceAddress.Bounds .backward ctx.initial.isSome final.isSome s
    (ctx.buffers do_.view (final.map TensorObject.view))
  doRead : ∀ h<s.BH,∀ t<s.T,∀ v<s.DV,do_.view.InBounds (h/s.H) (h%s.H) t v
  finalRead : ∀ f,final=some f → ∀ h<s.BH,∀ k<s.DK,∀ v<s.DV,
    f.view.InBounds (h/s.H) (h%s.H) k v

/-- A forward call carries no future cotangent. A backward call is quantified
only after its returned context exists; saved tensor version checking is a
separate, ordinary autograd/storage condition. -/
structure ForwardCall (α : Type) (s : Shape) (caller : Set Nat) where
  args : ForwardArguments
  currentDevice : Nat
  scope : ForwardScope s args currentDevice
  allocation : Allocation 3 caller
  inputs : ∀ id,id∈args.buffers.storageIds → id∈caller
  before : Memory α

structure BackwardCall (α : Type) (s : Shape) (ctx : SavedContext) (forwardSnapshot : Memory α)
    (current : Nat) (caller : Set Nat) where
  do_ : TensorObject
  final : Option TensorObject
  scope : BackwardScope s ctx do_ final current
  allocation : Allocation 6 caller
  inputs : ∀ id,id∈(ctx.buffers do_.view (final.map TensorObject.view)).storageIds → id∈caller
  before : Memory α
  saved : ctx.Unchanged forwardSnapshot before

#axiomsClean ForwardScope.bounds
#axiomsClean BackwardScope.bounds
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
