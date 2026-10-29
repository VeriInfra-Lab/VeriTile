import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.Native

namespace VeriTile.Bench.Repairs.RopeBackwardTransform.Checked
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding.Memory (API)

def launched (r : Request) : Prop := r.rows≠0 ∧ r.half≠0 ∧ ¬(r.qHeads=0 ∧ r.kHeads=0)
instance (r : Request) : Decidable (launched r) := inferInstanceAs (Decidable (_∧_∧_))

def outputLast (r : Request) (k : Bool) :=
  (r.rows-1)*(r.heads k*r.dimension)+(r.paddedHeads k-1)*r.dimension+(r.paddedDim/2-1)+r.half
def tableLast (r : Request) (rowStride colStride : Nat) :=
  (r.sequence-1)*rowStride+(r.paddedDim/2-1)*colStride

def payload (t : Native.Tensor) : Bool :=
  t.strided && t.cuda && t.materialized && decide (t.dtype="float32") && decide (t.elementBytes=4)

def separate (a b : Native.Tensor) : Bool := decide (a.storage≠b.storage) &&
  decide (a.allocationBase=a.allocationEnd ∨ b.allocationBase=b.allocationEnd ∨
    a.allocationEnd≤b.allocationBase ∨ b.allocationEnd≤a.allocationBase)

def fits (t : Native.Tensor) (count span : Nat) : Bool :=
  decide (count=0 ∨ (t.allocationBase≤t.base ∧ t.base+4*span≤t.allocationEnd ∧
    t.allocationEnd≤2^64 ∧ t.base%4=0))

def metadataCheck {α β Store Byte : Type} [Inhabited α] (api : API β Store Byte)
    (embed : β→α) (r : Request) (m : Source.Memory α) (call : Native.Preparation api embed r m) : Bool :=
  [call.input Bool.false,call.input Bool.true,call.cosine,call.sine].all payload &&
  [call.input Bool.false,call.input Bool.true,call.cosine,call.sine].all
    (fun t => decide (t.device=(call.input Bool.false).device)) &&
  (call.input Bool.false).contiguous && (call.input Bool.true).contiguous &&
  fits (call.input Bool.false) (r.size Bool.false) (r.size Bool.false) &&
  fits (call.input Bool.true) (r.size Bool.true) (r.size Bool.true) &&
  fits call.cosine (r.cosRows*r.half) (Native.span r.cosRows r.half r.cosRowStride r.cosColStride) &&
  fits call.sine (r.sinRows*r.half) (Native.span r.sinRows r.half r.sinRowStride r.sinColStride) &&
  decide (r.sequence≤r.cosRows ∧ r.sequence≤r.sinRows) &&
  separate (call.input Bool.false) (call.input Bool.true) &&
  separate (call.input Bool.false) call.cosine && separate (call.input Bool.false) call.sine &&
  separate (call.input Bool.true) call.cosine && separate (call.input Bool.true) call.sine

/-- Maximum padded offsets are checked with ordinary finite arithmetic. No
quantified index domain is supplied by the caller. Empty/no-pair launches skip
pointer checks exactly as the executable wrapper does. -/
def launchCheck {α β Store Byte : Type} [Inhabited α] (api : API β Store Byte)
    (embed : β→α) (r : Request) (m : Source.Memory α) (call : Native.Preparation api embed r m) : Bool :=
  decide (r.batches<2^31 ∧ r.rows<2^31 ∧ r.sequence<2^31 ∧ r.dimension<2^31 ∧ r.paddedDim≤65536 ∧
    r.cosRowStride<2^31 ∧ r.cosColStride<2^31 ∧ r.sinRowStride<2^31 ∧ r.sinColStride<2^31) &&
  decide (∀ k : Bool,r.heads k<2^31 ∧ r.heads k*r.dimension<2^31 ∧ r.paddedHeads k*r.paddedDim≤65536) &&
  decide (launched r → (∀ k : Bool,outputLast r k<2^31) ∧
    tableLast r r.cosRowStride r.cosColStride<2^31 ∧ tableLast r r.sinRowStride r.sinColStride<2^31) &&
  decide (launched r → (∀ k : Bool,(call.output k).base+4*outputLast r k+4≤2^64) ∧
    call.cosine.base+4*tableLast r r.cosRowStride r.cosColStride+4≤2^64 ∧
    call.sine.base+4*tableLast r r.sinRowStride r.sinColStride+4≤2^64)

def check {α β Store Byte : Type} [Inhabited α] (api : API β Store Byte)
    (embed : β→α) (r : Request) (m : Source.Memory α) (call : Native.Preparation api embed r m) : Bool :=
  metadataCheck api embed r m call && launchCheck api embed r m call

end VeriTile.Bench.Repairs.RopeBackwardTransform.Checked
