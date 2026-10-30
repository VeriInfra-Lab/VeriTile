import bench.repairs.tritonbench_g.rope_embedding.Proofs.Native
namespace VeriTile.Bench.Repairs.RopeEmbedding.Checked
open VeriTile.Triton FP.Structural

def payload (t : Native.Tensor) :=
  t.strided && t.cuda && t.contiguous && t.materialized &&
  decide (t.dtype="float32") && decide (t.elementBytes=4)

def fits (t : Native.Tensor) : Bool :=
  decide (t.shape.prod=0 ∨ (t.allocationBase≤t.base ∧ t.base+4*t.shape.prod≤t.allocationEnd ∧
    t.allocationEnd≤2^64 ∧ t.base%4=0))

def separate (a b : Native.Tensor) : Bool :=
  decide (a.allocationBase=a.allocationEnd ∨ b.allocationBase=b.allocationEnd ∨
    a.allocationEnd≤b.allocationBase ∨ b.allocationEnd≤a.allocationBase)

def outputLast (r : Request) :=
  ((r.rows-1)*r.heads+(r.heads-1))*r.dimension+(r.block-1)+r.half
def tableLast (r : Request) := (r.sequence-1)*r.half+(r.block-1)

def metadataCheck {α β Store Byte : Type} [Inhabited α] (api : Memory.API β Store Byte)
    (embed : β→α) (r : Request) (s : State α) (call : Native.Preparation api embed r s)
    (clone squeeze : Bool) : Bool :=
  decide (call.cloneInput=clone ∧ call.squeezed=squeeze) &&
  [call.input,call.cosineInput,call.sineInput].all payload &&
  [call.input,call.cosineInput,call.sineInput].all fits &&
  [call.cosineInput,call.sineInput].all (fun t => decide (t.device=call.input.device)) &&
  decide (Native.tableShape r.half call.cosineRows call.cosine.shape ∧
    Native.tableShape r.half call.sineRows call.sine.shape) &&
  decide (r.sequence≤call.cosineRows ∧ r.sequence≤call.sineRows) &&
  decide (r.half≠0 → call.cosine.strides[0]! = r.half ∧ call.sine.strides[0]! = r.half) &&
  (clone || (separate call.input call.cosineInput && separate call.input call.sineInput))

def launchCheck {α β Store Byte : Type} [Inhabited α] (api : Memory.API β Store Byte)
    (embed : β→α) (r : Request) (s : State α) (call : Native.Preparation api embed r s) : Bool :=
  decide (r.block≤65536 ∧ r.batches<2^31 ∧ r.heads+3<2^31 ∧ r.sequence<2^31 ∧ r.dimension<2^31 ∧
    r.rows<2^31 ∧ r.heads*r.dimension<2^31 ∧ r.groups<2^31) &&
  decide (r.size≠0 ∧ r.half≠0 → outputLast r<2^31 ∧ tableLast r<2^31) &&
  decide (r.size≠0 ∧ r.half≠0 → call.output.base+4*outputLast r+4≤2^64 ∧
    call.cosine.base+4*tableLast r+4≤2^64 ∧ call.sine.base+4*tableLast r+4≤2^64)

def check {α β Store Byte : Type} [Inhabited α] (api : Memory.API β Store Byte)
    (embed : β→α) (r : Request) (s : State α) (call : Native.Preparation api embed r s)
    (clone squeeze : Bool) : Bool :=
  metadataCheck api embed r s call clone squeeze && launchCheck api embed r s call

def cacheCheck (r : Request) (groups block warps : Nat) : Bool :=
  decide ((groups,block,warps)=(r.groups,r.block,r.warps))

end VeriTile.Bench.Repairs.RopeEmbedding.Checked
