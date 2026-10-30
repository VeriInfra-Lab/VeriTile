import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.Model
import bench.repairs.tritonbench_g.rotary_emb.Contract

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Checked
abbrev Tensor := VeriTile.Bench.Repairs.RotaryEmb.Checked.Tensor

structure Call where
  request : Request
  currentDevice : Nat
  /-- Q, K, cosine, sine, cache, block tables, lengths in source argument order. -/
  tensor : Fin 7 → Tensor

def role (key : Bool) : Fin 7 := if key then 1 else 0

def expectedShape (r : Request) (i : Fin 7) : List Nat :=
  if i=0 then [r.tokens,r.qHeads,r.width]
  else if i=1 then [r.tokens,r.kHeads,r.width]
  else if i=2 then [r.cosRows,r.cosWidth]
  else if i=3 then [r.sinRows,r.sinWidth]
  else if i=4 then [r.blocks,r.kHeads,r.blockSize,r.width]
  else if i=5 then [r.tokens,r.tableColumns]
  else [r.tokens]

def canonical : List Nat → List Nat
  | [] => []
  | _::tail => (tail.map (max 1)).prod :: canonical tail

def base (r : Request) (i : Fin 7) : Nat :=
  if i=0 then r.qBase else if i=1 then r.kBase else if i=2 then r.cosBase
  else if i=3 then r.sinBase else if i=4 then r.cacheBase else if i=5 then r.tableBase else r.lengthBase

def present (r : Request) (i : Fin 7) : Bool := decide (i.val<4) || r.cached

def separate (a b : Tensor) : Prop :=
  a.shape.prod=0 ∨ b.shape.prod=0 ∨
  a.allocationBase+a.allocationBytes≤b.allocationBase ∨ b.allocationBase+b.allocationBytes≤a.allocationBase
instance (a b : Tensor) : Decidable (separate a b) := inferInstanceAs (Decidable (_∨_∨_∨_))

def shapeCheck (r : Request) : Bool :=
  decide (0<r.qHeads ∧ 0<r.kHeads ∧ r.qHeads%r.kHeads=0 ∧
    r.tokens≤r.cosRows ∧ r.tokens≤r.sinRows ∧ r.half≤r.cosWidth ∧ r.half≤r.sinWidth ∧
    (r.cached=true → 0<r.blockSize))

def tensorCheck (c : Call) : Bool :=
  decide (∀ i : Fin 7,present c.request i=true →
    (c.tensor i).shape=expectedShape c.request i ∧ (c.tensor i).strided=true ∧
    (c.tensor i).cuda=true ∧ (c.tensor i).device=c.request.device ∧
    (c.tensor i).contiguous=true ∧ (c.tensor i).storageOffset=0 ∧
    (c.tensor i).negative=false ∧ (c.tensor i).conjugate=false ∧
    (c.tensor i).dtype=(if i.val<5 then "float32" else "int64") ∧
    (c.tensor i).elementBytes=(if i.val<5 then 4 else 8) ∧
    ((c.tensor i).shape.prod=0 ∨
      ((c.tensor i).base=(c.tensor i).allocationBase ∧ (c.tensor i).base=4*base c.request i ∧
       (c.tensor i).base%(c.tensor i).elementBytes=0 ∧
       (c.tensor i).elementBytes*(c.tensor i).shape.prod≤(c.tensor i).allocationBytes ∧
       (c.tensor i).base+(c.tensor i).allocationBytes≤2^64))) &&
  decide (∀ i j : Fin 7,present c.request i=true → present c.request j=true → i≠j →
    separate (c.tensor i) (c.tensor j))

def empty (r : Request) : Bool := decide (r.tokens=0 ∨ r.half=0)

def metadataCheck (r : Request) (indices : Indices) : Bool :=
  if r.cached then
    decide (∀ t : Fin r.tokens,
      0 < indices (r.lengthBase+2*t.val) ∧ indices (r.lengthBase+2*t.val)<2^63 ∧
      r.past indices t.val/r.blockSize<r.tableColumns ∧
      0 ≤ indices (r.tableBase+2*(t.val*r.tableColumns+r.past indices t.val/r.blockSize)) ∧
      r.block indices t.val<r.blocks) &&
    decide (∀ a b : Fin r.tokens,a≠b →
      (r.block indices a.val,r.slot indices a.val)≠(r.block indices b.val,r.slot indices b.val))
  else true

def paddedTokens (r : Request) := ((r.tokens+3)/4)*4

def lastOffset (r : Request) (i : Fin 7) : Nat :=
  if i=0 then (paddedTokens r-1)*r.qHeads*r.width+(r.qHeads-1)*r.width+r.half+r.blockPairs-1
  else if i=1 then (paddedTokens r-1)*r.kHeads*r.width+(r.qHeads-1)*r.width+r.half+r.blockPairs-1
  else if i=2 then (paddedTokens r-1)*r.cosWidth+r.blockPairs-1
  else if i=3 then (paddedTokens r-1)*r.sinWidth+r.blockPairs-1
  else (r.blocks-1)*r.kHeads*r.blockSize*r.width+(r.qHeads-1)*r.blockSize*r.width+
    (r.blockSize-1)*r.width+r.half+r.blockPairs-1

def launchCheck (c : Call) (indices : Indices) : Bool :=
  metadataCheck c.request indices && decide (c.currentDevice=c.request.device) &&
  decide (∀ i : Fin 7,present c.request i=true →
    (c.tensor i).strides=canonical (c.tensor i).shape ∧ (c.tensor i).shape.prod<2^60) &&
  decide (∀ i : Fin 5,present c.request ⟨i.val,by omega⟩=true →
    4*lastOffset c.request ⟨i.val,by omega⟩+4<2^63 ∧
    4*base c.request ⟨i.val,by omega⟩+4*lastOffset c.request ⟨i.val,by omega⟩+4≤2^64)

def check (c : Call) (indices : Indices) : Bool :=
  shapeCheck c.request && tensorCheck c && (empty c.request || launchCheck c indices)

end VeriTile.Bench.Repairs.RotaryEmbNopad.Checked
