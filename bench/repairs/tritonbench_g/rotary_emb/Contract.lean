import bench.repairs.tritonbench_g.rotary_emb.Proofs.Host

namespace VeriTile.Bench.Repairs.RotaryEmb.Checked

/-- Direct observations; argument classification precedes these queries. -/
structure Tensor where
  object : Nat
  device : Nat
  shape : List Nat
  strides : List Nat
  base : Nat
  allocationBase : Nat
  allocationBytes : Nat
  storageOffset : Nat
  dtype : String
  elementBytes : Nat
  strided : Bool
  cuda : Bool
  contiguous : Bool
  negative : Bool
  conjugate : Bool

structure Call where
  shape : Shape
  tensor : Fin 4 → Tensor
  /-- Classification, finite range and conversion are the literal Python
  int(width*factor), not an exact-real multiplication substituted for it. -/
  factorAccepted : Bool

def role (key : Bool) : Fin 4 := if key then 1 else 0
def coefficientRole (sine : Bool) : Fin 4 := if sine then 3 else 2

def Call.metadata (c : Call) : Host.Metadata :=
  ⟨c.shape,(fun r => match r with
    | .q => (c.tensor 0).object
    | .k => (c.tensor 1).object
    | .cos => (c.tensor 2).object
    | .sin => (c.tensor 3).object
    | .other n => n),(c.tensor 0).device⟩

def expectedShape (s : Shape) (i : Fin 4) : List Nat :=
  if i=0 then [s.tokens,s.headsQ,s.width]
  else if i=1 then [s.tokens,s.headsK,s.width]
  else [s.tokens,if i=2 then s.cosWidth else s.sinWidth]

def expectedStrides (s : Shape) (i : Fin 4) : List Nat :=
  if i=0 then [s.headsQ*s.width,s.width,1]
  else if i=1 then [s.headsK*s.width,s.width,1]
  else [if i=2 then s.cosWidth else s.sinWidth,1]

def extent (s : Shape) (i : Fin 4) := (expectedShape s i).prod

def lastOffset (s : Shape) (i : Fin 4) : Nat :=
  if i=0 then (16*s.seqBlocks-1)*s.headsQ*s.width+(4*s.headBlocks-1)*s.width+(2*s.pairBlock-1)
  else if i=1 then (16*s.seqBlocks-1)*s.headsK*s.width+(4*s.headBlocks-1)*s.width+(2*s.pairBlock-1)
  else (16*s.seqBlocks-1)*(if i=2 then s.cosWidth else s.sinWidth)+(2*s.pairBlock-1)

def separate (a b : Tensor) : Prop :=
  a.allocationBase+a.allocationBytes≤b.allocationBase ∨
  b.allocationBase+b.allocationBytes≤a.allocationBase

instance (a b : Tensor) : Decidable (separate a b) := inferInstanceAs (Decidable (_ ∨ _))

def shapeCheck (s : Shape) : Bool :=
  decide (0<s.headsQ ∧ 0<s.headsK ∧ 0<s.width ∧ s.rotary≤s.width ∧
    s.rotary%2=0 ∧ s.rotary≤s.cosWidth ∧ s.rotary≤s.sinWidth)

def metadataCheck (c : Call) : Bool :=
  c.factorAccepted && decide (∀ i : Fin 4,
    (c.tensor i).strided=true ∧ (c.tensor i).cuda=true ∧
    (c.tensor i).contiguous=true ∧ (c.tensor i).negative=false ∧
    (c.tensor i).conjugate=false ∧ (c.tensor i).dtype="float32" ∧
    (c.tensor i).elementBytes=4 ∧ (c.tensor i).storageOffset=0 ∧
    (c.tensor i).device=(c.tensor 0).device ∧
    (c.tensor i).shape=expectedShape c.shape i ∧ extent c.shape i≤(2^63-1)/4)

def launchCheck (c : Call) : Bool :=
  decide (∀ i : Fin 4,
    (c.tensor i).strides=expectedStrides c.shape i ∧
    (c.tensor i).base=(c.tensor i).allocationBase ∧ (c.tensor i).base%4=0 ∧
    4*extent c.shape i≤(c.tensor i).allocationBytes ∧
    (c.tensor i).base+(c.tensor i).allocationBytes≤2^64 ∧
    4*lastOffset c.shape i+4<2^63 ∧ (c.tensor i).base+4*lastOffset c.shape i+4≤2^64) &&
  decide (∀ i j : Fin 4,i≠j → separate (c.tensor i) (c.tensor j)) &&
  decide (∀ v∈[c.shape.tokens,c.shape.headsQ,c.shape.headsK,c.shape.width,
    c.shape.cosWidth,c.shape.sinWidth,16*c.shape.seqBlocks,4*c.shape.headBlocks,
    2*c.shape.pairBlock,c.shape.headsQ*c.shape.width,c.shape.headsK*c.shape.width],v<2^63)

def check (c : Call) : Bool :=
  shapeCheck c.shape && metadataCheck c && (Host.empty c.shape || launchCheck c)

end VeriTile.Bench.Repairs.RotaryEmb.Checked
