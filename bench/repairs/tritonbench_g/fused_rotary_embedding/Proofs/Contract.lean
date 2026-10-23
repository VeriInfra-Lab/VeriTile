import bench.repairs.tritonbench_g.fused_rotary_embedding.Proofs.Host

namespace VeriTile.Bench.Repairs.FusedRotaryEmbedding.Checked
open Classical
noncomputable section

/-- Observed ordinary Tensor descriptor; no payload normalization is performed. -/
structure Tensor where
  object : Nat
  shape : List Nat
  strides : List Nat
  device : Nat
  dtype : String
  exactTensor : Bool
  cuda : Bool
  strided : Bool
  contiguous : Bool
  conjugate : Bool
  negative : Bool
  offset : Nat
  base : Nat
  storageBase : Nat
  storageBytes : Nat
  deriving DecidableEq

def canonical : List Nat → List Nat
  | [] => []
  | _::tail => tail.prod :: canonical tail

def regions : List Region := [.q,.k,.v,.cos,.sin,.kcache,.vcache,.table,.lengths]

structure Call where
  shape : Shape
  tensor : Region → Tensor
  device : Nat
  cosRows : Nat
  sinRows : Nat

def sizes (c : Call) (r : Region) : List Nat :=
  let s := c.shape
  match r with
  | .q => [s.tokens,s.headsQ,s.width]
  | .k | .v => [s.tokens,s.headsK,s.width]
  | .cos => [c.cosRows,s.cosWidth]
  | .sin => [c.sinRows,s.sinWidth]
  | .kcache => if s.newLayout then [s.blocks,s.headsK,s.width/s.split,s.blockSize,s.split]
    else [s.blocks,s.headsK,s.blockSize,s.width]
  | .vcache => [s.blocks,s.headsK,s.blockSize,s.width]
  | .table => [s.tokens,s.tableWidth]
  | .lengths => [s.tokens]
  | .other _ => []

def extent (c : Call) (r : Region) : Nat :=
  if r∈regions then (c.tensor r).shape.prod else 0

def sourceExtent (s : Shape) : Region → Nat
  | .q => s.tokens*s.headsQ*s.width
  | .k | .v => s.tokens*s.headsK*s.width
  | .cos => s.tokens*s.cosWidth
  | .sin => s.tokens*s.sinWidth
  | .kcache | .vcache => s.blocks*s.headsK*s.blockSize*s.width
  | .table => s.tokens*s.tableWidth
  | .lengths => s.tokens
  | .other _ => 0

/-- Upper envelopes include every padded lane before its load/store mask. -/
def formed (s : Shape) : Region → Nat
  | .q => sourceExtent s .q+s.width+s.halfBlock+s.fullBlock
  | .k | .v => sourceExtent s .k+s.width+s.halfBlock+s.fullBlock
  | .cos => sourceExtent s .cos+s.halfBlock
  | .sin => sourceExtent s .sin+s.halfBlock
  | .kcache => sourceExtent s .kcache+(s.width+s.halfBlock+s.fullBlock)*(s.blockSize+1)
  | .vcache => sourceExtent s .vcache+s.fullBlock
  | .table => sourceExtent s .table
  | .lengths => sourceExtent s .lengths
  | .other _ => 0

def ordinary (c : Call) (r : Region) : Prop :=
  let t := c.tensor r
  t.exactTensor=true ∧ t.cuda=true ∧ t.strided=true ∧ t.contiguous=true ∧
  t.conjugate=false ∧ t.negative=false ∧ t.offset=0 ∧ t.device=c.device ∧
  t.dtype=(if r=.table ∨ r=.lengths then "int32" else "float32") ∧
  t.shape=sizes c r ∧ t.shape.prod≤(2^63-1)/4 ∧
  (t.shape.prod≠0 → t.strides=canonical t.shape)

def separate (a b : Tensor) : Prop :=
  a.base+a.storageBytes≤b.base ∨ b.base+b.storageBytes≤a.base

def livePointer (c : Call) (r : Region) : Prop :=
  let t := c.tensor r
  t.base%4=0 ∧ t.base=t.storageBase ∧
  formed c.shape r≤(2^63-1)/4 ∧ t.base+4*formed c.shape r≤2^64 ∧
  4*t.shape.prod≤t.storageBytes ∧ t.base+t.storageBytes≤2^64

def shapeCheck (c : Call) : Bool := decide (c.shape.Valid ∧ c.shape.tokens≤c.cosRows ∧
  c.shape.tokens≤c.sinRows ∧ (c.shape.newLayout=false → c.shape.split=c.shape.width))
def tensorsCheck (c : Call) : Bool := regions.all fun r => decide (ordinary c r)
def pointerCheck (c : Call) : Bool := decide (c.shape.tokens=0) ||
  (decide (c.shape.headsQ<2^31 ∧ c.shape.tokens≤65535) &&
   regions.all (fun r => decide (livePointer c r)) &&
   regions.all (fun r => regions.all (fun q => decide (r=q ∨ separate (c.tensor r) (c.tensor q)))))

def metadataCheck (s : Shape) (m : Physical.Bytes) : Bool :=
  let c := Physical.configuration s m
  decide ((∀ t : Fin s.tokens,0<c.metadata.lengths t ∧ c.metadata.lengths t<2^31 ∧
    c.past t/s.blockSize<s.tableWidth ∧ c.block t<s.blocks ∧ c.block t<2^31) ∧
    (∀ t u : Fin s.tokens,c.block t=c.block u → c.slot t=c.slot u → t=u))

def check (c : Call) (m : Physical.Bytes) : Bool :=
  shapeCheck c && tensorsCheck c && pointerCheck c && metadataCheck c.shape m

def host (c : Call) (m : Physical.Bytes) : Host.Call :=
  ⟨Physical.configuration c.shape m,fun r => (c.tensor r).object,c.device⟩

theorem checked_parts (c : Call) (m : Physical.Bytes) (h : check c m=true) :
    (shapeCheck c=true ∧ tensorsCheck c=true) ∧ pointerCheck c=true ∧ metadataCheck c.shape m=true := by
  simpa only [check,Bool.and_eq_true,and_assoc] using h

theorem shape_valid (c : Call) (m : Physical.Bytes) (h : check c m=true) : c.shape.Valid := by
  have h := (checked_parts c m h).1.1
  simp only [shapeCheck,decide_eq_true_eq] at h
  exact h.1

theorem ordinary_checked (c : Call) (m : Physical.Bytes) (h : check c m=true)
    (r : Region) (hr : r∈regions) : ordinary c r := by
  have h := (checked_parts c m h).1.2
  simpa only [decide_eq_true_eq] using (List.all_eq_true.mp h) r hr

theorem live_checked (c : Call) (m : Physical.Bytes) (h : check c m=true)
    (live : 0<c.shape.tokens) :
    (c.shape.headsQ<2^31 ∧ c.shape.tokens≤65535) ∧
    (∀ r,r∈regions → livePointer c r) ∧
    (∀ r,r∈regions → ∀ q,q∈regions → r≠q → separate (c.tensor r) (c.tensor q)) := by
  have h := (checked_parts c m h).2.1
  simp only [pointerCheck,show ¬c.shape.tokens=0 by omega,decide_false,Bool.false_or,
    Bool.and_eq_true,decide_eq_true_eq] at h
  refine ⟨h.1.1,?_,?_⟩
  · intro r hr
    simpa using (List.all_eq_true.mp h.1.2) r hr
  · intro r hr q hq ne
    have hd := (List.all_eq_true.mp ((List.all_eq_true.mp h.2) r hr)) q hq
    simpa [ne] using hd

theorem config_valid (c : Call) (m : Physical.Bytes) (h : check c m=true) : (host c m).config.Valid := by
  have hm := (checked_parts c m h).2.2
  simp only [metadataCheck,decide_eq_true_eq] at hm
  refine ⟨shape_valid c m h,?_,?_,?_,?_,?_,?_⟩
  · intro t ht; exact (hm.1 ⟨t,ht⟩).1
  · intro t ht; exact (hm.1 ⟨t,ht⟩).2.1
  · intro t ht; exact (hm.1 ⟨t,ht⟩).2.2.1
  · intro t ht; exact (hm.1 ⟨t,ht⟩).2.2.2.1
  · intro t ht; exact (hm.1 ⟨t,ht⟩).2.2.2.2
  · intro t ht u hu hb hs
    exact congrArg Fin.val (hm.2 ⟨t,ht⟩ ⟨u,hu⟩ hb hs)

theorem source_extent_le (c : Call) (m : Physical.Bytes) (h : check c m=true)
    (r : Region) : sourceExtent c.shape r≤extent c r := by
  have hs := (checked_parts c m h).1.1
  simp only [shapeCheck,decide_eq_true_eq] at hs
  have eq (r : Region) (hr : r∈regions) := (ordinary_checked c m h r hr).2.2.2.2.2.2.2.2.2.1
  have hd : c.shape.width/c.shape.split*c.shape.split=c.shape.width :=
    Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hs.1.divides)
  cases r <;> simp [sourceExtent,extent,regions,eq,sizes,List.prod_cons,List.prod_nil,Nat.mul_assoc]
  · exact Nat.mul_le_mul_right _ hs.2.1
  · exact Nat.mul_le_mul_right _ hs.2.2.1
  · split
    · simp only [List.prod_cons,List.prod_nil,Nat.mul_one]
      nlinarith only [congrArg (fun v => c.shape.blocks*c.shape.headsK*c.shape.blockSize*v) hd]
    · simp [List.prod_cons,List.prod_nil,Nat.mul_assoc]

theorem host_admitted (c : Call) (m : Physical.Bytes) (h : check c m=true) : Host.Admitted (host c m) := by
  have bound (r : Region) (hr : r∈regions) : sourceExtent c.shape r≤(2^63-1)/4 := by
    have size := (ordinary_checked c m h r hr).2.2.2.2.2.2.2.2.2.2.1
    have le := source_extent_le c m h r
    exact le.trans (by simpa [extent,hr] using size)
  have hs := (checked_parts c m h).1.1
  simp only [shapeCheck,decide_eq_true_eq] at hs
  refine ⟨config_valid c m h,hs.2.2.2,?_,?_,?_,?_,?_,?_⟩
  · intro kind; cases kind
    · simpa [host,Physical.configuration,sourceExtent,Shape.heads] using bound .q (by simp [regions])
    · simpa [host,Physical.configuration,sourceExtent,Shape.heads] using bound .k (by simp [regions])
    · simpa [host,Physical.configuration,sourceExtent,Shape.heads] using bound .v (by simp [regions])
  · exact bound .kcache (by simp [regions])
  · exact bound .cos (by simp [regions])
  · exact bound .sin (by simp [regions])
  · exact bound .table (by simp [regions])
  · exact bound .lengths (by simp [regions])

#axiomsClean config_valid
#axiomsClean host_admitted
#specNonCircular check avoiding [Host.selected,Host.mathematical,actualSource]
end
end VeriTile.Bench.Repairs.FusedRotaryEmbedding.Checked
