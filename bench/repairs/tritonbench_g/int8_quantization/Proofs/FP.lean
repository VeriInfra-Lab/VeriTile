import bench.repairs.tritonbench_g.int8_quantization.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.Int8Quantization.Floating
noncomputable section
set_option maxHeartbeats 3000000

/-- All values are FP32 words in the intended instantiation; Q's factor is
the actual host constant rounded to FP32, not the ideal Real constant. -/
structure Constants (α : Type) where
  zero : α
  one : α
  minusOne : α
  half : α
  positive127 : α
  negative127 : α
  largest : α
  factor : α

/-- Exact FP32 primitives. `maximum` is one maximum reduction on the actual
padded lane order. Integer bad-flag reduction is ordinary Boolean any. -/
structure Numerics (α : Type) where
  abs : α → α
  mul : α → α → α
  div : α → α → α
  add : α → α → α
  max : α → α → α
  min : α → α → α
  le : α → α → Bool
  lt : α → α → Bool
  maximum : {n : Nat} → (Fin n → α) → α
  int8 : α → Int

variable {α : Type} {n : Nat}
def scaled (c : Constants α) (o : Numerics α) (query : Bool) (x : Fin n → α) : Fin n → α :=
  fun i=> if query then o.mul (x i) c.factor else x i
def finite (c : Constants α) (o : Numerics α) (x : Fin n → α) : Fin n → Bool :=
  fun i=> o.le (o.abs (x i)) c.largest
def bad (flags : Fin n → Bool) : Bool := (List.finRange n).any (fun i=>!flags i)
def safe (c : Constants α) (x : Fin n → α) (flags : Fin n → Bool) : Fin n → α :=
  fun i=> if flags i then x i else c.zero
def scale (c : Constants α) (o : Numerics α) (x : Fin n → α) : α :=
  o.div (o.maximum (fun i=> o.abs (x i))) c.positive127
def usable (c : Constants α) (o : Numerics α) (flags : Fin n → Bool) (s : α) : Bool :=
  (!bad flags) && o.lt c.zero s
def normalized (c : Constants α) (o : Numerics α) (x : α) (s : α) (use : Bool) : α :=
  let denominator:=if use then s else c.one
  let quotient:=o.div x denominator
  if use then quotient else c.zero
def clipped (c : Constants α) (o : Numerics α) (x : α) : α :=
  o.min (o.max x c.negative127) c.positive127
def bias (c : Constants α) (o : Numerics α) (x : α) : α :=
  o.mul c.half (if o.le c.zero x then c.one else c.minusOne)
def bytes (c : Constants α) (o : Numerics α) (x : α) : Int :=
  o.int8 (o.add x (bias c o x))

def run (c : Constants α) (o : Numerics α) (query : Bool) (x : Fin n → α) : (Fin n → Int) × α :=
  let a:=scaled c o query x
  let flags:=finite c o a
  let b:=safe c a flags
  let s:=scale c o b
  let use:=usable c o flags s
  (fun i=> bytes c o (clipped c o (normalized c o (b i) s use)),
   if bad flags then c.zero else s)

/-- Every equality is a single primitive call at a reached reference operand.
No field mentions run, source execution, final output, memory, grid or host. -/
structure Local (c : Constants α) (native reference : Numerics α)
    (query : Bool) (x : Fin n → α) : Prop where
  prescale : query=Bool.true → ∀ i,native.mul (x i) c.factor=reference.mul (x i) c.factor
  classify_abs : ∀ i,native.abs (scaled c reference query x i)=reference.abs (scaled c reference query x i)
  classify_le : ∀ i,native.le (reference.abs (scaled c reference query x i)) c.largest=
    reference.le (reference.abs (scaled c reference query x i)) c.largest
  max_abs : ∀ i,native.abs (safe c (scaled c reference query x) (finite c reference (scaled c reference query x)) i)=
    reference.abs (safe c (scaled c reference query x) (finite c reference (scaled c reference query x)) i)
  maximum : native.maximum (fun i=> reference.abs (safe c (scaled c reference query x)
    (finite c reference (scaled c reference query x)) i))=
    reference.maximum (fun i=> reference.abs (safe c (scaled c reference query x)
    (finite c reference (scaled c reference query x)) i))
  scale_div : native.div (reference.maximum (fun i=> reference.abs (safe c (scaled c reference query x)
    (finite c reference (scaled c reference query x)) i))) c.positive127=
    scale c reference (safe c (scaled c reference query x) (finite c reference (scaled c reference query x)))
  positive : native.lt c.zero (scale c reference (safe c (scaled c reference query x) (finite c reference (scaled c reference query x))))=
    reference.lt c.zero (scale c reference (safe c (scaled c reference query x) (finite c reference (scaled c reference query x))))
  normalize : ∀ i,
    let a:=scaled c reference query x
    let f:=finite c reference a
    let b:=safe c a f
    let s:=scale c reference b
    let use:=usable c reference f s
    native.div (b i) (if use then s else c.one)=reference.div (b i) (if use then s else c.one)
  lower : ∀ i,
    let a:=scaled c reference query x
    let f:=finite c reference a
    let b:=safe c a f
    let s:=scale c reference b
    let z:=normalized c reference (b i) s (usable c reference f s)
    native.max z c.negative127=reference.max z c.negative127
  upper : ∀ i,
    let a:=scaled c reference query x
    let f:=finite c reference a
    let b:=safe c a f
    let s:=scale c reference b
    let z:=normalized c reference (b i) s (usable c reference f s)
    native.min (reference.max z c.negative127) c.positive127=reference.min (reference.max z c.negative127) c.positive127
  sign : ∀ i,
    let a:=scaled c reference query x
    let f:=finite c reference a
    let b:=safe c a f
    let s:=scale c reference b
    let z:=clipped c reference (normalized c reference (b i) s (usable c reference f s))
    native.le c.zero z=reference.le c.zero z
  half_mul : ∀ i,
    let a:=scaled c reference query x
    let f:=finite c reference a
    let b:=safe c a f
    let s:=scale c reference b
    let z:=clipped c reference (normalized c reference (b i) s (usable c reference f s))
    native.mul c.half (if reference.le c.zero z then c.one else c.minusOne)=bias c reference z
  add_bias : ∀ i,
    let a:=scaled c reference query x
    let f:=finite c reference a
    let b:=safe c a f
    let s:=scale c reference b
    let z:=clipped c reference (normalized c reference (b i) s (usable c reference f s))
    native.add z (bias c reference z)=reference.add z (bias c reference z)
  cast : ∀ i,
    let a:=scaled c reference query x
    let f:=finite c reference a
    let b:=safe c a f
    let s:=scale c reference b
    let z:=clipped c reference (normalized c reference (b i) s (usable c reference f s))
    native.int8 (reference.add z (bias c reference z))=reference.int8 (reference.add z (bias c reference z))

theorem local_equiv (c : Constants α) (native reference : Numerics α)
    (query : Bool) (x : Fin n → α) (laws : Local c native reference query x) :
    run c native query x=run c reference query x := by
  have hs : scaled c native query x=scaled c reference query x := by
    funext i
    cases hq : query
    · simp [scaled,hq]
    · simp [scaled,hq,laws.prescale hq]
  have hf : finite c native (scaled c reference query x)=finite c reference (scaled c reference query x) := by
    funext i;simp [finite,laws.classify_abs,laws.classify_le]
  have hscale : scale c native (safe c (scaled c reference query x) (finite c reference (scaled c reference query x)))=
    scale c reference (safe c (scaled c reference query x) (finite c reference (scaled c reference query x))) := by
    simp only [scale,laws.max_abs,laws.maximum]
    exact laws.scale_div
  have huse : usable c native (finite c reference (scaled c reference query x))
      (scale c reference (safe c (scaled c reference query x) (finite c reference (scaled c reference query x))))=
    usable c reference (finite c reference (scaled c reference query x))
      (scale c reference (safe c (scaled c reference query x) (finite c reference (scaled c reference query x)))) := by
    simp [usable,laws.positive]
  dsimp only [run]
  rw [hs,hf,hscale,huse]
  apply Prod.ext
  · funext i
    have hn:=laws.normalize i
    have hl:=laws.lower i
    have hu:=laws.upper i
    have hsg:=laws.sign i
    have hm:=laws.half_mul i
    have ha:=laws.add_bias i
    have hc:=laws.cast i
    dsimp only at hn hl hu hsg hm ha hc
    let b:=safe c (scaled c reference query x) (finite c reference (scaled c reference query x))
    let s:=scale c reference b
    let use:=usable c reference (finite c reference (scaled c reference query x)) s
    have hnval : normalized c native (b i) s use=normalized c reference (b i) s use := by
      simp only [normalized,show native.div (b i) (if use then s else c.one)=
        reference.div (b i) (if use then s else c.one) from hn]
    change bytes c native (clipped c native (normalized c native (b i) s use))=
      bytes c reference (clipped c reference (normalized c reference (b i) s use))
    rw [hnval]
    let z:=normalized c reference (b i) s use
    have hcval : clipped c native z=clipped c reference z := by
      unfold clipped
      rw [show native.max z c.negative127=reference.max z c.negative127 from hl]
      exact hu
    change bytes c native (clipped c native z)=bytes c reference (clipped c reference z)
    rw [hcval]
    have hbias : bias c native (clipped c reference z)=bias c reference (clipped c reference z) := by
      unfold bias
      rw [show native.le c.zero (clipped c reference z)=reference.le c.zero (clipped c reference z) from hsg]
      exact hm
    unfold bytes
    rw [hbias,show native.add (clipped c reference z) (bias c reference (clipped c reference z))=
      reference.add (clipped c reference z) (bias c reference (clipped c reference z)) from ha]
    exact hc
  · rfl

#axiomsClean local_equiv
end
end VeriTile.Bench.Repairs.Int8Quantization.Floating
