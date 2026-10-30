import bench.repairs.tritonbench_g.fused_rotary_embedding.Kernels
import Mathlib.Data.Nat.Log

namespace VeriTile.Bench.Repairs.FusedRotaryEmbedding

inductive Kind where
  | q | k | v
  deriving DecidableEq, Repr
def Kind.input : Kind → Region | .q => .q | .k => .k | .v => .v
def Kind.output : Kind → Region | .q => .q | .k => .kcache | .v => .vcache

structure Shape where
  tokens : Nat
  headsQ : Nat
  headsK : Nat
  width : Nat
  cosWidth : Nat
  sinWidth : Nat
  blocks : Nat
  blockSize : Nat
  tableWidth : Nat
  split : Nat
  newLayout : Bool
  deriving DecidableEq, Repr
def Shape.half (s : Shape) := s.width / 2
def Shape.groups (s : Shape) := s.headsQ / s.headsK
def Shape.heads (s : Shape) (kind : Kind) := if kind=.q then s.headsQ else s.headsK
def Shape.halfBlock (s : Shape) := 2 ^ Nat.clog 2 s.half
def Shape.fullBlock (s : Shape) := 2 ^ Nat.clog 2 s.width
def Shape.data (s : Shape) (kind : Kind) (t h d : Nat) := (t*s.heads kind+h)*s.width+d
def Shape.coeff (s : Shape) (sine : Bool) (t d : Nat) := t*(if sine then s.sinWidth else s.cosWidth)+d
def Shape.cache (s : Shape) (key : Bool) (block head slot dim : Nat) : Nat :=
  if key && s.newLayout then
    (((block*s.headsK+head)*(s.width/s.split)+dim/s.split)*s.blockSize+slot)*s.split+dim%s.split
  else ((block*s.headsK+head)*s.blockSize+slot)*s.width+dim

structure Shape.Valid (s : Shape) : Prop where
  hq : 0<s.headsQ
  hk : 0<s.headsK
  width : 0<s.width
  even : s.width%2=0
  grouping : s.headsQ%s.headsK=0
  cos : s.half≤s.cosWidth
  sin : s.half≤s.sinWidth
  blocks : 0<s.blocks
  blockSize : 0<s.blockSize
  tableWidth : 0<s.tableWidth
  split : 0<s.split
  divides : s.width%s.split=0

/-- Values are decoded nonnegative signed-int32 metadata, not output ports. -/
structure Metadata where
  lengths : Nat → Nat
  table : Nat → Nat

structure Config where
  shape : Shape
  metadata : Metadata
def Config.past (c : Config) (t : Nat) := c.metadata.lengths t-1
def Config.block (c : Config) (t : Nat) := c.metadata.table (t*c.shape.tableWidth+c.past t/c.shape.blockSize)
def Config.slot (c : Config) (t : Nat) := c.past t%c.shape.blockSize

structure Config.Valid (c : Config) : Prop where
  shape : c.shape.Valid
  lengthPositive : ∀ t<c.shape.tokens,0<c.metadata.lengths t
  length32 : ∀ t<c.shape.tokens,c.metadata.lengths t<2^31
  lengthTable : ∀ t<c.shape.tokens,c.past t/c.shape.blockSize<c.shape.tableWidth
  blockBound : ∀ t<c.shape.tokens,c.block t<c.shape.blocks
  block32 : ∀ t<c.shape.tokens,c.block t<2^31
  distinct : ∀ t<c.shape.tokens,∀ u<c.shape.tokens,c.block t=c.block u → c.slot t=c.slot u → t=u

structure Work where
  kind : Kind
  token : Nat
  head : Nat
  pair : Nat
  deriving DecidableEq, Repr
def Shape.Active (s : Shape) (w : Work) := w.token<s.tokens ∧ w.head<s.heads w.kind ∧ w.pair<s.half
instance (s : Shape) (w : Work) : Decidable (s.Active w) := inferInstanceAs (Decidable (_∧_∧_))
def Shape.dim (s : Shape) (w : Work) (second : Bool) := w.pair+if second then s.half else 0
def Shape.input (s : Shape) (w : Work) (second : Bool) := s.data w.kind w.token w.head (s.dim w second)
def Config.output (c : Config) (w : Work) (second : Bool) :=
  if w.kind=.q then c.shape.input w second
  else c.shape.cache (w.kind=.k) (c.block w.token) w.head (c.slot w.token) (c.shape.dim w second)

theorem width_halves (s : Shape) (h : s.Valid) : 2*s.half=s.width := by
  have := h.even
  unfold Shape.half
  omega
theorem dim_bound (s : Shape) (h : s.Valid) (w : Work) (hw : s.Active w) (o : Bool) : s.dim w o<s.width := by
  have := width_halves s h
  have := hw.2.2
  cases o <;> simp [Shape.dim] <;> omega
theorem dim_injective (s : Shape) (a b : Work) (ha : a.pair<s.half) (hb : b.pair<s.half)
    (oa ob : Bool) (eq : s.dim a oa=s.dim b ob) : a.pair=b.pair ∧ oa=ob := by
  cases oa <;> cases ob <;> simp_all [Shape.dim] <;> omega

theorem radix_inj (n a b a' b' : Nat) (hb : b<n) (hb' : b'<n)
    (eq : a*n+b=a'*n+b') : a=a' ∧ b=b' := by
  have hn : 0<n := lt_of_le_of_lt (Nat.zero_le _) hb
  have qa := congrArg (fun z => z/n) eq
  have ra := congrArg (fun z => z%n) eq
  constructor
  · simpa only [Nat.add_comm (_*n),Nat.add_mul_div_right _ _ hn,Nat.div_eq_of_lt hb,
      Nat.div_eq_of_lt hb',Nat.zero_add] using qa
  · simpa [Nat.add_mod,Nat.mod_eq_of_lt hb,Nat.mod_eq_of_lt hb'] using ra

theorem data_injective (s : Shape) (kind : Kind) (t h d t' h' d' : Nat)
    (hh : h<s.heads kind) (hh' : h'<s.heads kind) (hd : d<s.width) (hd' : d'<s.width)
    (eq : s.data kind t h d=s.data kind t' h' d') : t=t' ∧ h=h' ∧ d=d' := by
  have e := radix_inj s.width _ _ _ _ hd hd' eq
  have f := radix_inj (s.heads kind) _ _ _ _ hh hh' e.1
  exact ⟨f.1,f.2,e.2⟩

theorem cache_injective (s : Shape) (hs : s.Valid) (key : Bool) (b h l d b' h' l' d' : Nat)
    (hh : h<s.headsK) (hh' : h'<s.headsK) (hl : l<s.blockSize) (hl' : l'<s.blockSize)
    (hd : d<s.width) (hd' : d'<s.width)
    (eq : s.cache key b h l d=s.cache key b' h' l' d') : b=b' ∧ h=h' ∧ l=l' ∧ d=d' := by
  unfold Shape.cache at eq
  split at eq
  · have er := radix_inj s.split _ _ _ _ (Nat.mod_lt _ hs.split) (Nat.mod_lt _ hs.split) eq
    have es := radix_inj s.blockSize _ _ _ _ hl hl' er.1
    have dw : s.width/s.split*s.split=s.width := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hs.divides)
    have hdq : d/s.split<s.width/s.split := (Nat.div_lt_iff_lt_mul hs.split).mpr (by simpa [dw] using hd)
    have hdq' : d'/s.split<s.width/s.split := (Nat.div_lt_iff_lt_mul hs.split).mpr (by simpa [dw] using hd')
    have ed := radix_inj (s.width/s.split) _ _ _ _ hdq hdq' es.1
    have eh := radix_inj s.headsK _ _ _ _ hh hh' ed.1
    refine ⟨eh.1,eh.2,es.2,?_⟩
    calc
      d = d%s.split+s.split*(d/s.split) := (Nat.mod_add_div _ _).symm
      _ = d'%s.split+s.split*(d'/s.split) := by rw [er.2,ed.2]
      _ = d' := Nat.mod_add_div _ _
  · have ed := radix_inj s.width _ _ _ _ hd hd' eq
    have es := radix_inj s.blockSize _ _ _ _ hl hl' ed.1
    have eh := radix_inj s.headsK _ _ _ _ hh hh' es.1
    exact ⟨eh.1,eh.2,es.2,ed.2⟩

theorem output_region_injective : Function.Injective Kind.output := by
  intro a b eq
  cases a <;> cases b <;> simp_all [Kind.output]

theorem output_injective (c : Config) (hc : c.Valid) (a b : Work) (oa ob : Bool)
    (ha : c.shape.Active a) (hb : c.shape.Active b)
    (region : a.kind.output=b.kind.output) (eq : c.output a oa=c.output b ob) : a=b ∧ oa=ob := by
  have kind := output_region_injective region
  have da := dim_bound c.shape hc.shape a ha oa
  have db := dim_bound c.shape hc.shape b hb ob
  have dimhead : a.token=b.token ∧ a.head=b.head ∧ c.shape.dim a oa=c.shape.dim b ob := by
    by_cases hq : a.kind=.q
    · have hq' : b.kind=.q := kind.symm.trans hq
      apply data_injective c.shape a.kind a.token a.head _ b.token b.head _ ha.2.1
        (by simpa [kind] using hb.2.1) da db
      simpa [Config.output,Shape.input,hq,hq'] using eq
    · have hq' : b.kind≠.q := by simpa [←kind] using hq
      have e := cache_injective c.shape hc.shape (a.kind=.k) (c.block a.token) a.head
        (c.slot a.token) _ (c.block b.token) b.head (c.slot b.token) _
        (by simpa [Shape.heads,hq] using ha.2.1) (by simpa [Shape.heads,hq'] using hb.2.1)
        (Nat.mod_lt _ hc.shape.blockSize) (Nat.mod_lt _ hc.shape.blockSize) da db
        (by simpa [Config.output,hq,hq',kind] using eq)
      exact ⟨hc.distinct _ ha.1 _ hb.1 e.1 e.2.2.1,e.2.1,e.2.2.2⟩
  have dims := dim_injective c.shape a b ha.2.2 hb.2.2 oa ob dimhead.2.2
  exact ⟨by cases a;cases b;simp_all [Work.mk.injEq],dims.2⟩

#axiomsClean output_injective
end VeriTile.Bench.Repairs.FusedRotaryEmbedding
