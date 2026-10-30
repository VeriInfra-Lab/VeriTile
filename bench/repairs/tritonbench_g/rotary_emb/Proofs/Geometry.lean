import bench.repairs.tritonbench_g.rotary_emb.Proofs.Model
import Mathlib.Data.Nat.Log

namespace VeriTile.Bench.Repairs.RotaryEmb

structure Shape where
  tokens : Nat
  headsQ : Nat
  headsK : Nat
  width : Nat
  rotary : Nat
  cosWidth : Nat
  sinWidth : Nat
  deriving DecidableEq, Repr

def Shape.heads (s : Shape) (key : Bool) := if key then s.headsK else s.headsQ
def Shape.pairs (s : Shape) := s.rotary / 2
def Shape.headBlocks (s : Shape) := (max s.headsQ s.headsK + 3) / 4
def Shape.seqBlocks (s : Shape) := (s.tokens + 15) / 16
def Shape.pairBlock (s : Shape) := 2 ^ Nat.clog 2 s.pairs

/-- Ordinary contiguous FP32 allocation contract. Tensor values are unrestricted
raw words; this does not impose a finite-value numerical domain. -/
structure Shape.Valid (s : Shape) : Prop where
  headsQ : 0 < s.headsQ
  headsK : 0 < s.headsK
  width : 0 < s.width
  rotary_le : s.rotary ≤ s.width
  rotary_even : s.rotary % 2 = 0
  cosWidth : s.rotary ≤ s.cosWidth
  sinWidth : s.rotary ≤ s.sinWidth

def Shape.address (s : Shape) (key : Bool) (token head dim : Nat) : Nat :=
  (token * s.heads key + head) * s.width + dim

def Shape.coefficient (s : Shape) (sine : Bool) (token dim : Nat) : Nat :=
  token * (if sine then s.sinWidth else s.cosWidth) + dim

structure Work where
  key : Bool
  token : Nat
  head : Nat
  pair : Nat
  deriving DecidableEq, Repr

def Shape.Active (s : Shape) (w : Work) : Prop :=
  w.token < s.tokens ∧ w.head < s.heads w.key ∧ w.pair < s.pairs

instance (s : Shape) (w : Work) : Decidable (s.Active w) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))

def Shape.cell (s : Shape) (w : Work) (odd : Bool) : Nat :=
  s.address w.key w.token w.head (2*w.pair + if odd then 1 else 0)

theorem heads_covered (s : Shape) (key : Bool) (h : Nat) (hh : h < s.heads key) :
    h < s.headBlocks*4 := by
  have hm : s.heads key ≤ max s.headsQ s.headsK := by
    cases key <;> simp [Shape.heads]
  unfold Shape.headBlocks
  omega

theorem pairs_covered (s : Shape) : s.pairs ≤ s.pairBlock :=
  Nat.le_pow_clog (by decide) _

theorem even_width (s : Shape) (h : s.Valid) : 2*s.pairs = s.rotary := by
  have he := h.rotary_even
  unfold Shape.pairs
  omega

theorem pair_dim_bound (s : Shape) (h : s.Valid) (w : Work)
    (active : s.Active w) (odd : Bool) :
    2*w.pair + (if odd then 1 else 0) < s.rotary := by
  have hp := active.2.2
  have he := even_width s h
  cases odd <;> simp only [Bool.false_eq_true, ↓reduceIte] <;> omega

theorem address_injective (s : Shape) (key : Bool) (n h d n' h' d' : Nat)
    (hh : h < s.heads key) (hh' : h' < s.heads key)
    (hd : d < s.width) (hd' : d' < s.width)
    (eq : s.address key n h d = s.address key n' h' d') :
    n=n' ∧ h=h' ∧ d=d' := by
  unfold Shape.address at eq
  have hw : 0 < s.width := lt_of_le_of_lt (Nat.zero_le _) hd
  have hhpos : 0 < s.heads key := lt_of_le_of_lt (Nat.zero_le _) hh
  have rows : n*s.heads key+h=n'*s.heads key+h' := by
    have quot := congrArg (fun a => a / s.width) eq
    simpa only [Nat.add_comm (_ * s.width), Nat.add_mul_div_right _ _ hw,
      Nat.div_eq_of_lt hd, Nat.div_eq_of_lt hd', Nat.zero_add] using quot
  have dims : d=d' := by
    have rem := congrArg (fun a => a % s.width) eq
    simpa [Nat.add_mod, Nat.mod_eq_of_lt hd, Nat.mod_eq_of_lt hd'] using rem
  have tokens : n=n' := by
    have quot := congrArg (fun a => a / s.heads key) rows
    simpa only [Nat.add_comm (_ * s.heads key), Nat.add_mul_div_right _ _ hhpos,
      Nat.div_eq_of_lt hh, Nat.div_eq_of_lt hh', Nat.zero_add] using quot
  exact ⟨tokens, by simpa [tokens] using rows, dims⟩

theorem dataRegion_injective : Function.Injective dataRegion := by
  intro a b h
  cases a <;> cases b <;> simp_all [dataRegion]

theorem cell_injective (s : Shape) (valid : s.Valid) (a b : Work) (oa ob : Bool)
    (ha : s.Active a) (hb : s.Active b)
    (region : dataRegion a.key = dataRegion b.key)
    (cell : s.cell a oa = s.cell b ob) : a=b ∧ oa=ob := by
  have hk := dataRegion_injective region
  have boundA := lt_of_lt_of_le (pair_dim_bound s valid a ha oa) valid.rotary_le
  have boundB := lt_of_lt_of_le (pair_dim_bound s valid b hb ob) valid.rotary_le
  have eq := address_injective s a.key a.token a.head _ b.token b.head _ ha.2.1
    (by simpa [hk] using hb.2.1) boundA boundB (by simpa [Shape.cell,hk] using cell)
  have hp : a.pair=b.pair := by
    have hd := eq.2.2
    cases oa <;> cases ob <;> simp_all <;> omega
  have ho : oa=ob := by
    have hd := eq.2.2
    cases oa <;> cases ob <;> simp_all <;> omega
  exact ⟨by cases a; cases b; simp_all [Work.mk.injEq],ho⟩

#axiomsClean heads_covered
#axiomsClean pairs_covered
#axiomsClean cell_injective
end VeriTile.Bench.Repairs.RotaryEmb
