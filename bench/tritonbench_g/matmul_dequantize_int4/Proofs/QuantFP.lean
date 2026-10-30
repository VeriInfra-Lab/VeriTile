import bench.tritonbench_g.matmul_dequantize_int4.Proofs.Quantizer
import bench.tritonbench_g.matmul_dequantize_int4.Proofs.Physical
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.QuantFP
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG
abbrev Word := FP.Value .fp32
noncomputable section
/-- Exact finite rational nearest-integer rule, including ties to even. -/
def nearestEven (q : Rat) : Int :=
  let n := q.num.natAbs/q.den
  let r := q.num.natAbs%q.den
  let mag := if 2*r>q.den ∨ (2*r=q.den ∧ n%2=1) then n+1 else n
  if q.num<0 then -(mag : Int) else mag

def clipRound (x : Word) : Packing.Nibble :=
  BitVec.ofInt 4 (max 0 (min 15 (nearestEven (x.toRat?.getD 0))))
/-- Numerical leaves of the eager PyTorch FP32 operations; each tensor
operation preserves its separate rounding point. -/
structure Instructions where
  sub : Word → Word → Word
  div : Word → Word → Word
  add : Word → Word → Word
  clipRound : Word → Packing.Nibble

def software : Instructions := ⟨FP.sub .ieeeValue,FP.div .ieeeValue,FP.add .ieeeValue,clipRound⟩
structure Obligations (p : Instructions) : Prop where
  sub : ∀ x y,p.sub x y=FP.sub .ieeeValue x y
  div : ∀ x y,p.div x y=FP.div .ieeeValue x y
  add : ∀ x y,p.add x y=FP.add .ieeeValue x y
  clipRound : ∀ x,p.clipRound x=clipRound x

/-- Scalar comparison/select leaves, plus the actual library reduction tree.
No zero is injected before the separate public where-clamp. -/
def ops (p : Instructions) (tree : Quantizer.Tree) : Quantizer.Ops Word where
  tree := tree
  zero := FP.Value.zero .fp32
  fifteen := FP.round .fp32 .ieeeValue 15
  max := fun x y => if FP.lt .ieeeValue x y then y else x
  min := fun x y => if FP.lt .ieeeValue y x then y else x
  sub := p.sub
  div := p.div
  neg := FP.Value.neg
  add := p.add
  fromNibble := fun q => FP.round .fp32 .ieeeValue (q.toNat : Rat)
  clipRound := p.clipRound

theorem instructions_eq (p : Instructions) (h : Obligations p) : p=software := by
  have hs : p.sub=FP.sub .ieeeValue := funext fun x => funext fun y => h.sub x y
  have hd : p.div=FP.div .ieeeValue := funext fun x => funext fun y => h.div x y
  have ha : p.add=FP.add .ieeeValue := funext fun x => funext fun y => h.add x y
  have hr : p.clipRound=clipRound := funext h.clipRound
  cases p
  simp_all [software]

theorem local_transport (p : Instructions) (h : Obligations p) (s : Quantizer.Shape) (x : Nat → Nat → Word) (tree : Quantizer.Tree) :
    Quantizer.source (ops p tree) s x=Quantizer.source (ops software tree) s x := by rw [instructions_eq p h]

/-- Ordinary quantization domain excludes undefined zero-range groups and
nonfinite arithmetic. These predicates inspect scalar values of the input
and independent reference, and do not assume execution or output equality. -/
def FiniteDomain (s : Quantizer.Shape) (x : Nat → Nat → Word) (tree : Quantizer.Tree) : Prop :=
  (∀ k n,k<s.K → n<s.N → ∃ q,(x k n).toRat?=some q) ∧
  (∀ g n,g<s.groups → n<s.N → ∃ q,0<q ∧
    (Quantizer.targetScale (ops software tree) s x g n).toRat?=some q) ∧
  (∀ g n,g<s.groups → n<s.N → ∃ q,
    (FP.div .ieeeValue (FP.Value.neg (Quantizer.low (ops software tree) s.group (Quantizer.groupValues s x g n)))
      (Quantizer.targetScale (ops software tree) s x g n)).toRat?=some q) ∧
  (∀ k n,k<s.K → n<s.N →
    ∃ u v,(FP.div .ieeeValue (x k n) (Quantizer.targetScale (ops software tree) s x (k/s.group) n)).toRat?=some u ∧
      (FP.add .ieeeValue (FP.div .ieeeValue (x k n) (Quantizer.targetScale (ops software tree) s x (k/s.group) n))
        ((ops software tree).fromNibble (Quantizer.targetZero (ops software tree) s x (k/s.group) n))).toRat?=some v)

theorem clip_bound (x : Word) : (clipRound x).toNat<16 := (clipRound x).isLt
#specNonCircular FiniteDomain avoiding [Quantizer.source,Quantizer.sourceQ,Quantizer.sourceScale]
#axiomsClean instructions_eq
#axiomsClean local_transport
#axiomsClean clip_bound
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.QuantFP
