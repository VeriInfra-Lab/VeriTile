import bench.tritonbench_g.matmul_dequantize_int4.Proofs.QuantHost
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.QuantReal
open Quantizer
noncomputable section
set_option maxHeartbeats 4000000
/-- Mathematical extrema of the set of original group values together with zero. -/
def values (g : Nat) (x : Nat → ℝ) : Finset ℝ := insert 0 ((Finset.range g).image x)
def upper (g : Nat) (x : Nat → ℝ) : ℝ := (values g x).sup' (by simp [values]) id
def lower (g : Nat) (x : Nat → ℝ) : ℝ := (values g x).inf' (by simp [values]) id

theorem upper_le (g : Nat) (x : Nat → ℝ) (z : ℝ) :
    upper g x≤z ↔ 0≤z ∧ ∀ k,k<g → x k≤z := by
  simp [upper,Finset.sup'_le_iff,values]
theorem le_lower (g : Nat) (x : Nat → ℝ) (z : ℝ) :
    z≤lower g x ↔ z≤0 ∧ ∀ k,k<g → z≤x k := by
  simp [lower,Finset.le_inf'_iff,values]
theorem tree_max_le (t : Tree) (x : Nat → ℝ) (z : ℝ) :
    t.eval max x≤z ↔ ∀ k∈t.leaves,x k≤z := by
  induction t with
  | leaf k => simp [Tree.eval,Tree.leaves]
  | node a b ha hb => simp [Tree.eval,Tree.leaves,ha,hb,or_imp,forall_and]
theorem le_tree_min (t : Tree) (x : Nat → ℝ) (z : ℝ) :
    z≤t.eval min x ↔ ∀ k∈t.leaves,z≤x k := by
  induction t with
  | leaf k => simp [Tree.eval,Tree.leaves]
  | node a b ha hb => simp [Tree.eval,Tree.leaves,ha,hb,or_imp,forall_and]

theorem extrema (t : Tree) (g : Nat) (h : t.Valid g) (x : Nat → ℝ) :
    max (t.eval max x) 0=upper g x ∧ min (t.eval min x) 0=lower g x := by
  have hm (z : ℝ) : max (t.eval max x) 0≤z ↔ upper g x≤z := by
    simp only [max_le_iff,tree_max_le,upper_le]
    constructor
    · rintro ⟨ha,hz⟩; exact ⟨hz,fun k hk => ha k (h.mem_iff.mpr (List.mem_range.mpr hk))⟩
    · rintro ⟨hz,ha⟩; exact ⟨fun k hk => ha k (List.mem_range.mp (h.mem_iff.mp hk)),hz⟩
  have hn (z : ℝ) : z≤min (t.eval min x) 0 ↔ z≤lower g x := by
    simp only [le_min_iff,le_tree_min,le_lower]
    constructor
    · rintro ⟨ha,hz⟩; exact ⟨hz,fun k hk => ha k (h.mem_iff.mpr (List.mem_range.mpr hk))⟩
    · rintro ⟨hz,ha⟩; exact ⟨fun k hk => ha k (List.mem_range.mp (h.mem_iff.mp hk)),hz⟩
  exact ⟨le_antisymm ((hm _).mpr le_rfl) ((hm _).mp le_rfl),
    le_antisymm ((hn _).mp le_rfl) ((hn _).mpr le_rfl)⟩

/-- Nearest integer with even ties, then bounded unsigned nibble. -/
def nearest (x : ℝ) : Int :=
  let q := Int.floor x
  if x-(q:ℝ)>1/2 ∨ (x-(q:ℝ)=1/2 ∧ q%2=1) then q+1 else q
def clip (x : ℝ) : Packing.Nibble := BitVec.ofInt 4 (max 0 (min 15 (nearest x)))
def ops (t : Tree) : Ops ℝ := ⟨t,0,15,max,min,(·-·),(·/·),(-·),(·+·),fun q => q.toNat,clip⟩
def scale (s : Shape) (x : Nat → Nat → ℝ) (g n : Nat) : ℝ :=
  (upper s.group (groupValues s x g n)-lower s.group (groupValues s x g n))/15
def zero (s : Shape) (x : Nat → Nat → ℝ) (g n : Nat) : Packing.Nibble :=
  clip (-lower s.group (groupValues s x g n)/scale s x g n)
def quant (s : Shape) (x : Nat → Nat → ℝ) (k n : Nat) : Packing.Nibble :=
  clip (x k n/scale s x (k/s.group) n+(zero s x (k/s.group) n).toNat)
def packedQ (s : Shape) (x : Nat → Nat → ℝ) (k n : Nat) : Packing.Word :=
  Packing.packed (fun i => quant s x (k*8+i.val) n)
def packedZ (s : Shape) (x : Nat → Nat → ℝ) (g n : Nat) : Packing.Word :=
  Packing.packed (fun i => zero s x g (n*8+i.val))

theorem target_scale (t : Tree) (s : Shape) (h : t.Valid s.group) (x : Nat → Nat → ℝ) (g n : Nat) :
    targetScale (ops t) s x g n=scale s x g n := by
  simp only [targetScale,Quantizer.scale,high,low,ops,scale,(extrema t _ h _).1,(extrema t _ h _).2]
theorem target_zero (t : Tree) (s : Shape) (h : t.Valid s.group) (x : Nat → Nat → ℝ) (g n : Nat) :
    targetZero (ops t) s x g n=zero s x g n := by
  simp only [targetZero,Quantizer.zero,Quantizer.scale,high,low,ops,zero,scale,
    (extrema t _ h _).1,(extrema t _ h _).2]
theorem target_quant (t : Tree) (s : Shape) (h : t.Valid s.group) (x : Nat → Nat → ℝ) (k n : Nat) :
    targetQ (ops t) s x k n=quant s x k n := by
  change clip (x k n/targetScale (ops t) s x (k/s.group) n+
    (targetZero (ops t) s x (k/s.group) n).toNat)=_
  rw [target_scale t s h,target_zero t s h]
  rfl

theorem whole_real (t : Tree) (s : Shape) (h : s.Valid) (ht : t.Valid s.group) (x : Nat → Nat → ℝ) :
    (∀ k n,k<s.K/8 → n<s.N → (source (ops t) s x).Q k n=packedQ s x k n) ∧
    (∀ g n,g<s.groups → n<s.N → (source (ops t) s x).S g n=scale s x g n) ∧
    (∀ g n,g<s.groups → n<s.N/8 → (source (ops t) s x).Z g n=packedZ s x g n) ∧
    (source (ops t) s x).group=s.group := by
  have hs := whole_quantizer (ops t) s h x ht
  refine ⟨?_,?_,?_,hs.2.2.2⟩
  · intro k n hk hn
    rw [hs.1 k n hk hn]
    unfold targetPackedQ packedQ
    congr 1
    funext i
    exact target_quant t s ht x _ n
  · intro g n hg hn
    rw [hs.2.1 g n hg hn,target_scale t s ht]
  · intro g n hg hn
    rw [hs.2.2.1 g n hg hn]
    unfold targetPackedZ packedZ
    congr 1
    funext i
    exact target_zero t s ht x g _

#specNonCircular packedQ avoiding [source,targetPackedQ,high,low,Tree.eval]
#specNonCircular packedZ avoiding [source,targetPackedZ,high,low,Tree.eval]
#axiomsClean extrema
#axiomsClean target_quant
#axiomsClean whole_real
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.QuantReal
