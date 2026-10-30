import bench.tritonbench_g.matmul_dequantize_int4.Proofs.Packing
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.Quantizer
set_option maxHeartbeats 4000000
abbrev Nibble := Packing.Nibble
abbrev Packed := Packing.Word
/- Scalar operations of the selected FP32 PyTorch graph. `clipRound` is one
scalar round-to-even, clamp-to-[0,15], and int32 conversion. No group reduction,
packed tensor, output tensor, or whole quantizer is a numerical port. -/
/-- Nonempty public-library reduction structure. Leaves name group coordinates;
legality requires each index exactly once. This is data, not a numerical port. -/
inductive Tree where
  | leaf : Nat → Tree
  | node : Tree → Tree → Tree
  deriving DecidableEq, Repr
def Tree.leaves : Tree → List Nat
  | .leaf i => [i]
  | .node a b => a.leaves++b.leaves
def Tree.eval {α} (f : α → α → α) (x : Nat → α) : Tree → α
  | .leaf i => x i
  | .node a b => f (a.eval f x) (b.eval f x)
def Tree.Valid (t : Tree) (g : Nat) : Prop := t.leaves.Perm (List.range g)
theorem Tree.eval_congr {α} (t : Tree) (f : α → α → α) (x y : Nat → α)
    (h : ∀ i∈t.leaves,x i=y i) : t.eval f x=t.eval f y := by
  induction t with
  | leaf i => exact h i (by simp [Tree.leaves])
  | node a b ha hb =>
    simp only [Tree.eval,ha (fun i hi => h i (by simp [Tree.leaves,hi])),
      hb (fun i hi => h i (by simp [Tree.leaves,hi]))]
structure Ops (α : Type) where
  tree : Tree
  zero : α
  fifteen : α
  max : α → α → α
  min : α → α → α
  sub : α → α → α
  div : α → α → α
  neg : α → α
  add : α → α → α
  fromNibble : Nibble → α
  clipRound : α → Nibble
structure Shape where
  K : Nat
  N : Nat
  group : Nat
  deriving DecidableEq, Repr
def Shape.groups (s : Shape) := s.K/s.group
structure Shape.Valid (s : Shape) : Prop where
  k : 0<s.K
  n : 0<s.N
  group : 0<s.group
  k8 : 8∣s.K
  n8 : 8∣s.N
  groups : s.group∣s.K
/-- Source amax/amin reduction, followed by the separate where clamp.
The final comparison keeps the reduction's signed zero on equality. -/
def high {α} (p : Ops α) (_g : Nat) (x : Nat → α) :=
  p.max (p.tree.eval p.max x) p.zero
def low {α} (p : Ops α) (_g : Nat) (x : Nat → α) :=
  p.min (p.tree.eval p.min x) p.zero
def scale {α} (p : Ops α) (g : Nat) (x : Nat → α) := p.div (p.sub (high p g x) (low p g x)) p.fifteen
def zero {α} (p : Ops α) (g : Nat) (x : Nat → α) := p.clipRound (p.div (p.neg (low p g x)) (scale p g x))
def quant {α} (p : Ops α) (g : Nat) (x : Nat → α) (v : α) :=
  p.clipRound (p.add (p.div v (scale p g x)) (p.fromNibble (zero p g x)))
/-- Target coordinates refer directly to the original K-by-N input. -/
def groupValues {α} (s : Shape) (input : Nat → Nat → α) (g n t : Nat) := input (g*s.group+t) n
def targetScale {α} (p : Ops α) (s : Shape) (input : Nat → Nat → α) (g n : Nat) :=
  scale p s.group (groupValues s input g n)
def targetZero {α} (p : Ops α) (s : Shape) (input : Nat → Nat → α) (g n : Nat) :=
  zero p s.group (groupValues s input g n)
def targetQ {α} (p : Ops α) (s : Shape) (input : Nat → Nat → α) (k n : Nat) :=
  quant p s.group (groupValues s input (k/s.group) n) (input k n)
def targetPackedQ {α} (p : Ops α) (s : Shape) (input : Nat → Nat → α) (k n : Nat) :=
  Packing.packed (fun i : Fin 8 => targetQ p s input (k*8+i.val) n)
def targetPackedZ {α} (p : Ops α) (s : Shape) (input : Nat → Nat → α) (g n : Nat) :=
  Packing.packed (fun i : Fin 8 => targetZero p s input g (n*8+i.val))

/-- Actual transpose, contiguous row-major copy, and view(-1, group). CUDA
copy preserves every element and changes only device/allocation identity. -/
def transposed {α} (input : Nat → Nat → α) (n k : Nat) := input k n
def copied {α} (s : Shape) (input : Nat → Nat → α) (flat : Nat) :=
  transposed input (flat/s.K) (flat%s.K)
def grouped {α} (s : Shape) (input : Nat → Nat → α) (row t : Nat) :=
  copied s input (row*s.group+t)

theorem group_coordinates (s : Shape) (h : s.Valid) (n g t : Nat)
    (hg : g<s.groups) (ht : t<s.group) :
    ((n*s.groups+g)*s.group+t)/s.K=n ∧
    ((n*s.groups+g)*s.group+t)%s.K=g*s.group+t := by
  have hk : s.groups*s.group=s.K := Nat.div_mul_cancel h.groups
  have bound : g*s.group+t<s.K := by
    have := Nat.mul_le_mul_right s.group (show g+1≤s.groups by omega)
    rw [Nat.add_mul,hk] at this
    omega
  have eq : (n*s.groups+g)*s.group+t=n*s.K+(g*s.group+t) := by rw [Nat.add_mul,Nat.mul_assoc,hk];omega
  rw [eq]
  constructor
  · rw [Nat.add_comm (n*s.K),Nat.add_mul_div_right _ _ h.k,Nat.div_eq_of_lt bound,Nat.zero_add]
  · rw [Nat.add_comm (n*s.K),Nat.add_mul_mod_self_right,Nat.mod_eq_of_lt bound]

theorem grouped_original {α} (s : Shape) (h : s.Valid) (input : Nat → Nat → α)
    (n g t : Nat) (hg : g<s.groups) (ht : t<s.group) :
    grouped s input (n*s.groups+g) t=groupValues s input g n t := by
  simp only [grouped,copied,transposed,(group_coordinates s h n g t hg ht).1,
    (group_coordinates s h n g t hg ht).2,groupValues]

theorem fold_values {α β} (f : β → α → β) (xs : List Nat) (a b : Nat → α)
    (h : ∀ i∈xs,a i=b i) (z : β) : xs.foldl (fun z i => f z (a i)) z=xs.foldl (fun z i => f z (b i)) z := by
  induction xs generalizing z with
  | nil => rfl
  | cons i xs ih =>
    simp only [List.foldl_cons,h i (by simp)]
    exact ih (fun j hj => h j (by simp [hj])) _

theorem stats_original {α} (p : Ops α) (s : Shape) (h : s.Valid) (input : Nat → Nat → α)
    (n g : Nat) (hg : g<s.groups) (tree : p.tree.Valid s.group) :
    high p s.group (grouped s input (n*s.groups+g))=high p s.group (groupValues s input g n) ∧
    low p s.group (grouped s input (n*s.groups+g))=low p s.group (groupValues s input g n) := by
  have eq (t : Nat) (ht : t∈p.tree.leaves) :
      grouped s input (n*s.groups+g) t=groupValues s input g n t :=
    grouped_original s h input n g t hg (List.mem_range.mp (tree.mem_iff.mp ht))
  constructor <;> (simp only [high,low]; congr 1; exact Tree.eval_congr _ _ _ _ eq)

def sourceScale {α} (p : Ops α) (s : Shape) (input : Nat → Nat → α) (n g : Nat) :=
  scale p s.group (grouped s input (n*s.groups+g))
def sourceZero {α} (p : Ops α) (s : Shape) (input : Nat → Nat → α) (n g : Nat) :=
  zero p s.group (grouped s input (n*s.groups+g))
def sourceQ {α} (p : Ops α) (s : Shape) (input : Nat → Nat → α) (n k : Nat) :=
  quant p s.group (grouped s input (n*s.groups+k/s.group))
    (grouped s input (n*s.groups+k/s.group) (k%s.group))

theorem scale_original {α} (p : Ops α) (s : Shape) (h : s.Valid) (input : Nat → Nat → α)
    (n g : Nat) (hg : g<s.groups) (tree : p.tree.Valid s.group) : sourceScale p s input n g=targetScale p s input g n := by
  simp only [sourceScale,targetScale,scale,(stats_original p s h input n g hg tree).1,(stats_original p s h input n g hg tree).2]
theorem zero_original {α} (p : Ops α) (s : Shape) (h : s.Valid) (input : Nat → Nat → α)
    (n g : Nat) (hg : g<s.groups) (tree : p.tree.Valid s.group) : sourceZero p s input n g=targetZero p s input g n := by
  simp only [sourceZero,targetZero,zero,scale,(stats_original p s h input n g hg tree).1,(stats_original p s h input n g hg tree).2]
theorem q_original {α} (p : Ops α) (s : Shape) (h : s.Valid) (input : Nat → Nat → α)
    (n k : Nat) (hk : k<s.K) (tree : p.tree.Valid s.group) : sourceQ p s input n k=targetQ p s input k n := by
  have hg : k/s.group<s.groups := by
    apply (Nat.div_lt_iff_lt_mul h.group).mpr
    simpa [Shape.groups,Nat.div_mul_cancel h.groups] using hk
  have hx := grouped_original s h input n (k/s.group) (k%s.group) hg (Nat.mod_lt _ h.group)
  simp only [groupValues,Nat.div_add_mod'] at hx
  simp only [sourceQ,targetQ,quant,zero,scale,hx,
    (stats_original p s h input n (k/s.group) hg tree).1,(stats_original p s h input n (k/s.group) hg tree).2]

/-- Vectorized outer packing loop. Each column starts at zero; its eight
updates are exactly the wrapping signed-int32 shift-and-add source recurrence. -/
def packColumns (values : Nat → Nat → Nibble) : Nat → Nat → Nat → Packed
  | 0 => fun _ _ => 0
  | count+1 => fun row col => if col=count then
      Packing.seeded (packColumns values count row col) (fun i => values row (count*8+i.val)) 8
    else packColumns values count row col

theorem packColumns_value (values : Nat → Nat → Nibble) (count row col : Nat) :
    packColumns values count row col=
      if col<count then Packing.packed (fun i : Fin 8 => values row (col*8+i.val)) else 0 := by
  induction count with
  | zero => simp [packColumns]
  | succ count ih =>
    by_cases he : col=count
    · subst col
      simp only [packColumns,ite_true,ih,Nat.lt_irrefl,ite_false,Packing.zero_seed_correct]
      simp
    · simp only [packColumns,he,ite_false,ih]
      have hi : (col<count+1) ↔ col<count := by omega
      simp only [hi]

structure Result (α : Type) where
  Q : Nat → Nat → Packed
  S : Nat → Nat → α
  Z : Nat → Nat → Packed
  group : Nat
/-- The three final transpose(...).contiguous() copies have the returned
Q[K/8,N], S[K/group,N], Z[K/group,N/8] layouts. -/
def source {α} (p : Ops α) (s : Shape) (input : Nat → Nat → α) : Result α where
  Q := fun k n => packColumns (sourceQ p s input) (s.K/8) n k
  S := fun g n => sourceScale p s input n g
  Z := fun g n => packColumns (fun g n => sourceZero p s input n g) (s.N/8) g n
  group := s.group

theorem whole_quantizer {α} (p : Ops α) (s : Shape) (h : s.Valid) (input : Nat → Nat → α) (tree : p.tree.Valid s.group) :
    (∀ k n,k<s.K/8 → n<s.N → (source p s input).Q k n=targetPackedQ p s input k n) ∧
    (∀ g n,g<s.groups → n<s.N → (source p s input).S g n=targetScale p s input g n) ∧
    (∀ g n,g<s.groups → n<s.N/8 → (source p s input).Z g n=targetPackedZ p s input g n) ∧
    (source p s input).group=s.group := by
  refine ⟨?_,?_,?_,rfl⟩
  · intro k n hk hn
    rw [show (source p s input).Q k n=packColumns (sourceQ p s input) (s.K/8) n k from rfl,
      packColumns_value,if_pos hk]
    unfold targetPackedQ
    congr 1
    funext i
    apply q_original p s h input n _ _ tree
    have hh := Nat.mul_le_mul_right 8 (show k+1≤s.K/8 by omega)
    rw [Nat.add_mul,Nat.div_mul_cancel h.k8] at hh
    have := i.isLt
    omega
  · intro g n hg hn
    exact scale_original p s h input n g hg tree
  · intro g n hg hn
    rw [show (source p s input).Z g n=packColumns (fun g n => sourceZero p s input n g) (s.N/8) g n from rfl,
      packColumns_value,if_pos hn]
    unfold targetPackedZ
    congr 1
    funext i
    exact zero_original p s h input (n*8+i.val) g hg tree

#specNonCircular targetPackedQ avoiding [source,sourceQ,grouped,copied,transposed,packColumns]
#specNonCircular targetPackedZ avoiding [source,sourceZero,grouped,copied,transposed,packColumns]
#axiomsClean group_coordinates
#axiomsClean q_original
#axiomsClean packColumns_value
#axiomsClean whole_quantizer
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.Quantizer
