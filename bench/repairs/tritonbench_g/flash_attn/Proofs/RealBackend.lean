import bench.repairs.tritonbench_g.flash_attn.Proofs.RealSum
-- Preserve the historical Bool spellings after importing the shared Triton DSL.
local notation "true" => Bool.true
local notation "false" => Bool.false
namespace VeriTile.Bench.Repairs.FlashAttn.Real
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
abbrev R := WithBot ℝ

def exp2 : R → R
  | none=>some 0
  | some a=>some (pow2 a)
def mma8 (_ : Width) (a b : Fin 8 → R) (seed : R) : R :=
  seed.map (fun z=>z+RealSum.sumReal (fun i=>(a ⟨i%8,Nat.mod_lt _ (by decide)⟩).unbotD 0*
    (b ⟨i%8,Nat.mod_lt _ (by decide)⟩).unbotD 0) 8)
def ops : Ops R where
  zero:=some 0
  negInf:=none
  log2e:=some 1.44269504
  add:=WithBot.realAdd
  sub:=WithBot.realSub
  mul:=WithBot.realMul
  fma:=fun a b c=>WithBot.realAdd (WithBot.realMul a b) c
  max:=max
  exp2:=exp2
  log2:=WithBot.realLog2
  div:=WithBot.realDiv
  narrow:=id
  widen:=id
  mma8:=mma8

def lift (m : Memory ℝ) : Memory R := fun r a=>some (m r a)

theorem mma_some (w : Width) (a b : Nat → ℝ) (chunk : Nat) (z : ℝ) :
    mma8 w (fun i=>some (a (chunk*8+i.val))) (fun i=>some (b (chunk*8+i.val))) (some z)=
      some (z+RealSum.sumReal (fun i=>a (chunk*8+i)*b (chunk*8+i)) 8) := by
  simp only [mma8,WithBot.unbotD_coe,Option.map_some]
  congr 2

theorem dot_some (w : Width) (k : Nat) (a b : Nat → ℝ) (z : ℝ) :
    dot ops w k (fun i=>some (a i)) (fun i=>some (b i)) (some z)=
      some (z+RealSum.sumReal (fun i=>a i*b i) (k/8*8)) := by
  unfold dot
  have aux : ∀ n,(List.range n).foldl (fun acc chunk=>
      ops.mma8 w (fun i=>some (a (chunk*8+i.val))) (fun i=>some (b (chunk*8+i.val))) acc) (some z)=
        some (z+RealSum.sumReal (fun i=>a i*b i) (n*8)) := by
    intro n
    induction n with
    | zero=>simp [RealSum.sumReal]
    | succ n ih=>
      simp only [List.range_succ,List.foldl_append,List.foldl_cons,List.foldl_nil,ih]
      simp only [ops]
      rw [mma_some]
      simp [Nat.succ_mul,RealSum.sum_add,add_assoc]
  exact aux (k/8)

theorem dot_bot (w : Width) (k : Nat) (a b : Nat → R) : dot ops w k a b none=none := by
  unfold dot
  have aux : ∀ ns : List Nat, ns.foldl (fun acc chunk=>
      ops.mma8 w (fun i=>a (chunk*8+i.val)) (fun i=>b (chunk*8+i.val)) acc) none=none := by
    intro ns
    induction ns with
    | nil=>rfl
    | cons n ns ih=>exact ih
  exact aux _

theorem dim_div (s : Shape) (hs : s.Valid) : s.dim/8*8=s.dim := by
  have hd : s.dim=16 ∨ s.dim=32 ∨ s.dim=64 ∨ s.dim=128 := by simpa using hs.2.2.2
  rcases hd with h|h|h|h <;> simp [h]

theorem query_real (x : Layout) (m : Memory ℝ) (scale : ℝ) (b h row e : Nat)
    (hr : row<x.shape.length) : query x ops (lift m) (some scale) b h row e=
      some (scale*1.44269504*logical m .q x.q b h row e) := by
  simp [query,ops,lift,logical,hr,WithBot.realMul]

theorem key_real (x : Layout) (m : Memory ℝ) (b h row e : Nat) :
    key x ops (lift m) b h row e=
      some (if row<x.shape.length then logical m .k x.k b h row e else 0) := by
  by_cases hr : row<x.shape.length <;> simp [key,ops,lift,logical,hr]
theorem value_real (x : Layout) (m : Memory ℝ) (b h row col : Nat) :
    value x ops (lift m) b h row col=
      some (if row<x.shape.length then logical m .v x.v b h row col else 0) := by
  by_cases hr : row<x.shape.length <;> simp [value,ops,lift,logical,hr]

theorem score_real (x : Layout) (hs : x.shape.Valid) (m : Memory ℝ) (scale : ℝ)
    (b h row keyRow : Nat) (hr : row<x.shape.length) :
    score x ops (lift m) (some scale) b h row keyRow=
      if visible x.shape row keyRow then some (Spec.score x m scale b h row keyRow) else none := by
  have hq : query x ops (lift m) (some scale) b h row=
      (fun e=>some (scale*1.44269504*logical m .q x.q b h row e)) :=
    funext (fun e=>query_real x m scale b h row e hr)
  have hk : key x ops (lift m) b h keyRow=
      (fun e=>some (if keyRow<x.shape.length then logical m .k x.k b h keyRow e else 0)) :=
    funext (key_real x m b h keyRow)
  unfold score
  rw [hq,hk]
  by_cases vis : visible x.shape row keyRow=true
  · have hkN : keyRow<x.shape.length := by
      have : keyRow<x.shape.length ∧ (x.shape.causal=false ∨ keyRow≤row) := by simpa [visible] using vis
      exact this.1
    simp only [vis,ite_true,ops]
    change dot Real.ops .n64 x.shape.dim _ _ (some 0)=_
    rw [dot_some,dim_div x.shape hs,RealSum.sumReal_list]
    simp [hkN,Spec.score]
  · simp only [vis,ite_false]
    exact dot_bot _ _ _ _

theorem tree_map (t : Reduction.Tree) (f : α → β) (a : α → α → α) (b : β → β → β)
    (input : Nat → α) (hom : ∀ x y,b (f x) (f y)=f (a x y)) :
    t.eval b (fun i=>f (input i))=f (t.eval a input) := by
  induction t with
  | leaf i=>rfl
  | merge l r hl hr=>simp [Reduction.Tree.eval,hl,hr,hom]
  | shuffle mask t ih=>exact ih

theorem reduce_sum (owner : Fin 4) (f : Nat → ℝ) :
    (Reduction.plan owner).eval ops.add (fun j=>some (f j))=
      some (((List.range 64).map f).sum) := by
  have hh:=tree_map (Reduction.plan owner) (fun x : ℝ=>(some x : R)) (·+·) ops.add f (fun _ _=>rfl)
  exact hh.trans (congrArg (fun z : ℝ=>(some z : R)) (Reduction.sum_exact owner f))

/-- Every reduction leaf participates once; the real max erasure may use
semilattice laws, whereas the native path retains all four physical trees. -/
theorem tree_max (t : Reduction.Tree) (f : Nat → R) :
    t.eval (fun (a b : R)=>a⊔b) f=(t.leaves.map f).foldr (fun (a b : R)=>a⊔b) ⊥ := by
  induction t with
  | leaf i=>simp [Reduction.Tree.eval,Reduction.Tree.leaves]
  | merge l r hl hr=>
    simp only [Reduction.Tree.eval,Reduction.Tree.leaves,List.map_append,hl,hr,List.foldr_append]
    have aux : ∀ xs : List R, ∀ z : R, xs.foldr (fun (a b : R)=>a⊔b) z=(xs.foldr (fun (a b : R)=>a⊔b) ⊥)⊔z := by
      intro xs
      induction xs with
      | nil=>intro z;simp
      | cons a xs ih=>
        intro z
        simp only [List.foldr_cons]
        rw [ih z]
        exact (sup_assoc _ _ _).symm
    exact (aux _ _).symm
  | shuffle mask t ih=>exact ih

theorem reduce_max (owner : Fin 4) (f : Nat → R) :
    (Reduction.plan owner).eval ops.max f=((List.range 64).map f).foldr (fun (a b : R)=>a⊔b) ⊥ := by
  change (Reduction.plan owner).eval (fun (a b : R)=>a⊔b) f=_
  rw [tree_max]
  exact List.Perm.foldr_eq (f:=fun (a b : R)=>a⊔b) ((Reduction.coverage owner).map f) ⊥

#axiomsClean dot_some
#axiomsClean score_real
#axiomsClean reduce_sum
#axiomsClean reduce_max
end
end VeriTile.Bench.Repairs.FlashAttn.Real
