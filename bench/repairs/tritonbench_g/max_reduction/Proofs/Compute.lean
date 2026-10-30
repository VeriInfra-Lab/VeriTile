import bench.repairs.tritonbench_g.max_reduction.Proofs.Native
import bench.repairs.tritonbench_g.max_reduction.Proofs.Address
import bench.repairs.tritonbench_g.max_reduction.Proofs.Real

namespace VeriTile.Bench.Repairs.MaxReduction.Compute
open VeriTile.Triton
noncomputable section
open Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 24000

structure Algebra (A : Type) where
  maximum : A → A → A
  padding : A
  indexedCombine : A × Nat → A × Nat → A × Nat

def native (ops : NativeOps) : Algebra Word := ⟨ops.maximum,negativeInfinity,nativePair ops⟩
def nativeReference (target : Native.Target) : Algebra Word :=
  ⟨Native.maximum target,negativeInfinity,Native.referencePair⟩

def realCombine (p q : WithBot ℝ × Nat) : WithBot ℝ × Nat :=
  match p.1,q.1 with
  | none,_ => q
  | _,none => p
  | some a,some b =>
      let selected := RealModel.pair (a,p.2) (b,q.2)
      (selected.1,selected.2)

def real : Algebra (WithBot ℝ) := ⟨max,⊥,realCombine⟩

def scalar (algebra : Algebra A) (tree : Tree) (input : Nat → A) (count start : Nat) : A :=
  scalarReduction algebra.maximum algebra.padding tree input count start

def dimension (algebra : Algebra A) (tree : Tree) (input : Nat → A) (N K j : Nat) : A × IndexWord :=
  let selected := tree.eval algebra.indexedCombine
    (fun n => (if n<N then input (j/K*N*K+n*K+j%K) else algebra.padding,n))
  (selected.1,indexStore selected.2)

theorem tree_congr (tree : Tree) (op : A → A → A) (f g : Nat → A)
    (same : ∀ i,i∈tree.leaves → f i=g i) : tree.eval op f=tree.eval op g :=
  Native.tree_equiv tree op op f g (fun _ _ => rfl) same

theorem scalar_local (algebra : Algebra A) (tree : Tree) (f g : Nat → A)
    (count start : Nat) (same : ∀ i,i<count → f i=g i) :
    scalar algebra tree f count start=scalar algebra tree g count start := by
  apply tree_congr
  intro lane _
  split_ifs with h
  · rw [same _ h]
  · rfl

theorem dimension_local (algebra : Algebra A) (tree : Tree) (M N K : Nat) :
    ∀ f g : Nat → A,(∀ i,i<M*N*K → f i=g i) → ∀ j,j<M*K →
      dimension algebra tree f N K j=dimension algebra tree g N K j := by
  intro f g hs j hj
  have hk : 0<K := by nlinarith
  have hm : j/K<M := (Nat.div_lt_iff_lt_mul hk).mpr hj
  have hjk := Nat.mod_lt j hk
  unfold dimension
  have he : tree.eval algebra.indexedCombine
      (fun n => (if n<N then f (j/K*N*K+n*K+j%K) else algebra.padding,n))=
      tree.eval algebra.indexedCombine
      (fun n => (if n<N then g (j/K*N*K+n*K+j%K) else algebra.padding,n)) := by
    apply tree_congr
    intro n _
    split_ifs with hn
    · have hb : j/K*N*K+n*K+j%K<M*N*K := Address.input_bound M N K _ n _ hm hn hjk
      rw [hs _ hb]
    · rfl
  dsimp only
  rw [he]

theorem native_scalar_equiv (ops : NativeOps) (target : Native.Target)
    (h : Native.LocalObligations ops target) (tree : Tree) (input : Nat → Word) (count start : Nat) :
    scalar (native ops) tree input count start=scalar (nativeReference target) tree input count start :=
  Native.stage_equiv ops target h tree input count start

theorem native_dimension_equiv (ops : NativeOps) (target : Native.Target)
    (h : Native.LocalObligations ops target) (tree : Tree) (input : Nat → Word) (N K j : Nat) :
    dimension (native ops) tree input N K j=dimension (nativeReference target) tree input N K j :=
  Native.dimension_equiv ops target h tree input N K (j/K) (j%K)

/-- The indexed Real projection discards bottom-valued padding, retaining
both the finite mathematical value and its original logical index. -/
def observe (p : WithBot ℝ × Nat) : Option (ℝ × Nat) :=
  match p.1 with | none => none | some a => some (a,p.2)

theorem observe_combine (p q : WithBot ℝ × Nat) :
    observe (realCombine p q)=RealModel.join (observe p) (observe q) := by
  rcases p with ⟨p,i⟩
  rcases q with ⟨q,j⟩
  cases p <;> cases q <;> rfl

theorem observe_tree (tree : Tree) (values : Nat → WithBot ℝ × Nat) :
    observe (tree.eval realCombine values)=tree.eval RealModel.join (fun i => observe (values i)) := by
  induction tree with
  | leaf i => rfl
  | node l r il ir => simp only [Tree.eval,observe_combine,il,ir]

theorem dimension_real (input : Nat → ℝ) (N K j width : Nat)
    (hn : 0<N) (cover : N≤width) (plan : ReductionPlan width) :
    ∃ (answer : ℝ) (index : Nat),dimension real plan.tree (fun i => (input i : WithBot ℝ)) N K j=
      ((answer : WithBot ℝ),indexStore index) ∧
      RealModel.IsLeastArgmax (fun n => input (j/K*N*K+n*K+j%K)) N answer index := by
  obtain ⟨answer,index,he,hs⟩ := RealModel.indexed_correct
    (fun n => input (j/K*N*K+n*K+j%K)) N width hn cover plan
  have ho := observe_tree plan.tree
    (fun n => (if n<N then (input (j/K*N*K+n*K+j%K) : WithBot ℝ) else ⊥,n))
  have hv : (fun i => observe (if i<N then (input (j/K*N*K+i*K+j%K) : WithBot ℝ) else ⊥,i))=
      (fun i => if i<N then some (input (j/K*N*K+i*K+j%K),i) else none) := by
    funext i; split_ifs <;> rfl
  rw [hv] at ho
  change _=RealModel.indexed plan.tree (fun n => input (j/K*N*K+n*K+j%K)) N at ho
  rw [he] at ho
  let p := plan.tree.eval realCombine
    (fun n => (if n<N then (input (j/K*N*K+n*K+j%K) : WithBot ℝ) else ⊥,n))
  have hp : observe p=some (answer,index) := ho
  have hp' : p=((answer : WithBot ℝ),index) := by
    rcases p with ⟨value,idx⟩
    cases value with
    | bot => simp [observe] at hp
    | coe value => simpa [observe] using hp
  refine ⟨answer,index,?_,hs⟩
  change (p.1,indexStore p.2)=_
  rw [hp']

#axiomsClean scalar_local
#axiomsClean dimension_local
#axiomsClean dimension_real
#axiomsClean native_dimension_equiv
end
end VeriTile.Bench.Repairs.MaxReduction.Compute
