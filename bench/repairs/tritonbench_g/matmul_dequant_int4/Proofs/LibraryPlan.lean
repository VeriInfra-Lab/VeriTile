import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.Memory
import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.RealSum
namespace VeriTile.Bench.Repairs.MatmulDequantInt4.Library
set_option maxHeartbeats 5000000
noncomputable section

/-- Fixed K16 scalar projections, not complete dots or tiles. Logical-to-physical
register mapping remains an ordinary selected-implementation ABI boundary. -/
inductive MMAKind where
  | mma16x8 | wgmma64x8 | wgmma64x16 | wgmma64x32
  | wgmma64x64 | wgmma64x128 | wgmma64x256
  deriving DecidableEq, Repr
structure Ops (α β : Type) where
  zeroInput : α
  zeroAcc : β
  widen : α → β
  narrow : β → α
  mul : β → β → β
  add : β → β → β
  fma : β → β → β → β
  mma16 : MMAKind → (Fin 16 → α) → (Fin 16 → α) → β → β

/-- Resolved finite library contraction. Input indices, partial sums, reduction
ordering, padding, and optional reduced-precision truncations are explicit. -/
inductive Plan (K : Nat) where
  | zero
  | product (k : Fin K)
  | fma (k : Fin K) (seed : Plan K)
  | add (left right : Plan K)
  | mma16 (kind : MMAKind) (lanes : Fin 16 → Option (Fin K)) (seed : Plan K)
  | roundHalf (seed : Plan K)
def Plan.terms : Plan K → List (Fin K)
  | .zero => []
  | .product k => [k]
  | .fma k seed => k::seed.terms
  | .add left right => left.terms++right.terms
  | .mma16 _ lanes seed => (List.ofFn lanes).filterMap id++seed.terms
  | .roundHalf seed => seed.terms
def Plan.Complete (plan : Plan K) := plan.terms.Perm (List.finRange K)
def Plan.eval (p : Ops α β) (a b : Fin K → α) : Plan K → β
  | .zero => p.zeroAcc
  | .product k => p.mul (p.widen (a k)) (p.widen (b k))
  | .fma k seed => p.fma (p.widen (a k)) (p.widen (b k)) (seed.eval p a b)
  | .add left right => p.add (left.eval p a b) (right.eval p a b)
  | .mma16 kind lanes seed => p.mma16 kind
      (fun i => ((lanes i).map a).getD p.zeroInput)
      (fun i => ((lanes i).map b).getD p.zeroInput) (seed.eval p a b)
  | .roundHalf seed => p.widen (p.narrow (seed.eval p a b))

def realOps : Ops ℝ ℝ :=
  ⟨0,0,id,id,(·*·),(·+·),(fun x y z => x*y+z),
    (fun _ a b seed => (List.ofFn (fun i => a i*b i)).sum+seed)⟩
theorem option_sum (xs : List (Option γ)) (f : γ → ℝ) :
    (xs.map (fun x => (x.map f).getD 0)).sum=((xs.filterMap id).map f).sum := by
  induction xs with
  | nil => rfl
  | cons x xs ih => cases x <;> simp [ih]
theorem Plan.eval_real (plan : Plan K) (a b : Fin K → ℝ) :
    plan.eval realOps a b=(plan.terms.map (fun k => a k*b k)).sum := by
  induction plan with
  | zero => rfl
  | product k => simp [eval,terms,realOps]
  | fma k seed ih =>
    change a k*b k+seed.eval realOps a b=_
    rw [ih];rfl
  | add left right il ir =>
    change left.eval realOps a b+right.eval realOps a b=_
    rw [il,ir]
    simp only [terms,List.map_append,List.sum_append]
  | mma16 kind lanes seed ih =>
    change (List.ofFn (fun i => ((lanes i).map a).getD 0*((lanes i).map b).getD 0)).sum+
      seed.eval realOps a b=_
    rw [ih]
    simp only [terms,List.map_append,List.sum_append]
    congr 1
    have per (i : Fin 16) : ((lanes i).map a).getD 0*((lanes i).map b).getD 0=
        ((lanes i).map (fun k => a k*b k)).getD 0 := by cases lanes i <;> simp
    simp_rw [per]
    simpa only [List.map_ofFn,Function.comp_def] using option_sum (List.ofFn lanes) (fun k => a k*b k)
  | roundHalf seed ih => exact ih
theorem Plan.real_complete (plan : Plan K) (complete : plan.Complete) (a b : Fin K → ℝ) :
    plan.eval realOps a b=(List.ofFn (fun k => a k*b k)).sum := by
  rw [plan.eval_real]
  have h := List.Perm.sum_eq (List.Perm.map (fun k => a k*b k) complete)
  simpa only [List.finRange,List.map_ofFn,Function.comp_def,id_eq] using h
theorem ofFn_sum (f : Nat → ℝ) (n : Nat) :
    (List.ofFn (fun i : Fin n => f i.val)).sum=sumReal f n := by
  induction n with
  | zero => simp [sumReal]
  | succ n ih => simp [-List.ofFn_succ,List.ofFn_succ_last,ih,sumReal]

structure Selected (s : Shape) where
  output : Nat → Nat → Plan s.k
  complete : ∀ r n,r<s.m → n<s.n → (output r n).Complete

/-- No numerical tensor contents or completed result enters the selector.
Linking this plan to the installed torch.mm/cuBLAS implementation is an explicit
ordinary library/ABI obligation, not a proved backend-selection theorem. -/
structure Metadata where
  libraryBuild : String
  algorithm : Nat
  device : Nat
  architecture : Nat
  workspaceBytes : Nat
  reducedPrecisionReduction : Bool
  aStrides : Nat × Nat
  bStrides : Nat × Nat
  outputStrides : Nat × Nat
structure Implementation where
  select : (s : Shape) → Metadata → Selected s

def value (s : Shape) (p : Ops α β) (selected : Selected s) (current : State α) (w : Work) : α :=
  p.narrow ((selected.output w.row w.col).eval p
    (fun k => current.values .a (w.row*s.k+k))
    (fun k => current.values .dequant (k*s.n+w.col)))
def run (s : Shape) (p : Ops α β) (selected : Selected s) (order : List Work) (current : State α) :=
  Writes.run .output s.n (value s p selected) order current
theorem stable (s : Shape) (p : Ops α β) (selected : Selected s) : Writes.Stable .output (value s p selected) := by
  intro initial current same w
  have a : current.values .a=initial.values .a := funext (fun a => same.1 .a a (by decide))
  have b : current.values .dequant=initial.values .dequant := funext (fun a => same.1 .dequant a (by decide))
  simp only [value,a,b]
theorem whole_result (s : Shape) (p : Ops α β) (selected : Selected s) (g : Grid s.m s.n)
    (order : List Work) (complete : g.Complete order) (initial : State α)
    (r n : Nat) (hr : r<s.m) (hn : n<s.n) :
    (run s p selected order initial).values .output (r*s.n+n)=value s p selected initial ⟨r,n⟩ :=
  Writes.complete_result .output _ (stable s p selected) g order complete initial ⟨r,n⟩ ⟨hr,hn⟩
theorem whole_frame (s : Shape) (p : Ops α β) (selected : Selected s) (g : Grid s.m s.n)
    (order : List Work) (complete : g.Complete order) (initial : State α)
    (region : Region) (address : Nat) (outside : region≠.output ∨ s.m*s.n≤address) :
    (run s p selected order initial).values region address=initial.values region address :=
  Writes.complete_frame .output _ g order complete initial region address outside

def scalarPlan (K : Nat) : Plan K := (List.finRange K).foldr Plan.fma .zero
theorem scalar_terms (ks : List (Fin K)) : (ks.foldr Plan.fma .zero).terms=ks := by
  induction ks with
  | nil => rfl
  | cons k ks ih => simp [Plan.terms,ih]
/-- Consistency witness only; no claim that any installed library selects it. -/
def scalarWitness : Implementation where
  select := fun s _ => ⟨fun _ _ => scalarPlan s.k,fun _ _ _ _ => by
    unfold scalarPlan Plan.Complete
    rw [scalar_terms]⟩

#axiomsClean Plan.eval_real
#axiomsClean Plan.real_complete
#axiomsClean whole_result
#axiomsClean whole_frame
#axiomsClean scalarWitness
end
end VeriTile.Bench.Repairs.MatmulDequantInt4.Library
