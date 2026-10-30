import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.Source
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.Math

namespace VeriTile.Bench.Repairs.FastCeLoss.RealProof
open Source VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000

def maximum {n : Nat} (values : Fin n → WithBot ℝ) : WithBot ℝ :=
  if h : 0<n then Finset.univ.sup' ⟨⟨0,h⟩,Finset.mem_univ _⟩ values else none

def hostLSE {n : Nat} (values : Fin n → WithBot ℝ) : WithBot ℝ :=
  let m := maximum values
  Option.map₂ (fun x y => x+y) m
    (WithBot.realLog (∑ i,WithBot.realExp (Option.map₂ (fun x y => x-y) (values i) m)))

def arithmetic : Arithmetic (WithBot ℝ) where
  literal := fun x => some x
  natural := fun n => some n
  negInf := none
  load := id
  cast := id
  add := Option.map₂ (·+·)
  sub := Option.map₂ (·-·)
  mul := Option.map₂ (·*·)
  div := Option.map₂ (·/·)
  exp := WithBot.realExp
  log := WithBot.realLog
  tanh := WithBot.realTanh
  maximum := maximum
  sum := fun xs => ∑ i,xs i
  hostSum := fun xs => ∑ i,xs i
  hostLSE := hostLSE
  hostAdd := Option.map₂ (·+·)
  hostDiv := Option.map₂ (·/·)
  upstreamDiv := Option.map₂ (·/·)

theorem stable_active {n : Nat} (p : Fin n → Prop) [DecidablePred p]
    (f : Fin n → ℝ) (hn : 0<n) (hp : (Finset.univ.filter p).Nonempty) :
    stableLSE arithmetic (fun i => if p i then some (f i) else none)=
      some (Real.log (∑ i ∈ Finset.univ.filter p,Real.exp (f i))) := by
  have h := VeriTile.Bench.TritonBenchG.FastCeLoss.RealMath.carrier_lse p f
    ⟨⟨0,hn⟩,Finset.mem_univ _⟩ hp
  simpa only [stableLSE,arithmetic,maximum,dif_pos hn] using h

theorem host_lse_finite {n : Nat} (f : Fin n → ℝ) (hn : 0<n) :
    arithmetic.hostLSE (fun i => some (f i))=some (Real.log (∑ i,Real.exp (f i))) := by
  have h := stable_active (fun _ : Fin n => True) f hn (by exact ⟨⟨0,hn⟩,by simp⟩)
  simpa [stableLSE,arithmetic,hostLSE] using h

theorem transform_finite (o : Options) (x : ℝ) :
    transform arithmetic o (some x)=some (Math.transformed o.softcap o.scaling x) := by
  rw [transform,o.capSource,o.scaleSource]
  by_cases hc : o.softcap=0 <;> by_cases hs : o.scaling=0
  all_goals simp [arithmetic,Math.transformed,Math.scale,hc,hs,WithBot.realTanh]

theorem tile_finite (o : Options) (s : Shape) (b : Buffers) (m : Memory (WithBot ℝ))
    (q : Task) (x : Nat → ℝ)
    (inputs : ∀ col<s.vocab, m (addr b.logits (q.1*s.vocab+col))=some (x col)) :
    tile arithmetic o s b m q = fun lane : Fin s.block =>
      if q.2*s.block+lane.val<s.vocab then
        some (Math.transformed o.softcap o.scaling (x (q.2*s.block+lane.val))) else none := by
  funext lane
  by_cases active : q.2*s.block+lane.val<s.vocab
  · simp only [tile,active,↓reduceIte,inputs _ active,arithmetic,id_eq]
    exact transform_finite o _
  · simp [tile,active,arithmetic]

theorem forward_lse (o : Options) (s : Shape) (b : Buffers) (m : Memory (WithBot ℝ))
    (q : Task) (x : Nat → ℝ) (blockPositive : 0<s.block) (active : q.2*s.block<s.vocab)
    (inputs : ∀ col<s.vocab, m (addr b.logits (q.1*s.vocab+col))=some (x col)) :
    forwardLSE arithmetic o s b m q = some (Real.log
      (∑ lane ∈ (Finset.univ : Finset (Fin s.block)).filter (fun j => q.2*s.block+j.val<s.vocab),
        Real.exp (Math.transformed o.softcap o.scaling (x (q.2*s.block+lane.val))))) := by
  rw [forwardLSE,tile_finite o s b m q x inputs]
  apply stable_active _ _ blockPositive
  exact ⟨⟨0,blockPositive⟩,by simp [active]⟩

def scalarGradient (o : Options) (label : Int) (col : Nat) (x lse dloss : ℝ) : ℝ :=
  if label = -100 then 0 else dloss *
    (Real.exp (Math.transformed o.softcap o.scaling x-lse) - if (col : Int)=label then 1 else 0) *
    Math.derivative o.softcap o.scaling x

theorem backward_scalar (o : Options) (label : Int) (col : Nat) (x lse dloss : ℝ) :
    backwardScalar arithmetic o label col (some x) (some lse) (some dloss)=
      some (scalarGradient o label col x lse dloss) := by
  unfold backwardScalar scalarGradient
  rw [o.scaleSource,o.capSource]
  by_cases hc : o.softcap=0 <;> by_cases hs : o.scaling=0
  all_goals by_cases ignored : label = -100
  all_goals by_cases target : (col : Int)=label
  all_goals simp [arithmetic,Math.transformed,Math.derivative,Math.scale,hc,hs,ignored,target,
    WithBot.realTanh,WithBot.realExp,mul_comm] <;> congr 1 <;> ring

#axiomsClean stable_active
#axiomsClean host_lse_finite
#axiomsClean transform_finite
#axiomsClean forward_lse
#axiomsClean backward_scalar
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.RealProof
