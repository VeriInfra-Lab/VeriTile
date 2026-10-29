import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.PublicMath
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Plans

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence
noncomputable section
open scoped BigOperators

namespace Source
def backwardRun (ops : Ops α) (c : Shape) (trees : Trees c) (seed : Bool) (m : Memory α) : Memory α :=
  if c.N=0 then emptyBackward ops c seed m else
    hostSum ops c trees (zeroTail ops c (backwardGrid ops c trees m c.programs))

def savedProduct (ops : Ops α) (c : Shape) (trees : Trees c) (m : Memory α) (p t : Nat) : α :=
  if t+1<c.N then tileValue ops c trees m p t else ops.zero

/-- The independent backward reference reads the saved forward allocation.
It has no forward input snapshot and permits incoming gradients to alias it. -/
def backwardTarget (ops : Ops α) (c : Shape) (trees : Trees c) (seed : Bool) (m : Memory α) : Memory α :=
  fun r p t j =>
    if r=.dx ∧ p<c.programs ∧ t<c.N ∧ j<1024 then Reference.dx ops c (inputsOf m) p t j
    else if r=.tileGate ∧ p<c.programs ∧ t<c.N ∧ j=0 then savedProduct ops c trees m p t
    else if r=.dseed ∧ p<c.programs ∧ t=0 ∧ j<1024 ∧ (c.N>0 ∨ seed) then
      Reference.reverse ops c (inputsOf m) p c.N j
    else if r=.dgate ∧ p<c.heads ∧ t<c.N ∧ j=0 then
      Reduction.eval ops (trees.host p t) (fun q => savedProduct ops c trees m (p*c.tiles+q) t)
    else m r p t j

theorem saved_scratch (ops : Ops α) (c : Shape) (trees : Trees c) (m : Memory α)
    (p t : Nat) (hp : p<c.programs) (ht : t<c.N) :
    zeroTail ops c (backwardImage ops c trees m c.programs) .tileGate p t 0=
      savedProduct ops c trees m p t := by
  by_cases last : t=c.N-1
  · simp [zeroTail,hp,last,show c.N>0 by omega,savedProduct,show ¬c.N-1+1<c.N by omega]
  · simp [zeroTail,last,backwardImage,hp,savedProduct,show t+1<c.N by omega]

theorem backward_complete (ops : Ops α) (c : Shape) (trees : Trees c) (seed : Bool) (m : Memory α) :
    backwardRun ops c trees seed m=backwardTarget ops c trees seed m := by
  unfold backwardRun
  by_cases zero : c.N=0
  · rw [if_pos zero]
    funext r p t j
    simp only [emptyBackward,backwardTarget,zero,Nat.not_lt_zero,false_and,and_false,↓reduceIte,
      Reference.reverse,Nat.lt_irrefl,false_or]
    split_ifs <;> simp_all
  · rw [if_neg zero,backwardGrid_complete]
    funext r p t j
    by_cases gate : r=.dgate ∧ p<c.heads ∧ t<c.N ∧ j=0
    · obtain ⟨rfl,hp,ht,rfl⟩ := gate
      simp only [hostSum,backwardTarget,reduceCtorEq,false_and,true_and,and_true,hp,ht,↓reduceIte]
      congr 1
      funext q
      exact saved_scratch ops c trees m _ t (c.program_bound p q hp q.isLt) ht
    · simp only [hostSum,if_neg gate]
      have nonempty : c.N>0 := by omega
      cases r with
      | dx =>
        by_cases hp : p<c.programs <;> by_cases ht : t<c.N <;>
          by_cases hj : j<1024 <;> by_cases hl : t=c.N-1 <;>
          simp_all [zeroTail,backwardImage,backwardTarget,Reference.dx] <;>
          (try split_ifs) <;> (try simp_all [Reference.reverse]) <;> omega
      | tileGate =>
        by_cases hp : p<c.programs
        · by_cases ht : t<c.N
          · by_cases hj : j=0
            · subst j
              simpa [backwardTarget,hp,ht] using saved_scratch ops c trees m p t hp ht
            · simp [zeroTail,backwardImage,backwardTarget,hj]
          · have ht1 : ¬t+1<c.N := by omega
            have hl : t≠c.N-1 := by omega
            simp [zeroTail,backwardImage,backwardTarget,ht,ht1,hl]
        · simp [zeroTail,backwardImage,backwardTarget,hp]
      | output | dseed | x | gate | seed | incoming | dgate | caller =>
        simp_all [zeroTail,backwardImage,backwardTarget]

theorem savedProduct_real (c : Shape) (trees : Trees c) (m : Memory ℝ) (p t : Nat) (ht : t<c.N) :
    savedProduct realOps c trees m p t=
      ∑ j : Fin 1024,MathTarget.reverse c (inputsOf m) p (t+1) j*m .output p t j := by
  unfold savedProduct
  split_ifs with h
  · rw [tileValue,Reduction.product_real]
    apply Finset.sum_congr rfl
    intro j _
    rw [RealProof.reverse c (inputsOf m) p _ j (by omega),show c.N-(c.N-1-t)=t+1 by omega]
  · simp [show t+1=c.N by omega,realOps]
end Source

namespace SplitNative
/-- The concrete singleton postlude retains the selected gate-first FMA.
All other operations follow the complete backward source, including Torch sum. -/
def backward (isa : ISA) (c : Shape) (trees : Trees c) (seed : Bool) (m : Memory Word) : Memory Word :=
  fun r p t j => if c.N=1 ∧ r=.dseed ∧ p<c.programs ∧ t=0 ∧ j<1024 then
    isa.fma (m .gate (p/c.tiles) 0 0) 0 (m .incoming p 0 j)
    else Source.backwardRun (wordOps isa) c trees seed m r p t j

theorem scalar_ext (isa : ISA) (laws : LocalConformance isa allOperands) : isa=SourceFP.scalar := by
  cases isa with
  | mk add mul fma =>
    have a : add=SourceFP.scalar.add := funext fun x => funext fun y => laws.add x y trivial
    have b : mul=SourceFP.scalar.mul := funext fun x => funext fun y => laws.mul x y trivial
    have c : fma=SourceFP.scalar.fma := funext fun x => funext fun y => funext fun z => laws.fma x y z trivial
    cases a
    cases b
    cases c
    rfl

theorem backward_complete (isa : ISA) (laws : LocalConformance isa allOperands)
    (c : Shape) (trees : Trees c) (seed : Bool) (m : Memory Word) :
    backward isa c trees seed m=Source.backwardTarget (wordOps SourceFP.scalar) c trees seed m := by
  rw [scalar_ext isa laws]
  funext r p t j
  unfold backward
  split_ifs with h
  · obtain ⟨hn,rfl,hp,rfl,hj⟩ := h
    simp only [Source.backwardTarget,reduceCtorEq,false_and,true_and,hp,hj,hn,Nat.lt_one_iff,
      ↓reduceIte,Reference.reverse,wordOps,inputsOf]
    exact SourceFP.zero_fma_swap _ _
  · rw [Source.backward_complete]

#axiomsClean backward_complete
end SplitNative
#axiomsClean Source.backward_complete
#axiomsClean Source.savedProduct_real
#specNonCircular Source.backwardTarget avoiding [Source.backwardRun,Source.backwardGrid,Source.run]
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence
