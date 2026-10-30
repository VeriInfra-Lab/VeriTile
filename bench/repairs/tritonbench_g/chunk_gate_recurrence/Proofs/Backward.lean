import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Forward
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence
noncomputable section
namespace Source

def tileValue (ops : Ops α) (c : Shape) (trees : Trees c) (m : Memory α) (p t : Nat) :=
  Reduction.product ops (trees.tile p t)
    (fun j => (Reference.reverse ops c (inputsOf m) p (c.N-1-t) j,m .output p t j))

def backwardResult (ops : Ops α) (c : Shape) (trees : Trees c) (p : Nat) (m : Memory α) (n : Nat) : State α :=
  ⟨fun r q t j =>
    if r=.dx ∧ q=p ∧ c.N-1-n≤t ∧ t<c.N-1 ∧ j<1024 then
      Reference.reverse ops c (inputsOf m) p (c.N-1-t) j
    else if r=.tileGate ∧ q=p ∧ c.N-1-n≤t ∧ t<c.N-1 ∧ j=0 then
      tileValue ops c trees m p t else m r q t j,
   Reference.reverse ops c (inputsOf m) p n⟩

theorem backwardLoop_complete (ops : Ops α) (c : Shape) (trees : Trees c) (p : Nat) (m : Memory α)
    (n : Nat) (hn : n≤c.N-1) :
    backwardLoop ops c trees p m n=backwardResult ops c trees p m n := by
  induction n with
  | zero =>
    dsimp only [backwardLoop,backwardResult]
    apply congrArg₂ State.mk
    · funext r q t j
      split_ifs <;> simp_all <;> omega
    · rfl
  | succ n ih =>
    have hidx : c.N-1-(c.N-(1+(n+1)))=n+1 := by omega
    have hidx2 : c.N-1-(c.N-1-n-1)=n+1 := by omega
    rw [backwardLoop,ih (by omega)]
    dsimp only [backwardStep,backwardResult]
    simp only [reduceCtorEq, false_and, ↓reduceIte]
    apply congrArg₂ State.mk
    · funext r q t j
      simp only [Memory.tile,Memory.scalar]
      split_ifs <;> simp_all [tileValue,Reference.reverse,inputsOf] <;> try omega
    · funext j
      rfl

def backwardImage (ops : Ops α) (c : Shape) (trees : Trees c) (m : Memory α) (count : Nat) : Memory α :=
  fun r p t j =>
    if r=.dx ∧ p<count ∧ t+1<c.N ∧ j<1024 then Reference.reverse ops c (inputsOf m) p (c.N-1-t) j
    else if r=.tileGate ∧ p<count ∧ t+1<c.N ∧ j=0 then tileValue ops c trees m p t
    else if r=.dseed ∧ p<count ∧ t=0 ∧ j<1024 ∧ c.N>0 then Reference.reverse ops c (inputsOf m) p c.N j
    else m r p t j

theorem backwardImage_inputs (ops : Ops α) (c : Shape) (trees : Trees c) (m : Memory α) (n : Nat) :
    inputsOf (backwardImage ops c trees m n)=inputsOf m := by rfl

theorem backwardImage_saved (ops : Ops α) (c : Shape) (trees : Trees c) (m : Memory α) (n p t j : Nat) :
    backwardImage ops c trees m n .output p t j=m .output p t j := by rfl

theorem tileValue_image (ops : Ops α) (c : Shape) (trees : Trees c) (m : Memory α) (n p t : Nat) :
    tileValue ops c trees (backwardImage ops c trees m n) p t=tileValue ops c trees m p t := by rfl

theorem backwardGrid_complete (ops : Ops α) (c : Shape) (trees : Trees c) (m : Memory α) (n : Nat) :
    backwardGrid ops c trees m n=backwardImage ops c trees m n := by
  induction n with
  | zero => funext r p t j; simp [backwardGrid,backwardImage]
  | succ n ih =>
    rw [backwardGrid,ih]
    by_cases hz : c.N=0
    · funext r p t j
      simp [backwardKernel,hz,backwardImage]
    · rw [backwardKernel,if_neg hz,backwardLoop_complete ops c trees n _ (c.N-1) (by omega)]
      dsimp only [backwardPost,backwardResult]
      simp only [reduceCtorEq,false_and,↓reduceIte,backwardImage_inputs,tileValue_image]
      have hN : c.N-1+1=c.N := by omega
      have hzero : c.N-1-(c.N-1)=0 := by omega
      have hcarry : (fun j => ops.fma (Reference.reverse ops c (inputsOf m) n (c.N-1) j)
        (m .gate (n/c.tiles) 0 0) (m .incoming n 0 j)) = Reference.reverse ops c (inputsOf m) n c.N := by
        rw [← hN]
        simp [Reference.reverse,inputsOf,hzero]
      rw [show (backwardImage ops c trees m n .gate)=m .gate from rfl,
        show (backwardImage ops c trees m n .incoming)=m .incoming from rfl,hcarry]
      funext r p t j
      simp only [Memory.tile,backwardImage]
      split_ifs <;> simp_all <;> omega

#axiomsClean backwardLoop_complete
#axiomsClean backwardGrid_complete
end Source
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence
