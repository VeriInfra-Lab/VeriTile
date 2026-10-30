import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Source
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence
noncomputable section
namespace Source

def forwardResult (ops : Ops α) (c : Shape) (hasSeed : Bool) (p : Nat) (m : Memory α) (n : Nat) : State α :=
  ⟨fun r q t j => if r=.output ∧ q=p ∧ t≤n ∧ j<1024 then
    Reference.forward ops c (inputsOf m) hasSeed p t j else m r q t j,
   Reference.forward ops c (inputsOf m) hasSeed p n⟩

theorem forwardLoop_complete (ops : Ops α) (c : Shape) (hasSeed : Bool) (p : Nat) (m : Memory α) (n : Nat) :
    forwardLoop ops c hasSeed p m n=forwardResult ops c hasSeed p m n := by
  induction n with
  | zero =>
    dsimp only [forwardLoop,forwardInit,forwardResult]
    apply congrArg₂ State.mk
    · funext r q t j
      simp only [Memory.tile,Nat.le_zero]
      split_ifs <;> simp_all [Reference.forward,inputsOf]
    · rfl

  | succ n ih =>
    rw [forwardLoop,ih]
    dsimp only [forwardStep,forwardResult]
    apply congrArg₂ State.mk
    · funext r q t j
      simp only [Memory.tile]
      split_ifs <;> simp_all [Reference.forward,inputsOf] <;> omega
    · funext j
      simp [Reference.forward,inputsOf]

def forwardImage (ops : Ops α) (c : Shape) (hasSeed : Bool) (m : Memory α) (count : Nat) : Memory α :=
  fun r p t j => if r=.output ∧ p<count ∧ t<c.N ∧ j<1024 then
    Reference.forward ops c (inputsOf m) hasSeed p t j else m r p t j

theorem forwardImage_inputs (ops : Ops α) (c : Shape) (hasSeed : Bool) (m : Memory α) (n : Nat) :
    inputsOf (forwardImage ops c hasSeed m n)=inputsOf m := by
  rfl

theorem forwardGrid_complete (ops : Ops α) (c : Shape) (hasSeed : Bool) (m : Memory α) (n : Nat) :
    forwardGrid ops c hasSeed m n=forwardImage ops c hasSeed m n := by
  induction n with
  | zero => funext r p t j; simp [forwardGrid,forwardImage]
  | succ n ih =>
    rw [forwardGrid,ih]
    by_cases hz : c.N=0
    · funext r p t j
      simp [forwardKernel,hz,forwardImage]
    · rw [forwardKernel,if_neg hz,forwardLoop_complete]
      unfold forwardResult
      rw [forwardImage_inputs]
      funext r p t j
      simp only [forwardImage]
      split_ifs <;> simp_all <;> omega

theorem forwardGrid_output (ops : Ops α) (c : Shape) (hasSeed : Bool) (m : Memory α)
    (p t j : Nat) (hp : p<c.programs) (ht : t<c.N) (hj : j<1024) :
    forwardGrid ops c hasSeed m c.programs .output p t j =
      Reference.forward ops c (inputsOf m) hasSeed p t j := by
  rw [forwardGrid_complete]
  simp [forwardImage,hp,ht,hj]

#axiomsClean forwardLoop_complete
#axiomsClean forwardGrid_complete
#axiomsClean forwardGrid_output
end Source
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence
