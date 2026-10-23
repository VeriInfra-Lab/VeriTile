import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelTensorInputs

/-! Input-only word formulas over actual logical Tensor elements. -/
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelLogical
open VeriTile.Triton FP.Structural KernelProgram KernelTensor KernelTensorInputs KernelCarrier
noncomputable section

theorem forward (isa : Instructions) (swap : Bool) (b : Buffers c s .forward seed)
    (read : Nat → Word) (p : Program c) (u : State Node) (n : Nat) (bound : n≤c.N) :
    KernelWords.forward isa swap (regional b read p u) seed n=
      KernelWords.forward isa swap (logical b read p) seed n := by
  induction n with
  | zero =>
    funext i
    simp only [KernelWords.forward,forward_seed b read p u i]
  | succ n ih =>
    funext i
    simp only [KernelWords.forward,ih (by omega),
      gate b read p u ⟨n,by omega⟩,forward_source b read p u ⟨n,by omega⟩ i]

theorem reverse (isa : Instructions) (swap : Bool) (b : Buffers c s .backward seed)
    (read : Nat → Word) (p : Program c) (u : State Node) (n : Nat) (bound : n≤c.N) :
    KernelWords.reverse isa swap (regional b read p u) c.N n=
      KernelWords.reverse isa swap (logical b read p) c.N n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    funext i
    simp only [KernelWords.reverse,ih (by omega),
      gate b read p u ⟨c.N-1-n,by omega⟩,backward_incoming b read p u ⟨c.N-1-n,by omega⟩ i]

def expected (isa : Instructions) (plan : Reduction.ProductPlan 1024)
    (b : Buffers c s kind seed) (read : Nat → Word) (p : Program c)
    (r : RegionName) (t : Nat) (i : Lane) : Word :=
  let a := logical b read p
  match kind with
  | .forward => KernelWords.forward isa (swapped .forward c) a seed t i
  | .backward =>
    if r="DI" then KernelWords.reverse isa (swapped .backward c) a c.N (c.N-1-t) i
    else if r="DG" then KernelWords.contraction isa plan
      (KernelWords.reverse isa (swapped .backward c) a c.N (c.N-1-t)) (a.saved t)
    else KernelWords.reverse isa (swapped .backward c) a c.N c.N i

theorem expected_eq (isa : Instructions) (plan : Reduction.ProductPlan 1024)
    (b : Buffers c s kind seed) (read : Nat → Word) (p : Program c) (u : State Node)
    (r : RegionName) (t : Fin c.N) (i : Lane) :
    KernelNative.expected isa plan kind c s seed p (physical b (fun a => atom (read a)) u) r t.val i=
      expected isa plan b read p r t.val i := by
  cases kind with
  | forward =>
    exact congrFun (forward isa _ b read p u t.val (Nat.le_of_lt t.isLt)) i
  | backward =>
    change (if r="DI" then KernelWords.reverse isa _ (regional b read p u) c.N (c.N-1-t.val) i
      else if r="DG" then KernelWords.contraction isa plan
        (KernelWords.reverse isa _ (regional b read p u) c.N (c.N-1-t.val)) ((regional b read p u).saved t.val)
      else KernelWords.reverse isa _ (regional b read p u) c.N c.N i)=_
    rw [reverse isa _ b read p u (c.N-1-t.val) (by omega),reverse isa _ b read p u c.N (Nat.le_refl _)]
    have saved : (regional b read p u).saved t.val=(logical b read p).saved t.val := by
      funext j
      exact backward_saved b read p u t j
    rw [saved]
    rfl

#axiomsClean expected_eq
#specNonCircular expected avoiding [FP.Structural.exec,FP.Structural.run,KernelExecution.expected,KernelNative.expected]
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelLogical
