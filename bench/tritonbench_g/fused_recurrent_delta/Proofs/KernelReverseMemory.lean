import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardMemory
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelReverseMemory
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters KernelBackwardRegisters KernelBackwardMemory
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 12000000
variable {α : Type} [Inhabited α]

def carry (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (initial : State α) (n : Nat) :=
  KernelBackwardReference.reverse M s scale (KernelBackwardReference.input M s f bh iv initial) n
def enhanced (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (initial : State α) (n : Nat) :=
  KernelBackwardReference.enhance M s scale (KernelBackwardReference.input M s f bh iv initial)
    (s.T-1-n) (carry M s f scale bh iv initial n)
def memory (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (initial : State α) : Nat → State α
  | 0 => initial
  | n+1 =>
    let a := KernelBackwardReference.input M s f bh iv initial
    let t := s.T-1-n
    let g := enhanced M s f scale bh iv initial n
    betaStore M s f bh iv t (KernelBackwardReference.dbeta M s a t g)
      (KernelMemory.rowStore M s "dv" bh iv s.BBV t (KernelBackwardReference.dv M s a t g)
        (keyStore M s "dk" (plane s bh iv) t (KernelBackwardReference.dk M s a t g)
          (memory M s f scale bh iv initial n)))

theorem readonly (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (initial : State α)
    (n : Nat) (r : RegionName) (off : Nat) (hk : r≠"dk") (hv : r≠"dv") (hb : r≠"dbeta") :
    (memory M s f scale bh iv initial n).mem r off=initial.mem r off := by
  induction n with
  | zero => rfl
  | succ n ih =>
    unfold memory
    rw [beta_frame M s f _ _ _ _ _ r off hb,
      KernelMemory.rowStore_frame M s "dv" _ _ _ _ _ _ r off (Or.inl hv),
      key_frame M s "dk" _ _ _ _ r off (Or.inl hk),ih]

theorem input_current (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial u : State α) (n : Nat) (same : u.mem=(memory M s f scale bh iv initial n).mem) :
    KernelBackwardReference.input M s f bh iv u=KernelBackwardReference.input M s f bh iv initial := by
  simp only [KernelBackwardReference.input,KernelReference.read,same,
    readonly M s f scale bh iv initial n "k" _ (by decide) (by decide) (by decide),
    readonly M s f scale bh iv initial n "q" _ (by decide) (by decide) (by decide),
    readonly M s f scale bh iv initial n "v" _ (by decide) (by decide) (by decide),
    readonly M s f scale bh iv initial n "beta" _ (by decide) (by decide) (by decide),
    readonly M s f scale bh iv initial n "do" _ (by decide) (by decide) (by decide),
    readonly M s f scale bh iv initial n "h0" _ (by decide) (by decide) (by decide),
    readonly M s f scale bh iv initial n "dht" _ (by decide) (by decide) (by decide)]

theorem step_values (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial u : State α) (n : Nat) (same : u.mem=(memory M s f scale bh iv initial n).mem) :
    let a := KernelBackwardReference.input M s f bh iv initial
    let t := s.T-1-n
    let h := carry M s f scale bh iv initial n
    let g := enhanced M s f scale bh iv initial n
    KernelBackwardValues.next M s f scale bh iv t h u=carry M s f scale bh iv initial (n+1) ∧
    KernelBackwardValues.dk M s f scale bh iv t h u=KernelBackwardReference.dk M s a t g ∧
    KernelBackwardValues.dv M s f scale bh iv t h u=KernelBackwardReference.dv M s a t g ∧
    KernelBackwardValues.dbeta M s f scale bh iv t h u=KernelBackwardReference.dbeta M s a t g := by
  have hi := input_current M s f scale bh iv initial u n same
  simp only [KernelBackwardValues.next,KernelBackwardValues.dk,KernelBackwardValues.dv,
    KernelBackwardValues.dbeta,KernelBackwardValues.enhanced,enhanced,hi,carry,
    KernelBackwardReference.reverse,and_self]

#axiomsClean readonly
#axiomsClean input_current
#axiomsClean step_values
#specNonCircular memory avoiding [FP.Structural.exec,FP.Structural.run,KernelProgram.source]
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelReverseMemory
