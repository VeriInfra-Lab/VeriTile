import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelRecomputeMemory
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelRecomputeReferenceMemory
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters KernelBackwardRegisters KernelBackwardMemory
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 14000000
variable {α : Type} [Inhabited α]

def carry (M : Algebra α) (s : Shape) (f : Flags) (bh iv : Nat) (initial : State α) (n : Nat) :=
  KernelBackwardReference.forward M s (KernelBackwardReference.input M s f bh iv initial) n
def corrected (M : Algebra α) (s : Shape) (f : Flags) (bh iv : Nat) (initial : State α) (n : Nat) :=
  KernelRecomputeValues.dk M s bh iv n (carry M s f bh iv initial n) initial
def query (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (initial : State α) (n : Nat) :=
  KernelBackwardReference.dq M s scale (KernelBackwardReference.input M s f bh iv initial) n
    (carry M s f bh iv initial (n+1))
def memory (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (initial : State α) : Nat → State α
  | 0 => initial
  | n+1 =>
    keyStore M s "dq" (plane s bh iv) n (query M s f scale bh iv initial n)
      (keyStore M s "dk" (plane s bh iv) n (corrected M s f bh iv initial n)
        (memory M s f scale bh iv initial n))

theorem readonly (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (initial : State α)
    (n : Nat) (r : RegionName) (off : Nat) (hk : r≠"dk") (hq : r≠"dq") :
    (memory M s f scale bh iv initial n).mem r off=initial.mem r off := by
  induction n with
  | zero => rfl
  | succ n ih =>
    unfold memory
    rw [key_frame M s "dq" _ _ _ _ r off (Or.inl hq),key_frame M s "dk" _ _ _ _ r off (Or.inl hk),ih]

theorem key_other_time (M : Algebra α) (s : Shape) (r : RegionName) (bh t n : Nat)
    (values : Fin s.BK → α) (u : State α) (i : Fin s.BK)
    (hi : i.val<s.K) (different : n≠t) :
    (keyStore M s r bh t values u).mem r (keyAddress s bh n i.val)=u.mem r (keyAddress s bh n i.val) := by
  apply key_frame
  right
  intro j hj same
  have equal : n*s.K+i.val=t*s.K+j.val := by
    dsimp only [keyAddress] at same
    omega
  exact different (Geometry.mixed_injective s.K n t _ _ hi hj equal).1

theorem future (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (initial : State α)
    (n t : Nat) (future : n≤t) (i : Fin s.BK) (active : i.val<s.K) :
    (memory M s f scale bh iv initial n).mem "dk" (keyAddress s (plane s bh iv) t i.val)=
      initial.mem "dk" (keyAddress s (plane s bh iv) t i.val) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    unfold memory
    rw [key_frame M s "dq" _ _ _ _ "dk" _ (Or.inl (by decide)),
      key_other_time M s "dk" (plane s bh iv) n t _ _ i active (by omega)]
    exact ih (by omega)

theorem input_current (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial u : State α) (n : Nat) (same : u.mem=(memory M s f scale bh iv initial n).mem) :
    KernelBackwardReference.input M s f bh iv u=KernelBackwardReference.input M s f bh iv initial := by
  simp only [KernelBackwardReference.input,KernelReference.read,same,
    readonly M s f scale bh iv initial n "k" _ (by decide) (by decide),
    readonly M s f scale bh iv initial n "q" _ (by decide) (by decide),
    readonly M s f scale bh iv initial n "v" _ (by decide) (by decide),
    readonly M s f scale bh iv initial n "beta" _ (by decide) (by decide),
    readonly M s f scale bh iv initial n "do" _ (by decide) (by decide),
    readonly M s f scale bh iv initial n "h0" _ (by decide) (by decide),
    readonly M s f scale bh iv initial n "dht" _ (by decide) (by decide)]

theorem dk_current (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial u : State α) (n : Nat) (same : u.mem=(memory M s f scale bh iv initial n).mem) :
    KernelRecomputeValues.dk M s bh iv n (carry M s f bh iv initial n) u=corrected M s f bh iv initial n := by
  have hk : (fun i : Fin s.BK => KernelReference.read M u "dk" (keyAddress s (plane s bh iv) n i.val)
      (decide (i.val<s.K)))=(fun i : Fin s.BK => KernelReference.read M initial "dk"
        (keyAddress s (plane s bh iv) n i.val) (decide (i.val<s.K))) := by
    funext i
    simp only [KernelReference.read,same]
    by_cases active : i.val<s.K
    · rw [future M s f scale bh iv initial n n (Nat.le_refl _) i active]
    · simp [active]
  have hv : (fun j : Fin s.BBV => KernelReference.read M u "dv" (valAddress s bh iv s.BBV n j.val)
      (decide (iv*s.BBV+j.val<s.V)))=(fun j : Fin s.BBV => KernelReference.read M initial "dv"
        (valAddress s bh iv s.BBV n j.val) (decide (iv*s.BBV+j.val<s.V))) := by
    funext j
    simp only [KernelReference.read,same,readonly M s f scale bh iv initial n "dv" _ (by decide) (by decide)]
  simp only [KernelRecomputeValues.dk,corrected,hk,hv]

theorem step_values (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial u : State α) (n : Nat) (same : u.mem=(memory M s f scale bh iv initial n).mem) :
    let h := carry M s f bh iv initial n
    KernelRecomputeValues.next M s f bh iv n h u=carry M s f bh iv initial (n+1) ∧
    KernelRecomputeValues.dk M s bh iv n h u=corrected M s f bh iv initial n ∧
    KernelRecomputeValues.dq M s f scale bh iv n h u=query M s f scale bh iv initial n := by
  have hi := input_current M s f scale bh iv initial u n same
  have next : KernelRecomputeValues.next M s f bh iv n (carry M s f bh iv initial n) u=
      carry M s f bh iv initial (n+1) := by
    simp only [KernelRecomputeValues.next,hi,carry,KernelBackwardReference.forward]
  exact ⟨next,dk_current M s f scale bh iv initial u n same,by
    simp only [KernelRecomputeValues.dq,next,hi,query]⟩

#axiomsClean readonly
#axiomsClean future
#axiomsClean input_current
#axiomsClean dk_current
#axiomsClean step_values
#specNonCircular memory avoiding [FP.Structural.exec,FP.Structural.run,KernelProgram.source]
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelRecomputeReferenceMemory
