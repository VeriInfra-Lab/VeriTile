import bench.tritonbench_g.chunk_retention_ops.Proofs.IOReadLocality

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
open VeriTile Triton FP.Structural KernelPacket
open KernelReal (parameters)
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open KernelBackward (PID Matrix)
open VeriTile.Bench.Repairs.ChunkLinearAttn (load rawLoad)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 1500000
set_option maxRecDepth 30000

private theorem load_congr (M : Algebra α) (m n : KernelPacket.Memory α) (r : RegionName)
    (base rows cols sr sc ro co br bc : Nat)
    (h : ∀ row col,row<rows → col<cols →
      (m r (base+row*sr+col*sc)).read .real=(n r (base+row*sr+col*sc)).read .real) :
    load M m r base rows cols sr sc ro co (br:=br) (bc:=bc)=
      load M n r base rows cols sr sc ro co := by
  funext i
  unfold load rawLoad
  congr 1
  split_ifs with active
  · exact h _ _ active.1 active.2
  · rfl

theorem backward_values_readEq (M : Algebra α) (D : DotPrimitive α) (a : Call)
    (stage : a.stage=.backwardState) (m n : KernelPacket.Memory α) (reads : ReadEq a m n) (count : Nat) :
    KernelBackward.values M D (parameters a.shape) a.pid m a.scale a.inputState count=
      KernelBackward.values M D (parameters a.shape) a.pid n a.scale a.inputState count := by
  have initial : KernelBackward.initialValue M (parameters a.shape) a.pid m a.inputState=
      KernelBackward.initialValue M (parameters a.shape) a.pid n a.inputState := by
    funext i
    unfold KernelBackward.initialValue
    split_ifs with seed active
    · congr 1
      exact reads ⟨5,by simp [inputs]⟩ (⟨_,active.1⟩,⟨_,active.2⟩,()) ⟨stage,seed⟩
    · rfl
    · rfl
  have left (block : Nat) : KernelBackward.leftValues M (parameters a.shape) a.pid m a.scale block=
      KernelBackward.leftValues M (parameters a.shape) a.pid n a.scale block := by
    have loads := load_congr M m n "q" (a.pid.z*(a.shape.T*a.shape.K)) a.shape.K a.shape.T 1 a.shape.K
      (a.pid.x*a.shape.BK) (KernelBackward.time (parameters a.shape) block*64) a.shape.BK 64 (by
        intro e t he ht
        simpa [inputOffset,inputs,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using
          reads ⟨0,by simp [inputs]⟩ (⟨t,ht⟩,⟨e,he⟩,()) (Or.inr (Or.inl stage)))
    funext i
    unfold KernelBackward.leftValues
    dsimp only [parameters] at loads ⊢
    rw [loads]
  have right (block : Nat) : KernelBackward.rightValues M (parameters a.shape) a.pid m block=
      KernelBackward.rightValues M (parameters a.shape) a.pid n block := by
    have loads := load_congr M m n "incoming" (a.pid.z*(a.shape.T*a.shape.V)) a.shape.T a.shape.V a.shape.V 1
      (KernelBackward.time (parameters a.shape) block*64) (a.pid.y*a.shape.BV) 64 a.shape.BV (by
        intro t j ht hj
        simpa [inputOffset,inputs] using
          reads ⟨3,by simp [inputs]⟩ (⟨t,ht⟩,⟨j,hj⟩,()) (Or.inl stage))
    funext i
    unfold KernelBackward.rightValues
    dsimp only [parameters] at loads ⊢
    rw [loads]
  induction count with
  | zero => exact initial
  | succ count ih =>
    unfold KernelBackward.values
    rw [ih]
    funext i
    simp only [KernelBackward.next,left,right]

structure GradientLoads (M : Algebra α) (a : Call) (m n : KernelPacket.Memory α) : Prop where
  query : KernelGradient.gradQuery M (parameters a.shape) a.pid m=KernelGradient.gradQuery M (parameters a.shape) a.pid n
  key : KernelGradient.gradKey M (parameters a.shape) a.pid m=KernelGradient.gradKey M (parameters a.shape) a.pid n
  value : ∀ vb,KernelGradient.valueTile M (parameters a.shape) a.pid m "v" vb=
    KernelGradient.valueTile M (parameters a.shape) a.pid n "v" vb
  incoming : ∀ vb,KernelGradient.valueTile M (parameters a.shape) a.pid m "incoming" vb=
    KernelGradient.valueTile M (parameters a.shape) a.pid n "incoming" vb
  history : ∀ vb,KernelGradient.gradHistory M (parameters a.shape) a.pid m vb=
    KernelGradient.gradHistory M (parameters a.shape) a.pid n vb
  dhistory : ∀ vb,KernelGradient.gradHistoryGradient M (parameters a.shape) a.pid m vb=
    KernelGradient.gradHistoryGradient M (parameters a.shape) a.pid n vb

theorem gradient_loads_readEq (M : Algebra α) (a : Call) (stage : a.stage=.backwardGradient)
    (m n : KernelPacket.Memory α) (reads : ReadEq a m n) : GradientLoads M a m n := by
  constructor
  · apply load_congr
    intro e t he ht
    simpa [inputOffset,inputs,parameters,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using
      reads ⟨0,by simp [inputs]⟩ (⟨t,ht⟩,⟨e,he⟩,()) (Or.inr (Or.inr stage))
  · apply load_congr
    intro t e ht he
    simpa [inputOffset,inputs,parameters] using
      reads ⟨1,by simp [inputs]⟩ (⟨t,ht⟩,⟨e,he⟩,()) (Or.inr (Or.inr stage))
  · intro vb
    apply load_congr
    intro t j ht hj
    simpa [inputOffset,inputs,parameters] using
      reads ⟨2,by simp [inputs]⟩ (⟨t,ht⟩,⟨j,hj⟩,()) (Or.inr (Or.inr stage))
  · intro vb
    apply load_congr
    intro t j ht hj
    simpa [inputOffset,inputs,parameters] using
      reads ⟨3,by simp [inputs]⟩ (⟨t,ht⟩,⟨j,hj⟩,()) (Or.inr stage)
  · intro vb
    apply load_congr
    intro j e hj he
    simpa [inputOffset,inputs,parameters,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using
      reads ⟨6,by simp [inputs]⟩ (⟨e,he⟩,⟨j,hj⟩,()) (Or.inr stage)
  · intro vb
    apply load_congr
    intro e j he hj
    simpa [inputOffset,inputs,parameters] using
      reads ⟨7,by simp [inputs]⟩ (⟨e,he⟩,⟨j,hj⟩,()) stage

theorem GradientLoads.states {M : Algebra α} {a : Call} {m n : KernelPacket.Memory α}
    (reads : GradientLoads M a m n) (D : DotPrimitive α) (count : Nat) :
    KernelGradient.gradStates M D (parameters a.shape) a.pid m a.scale count=
      KernelGradient.gradStates M D (parameters a.shape) a.pid n a.scale count := by
  induction count with
  | zero => rfl
  | succ count ih => simp only [KernelGradient.gradStates,KernelGradient.gradNext,ih,
      reads.value,reads.incoming,reads.history,reads.dhistory]

theorem GradientLoads.valueResult {M : Algebra α} {a : Call} {m n : KernelPacket.Memory α}
    (reads : GradientLoads M a m n) (D : DotPrimitive α) (vb : Nat) :
    KernelGradient.gradValue M D (parameters a.shape) a.pid m a.scale vb=
      KernelGradient.gradValue M D (parameters a.shape) a.pid n a.scale vb := by
  funext i
  simp only [KernelGradient.gradValue,KernelGradient.gradScore,reads.query,reads.key,reads.dhistory,reads.incoming]

theorem GradientLoads.queryResult {M : Algebra α} {a : Call} {m n : KernelPacket.Memory α}
    (reads : GradientLoads M a m n) (D : DotPrimitive α) (count : Nat) :
    KernelGradient.gradQueryResult M D (parameters a.shape) a.pid m a.scale
      (KernelGradient.gradStates M D (parameters a.shape) a.pid m a.scale count)=
    KernelGradient.gradQueryResult M D (parameters a.shape) a.pid n a.scale
      (KernelGradient.gradStates M D (parameters a.shape) a.pid n a.scale count) := by
  funext i
  simp only [KernelGradient.gradQueryResult,reads.states D,reads.key]

theorem GradientLoads.keyResult {M : Algebra α} {a : Call} {m n : KernelPacket.Memory α}
    (reads : GradientLoads M a m n) (D : DotPrimitive α) (count : Nat) :
    KernelGradient.gradKeyResult M D (parameters a.shape) a.pid m a.scale
      (KernelGradient.gradStates M D (parameters a.shape) a.pid m a.scale count)=
    KernelGradient.gradKeyResult M D (parameters a.shape) a.pid n a.scale
      (KernelGradient.gradStates M D (parameters a.shape) a.pid n a.scale count) := by
  funext i
  simp only [KernelGradient.gradKeyResult,reads.states D,reads.query]

#axiomsClean backward_values_readEq
#axiomsClean gradient_loads_readEq
#axiomsClean GradientLoads.valueResult
#axiomsClean GradientLoads.queryResult
#axiomsClean GradientLoads.keyResult
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
