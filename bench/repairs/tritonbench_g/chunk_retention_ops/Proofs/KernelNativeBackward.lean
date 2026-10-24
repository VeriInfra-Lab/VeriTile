import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelNativeResults
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelNative
open VeriTile Triton FP.Structural
open KernelBackward (PID Carry)
open KernelReal (parameters)
open VeriTile.Bench.Repairs.ChunkLinearAttn (hActive hOffset fOffset)
open KernelPacket (Call)
noncomputable section
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

def regularDot : VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput.DotPrimitive Term := ⟨(S.dot (α:=Word)).apply⟩

theorem backward_factor (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (n : Nat) :
    S.lower (sharedOps ops) literal (KernelBackward.factor model (parameters c) pid n)=
      decay ops c pid.z (c.length (c.chunks-1-n)) := by
  change ops.exp2 (ops.mul (ops.integer (c.length (c.chunks-1-n)))
    (S.lower (sharedOps ops) literal (KernelBackward.logRate model (parameters c) pid)))=_
  rw [rate_eq ops literal laws]
  rfl

theorem backward_initial (isa : ISA) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps (wordOps isa)) literal)
    (c : Shape) (pid : PID) (m : RegionName→Nat→Word) (terminal : Bool)
    (i : TileIndex [c.BK,c.BV]) :
    S.lower (sharedOps (wordOps isa)) literal (KernelBackward.initialValue model (parameters c) pid (input m) terminal i)=
      Reference.reverseCarry (wordOps isa) c (data (wordOps isa) c m Bool.false terminal) (literal 1) pid.z 0
        (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val) := by
  cases terminal <;> simp [KernelBackward.initialValue,Reference.reverseCarry,data,input,parameters,
    L.zero,model,S.algebra,lower_ite,S.lower,masked,laws.zero,wordOps]

  all_goals rfl

theorem backward_left (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (m : RegionName→Nat→Word) (terminal : Bool) (scale : ℝ)
    (n : Nat) (i : TileIndex [c.BK,64]) :
    S.lower (sharedOps ops) literal (KernelBackward.leftValues model (parameters c) pid (input m) scale n i)=
      ops.cast (ops.mul (qAt ops c (data ops c m Bool.false terminal) pid.z
        ((c.chunks-1-n)*64+i.2.1.val) (pid.x*c.BK+i.1.val)) (literal scale)) := by
  simp [KernelBackward.leftValues,KernelBackward.L.cast,KernelBackward.L.mul,KernelBackward.L.scalar,
    L.load,L.rawLoad,L.zero,model,S.algebra,S.lower,lower_ite,parameters,input,qAt,masked,data,
    KernelBackward.time,laws.zero,and_comm,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  rfl

theorem backward_right (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (m : RegionName→Nat→Word) (terminal : Bool)
    (n : Nat) (i : TileIndex [64,c.BV]) :
    S.lower (sharedOps ops) literal (KernelBackward.rightValues model (parameters c) pid (input m) n i)=
      ops.cast (ops.mul (gAt ops c (data ops c m Bool.false terminal) pid.z
        ((c.chunks-1-n)*64+i.1.val) (pid.y*c.BV+i.2.1.val))
        (decay ops c pid.z (i.1.val+1))) := by
  change ops.cast (ops.mul (S.lower (sharedOps ops) literal
    (L.load model (input m) "incoming" (pid.z*(c.T*c.V)) c.T c.V c.V 1 ((c.chunks-1-n)*64) (pid.y*c.BV) i))
    (S.lower (sharedOps ops) literal (KernelBackward.weight model (parameters c) pid i.1.val)))=_
  rw [weight_eq ops literal laws]
  simp [L.load,L.rawLoad,L.zero,model,S.algebra,S.lower,lower_ite,input,gAt,masked,data,laws.zero]
  rfl

theorem backward_next (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (m : RegionName→Nat→Word) (terminal : Bool) (scale : ℝ) (n : Nat)
    (carry : TileIndex [c.BK,c.BV]→Term) (target : Tile Word) (i : TileIndex [c.BK,c.BV])
    (hc : S.lower (sharedOps ops) literal (carry i)=target (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val)) :
    S.lower (sharedOps ops) literal (KernelBackward.next model regularDot (parameters c) pid (input m) scale n carry i)=
      Body.nextDH ops c (data ops c m Bool.false terminal) (literal scale) pid.z (c.chunks-1-n) target
        (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val) := by
  change S.lower (sharedOps ops) literal (.binary .add
    (.binary .mul (KernelBackward.factor model (parameters c) pid n) (carry i))
    (.pendingDot 64 (fun t=>KernelBackward.leftValues model (parameters c) pid (input m) scale n (i.1,t,()))
      (fun t=>KernelBackward.rightValues model (parameters c) pid (input m) n (t,i.2.1,()))))=_
  rw [lower_live_dot]
  change chain ops 64 _ _ (ops.mul (S.lower (sharedOps ops) literal (KernelBackward.factor model (parameters c) pid n))
    (S.lower (sharedOps ops) literal (carry i)))=_
  rw [backward_factor ops literal laws,hc]
  apply chain_congr
  · intro t ht
    simp only [ht,↓reduceDIte]
    exact backward_left ops literal laws c pid m terminal scale n (i.1,⟨t,ht⟩,())
  · intro t ht
    simp only [ht,↓reduceDIte]
    exact backward_right ops literal laws c pid m terminal n (⟨t,ht⟩,i.2.1,())

theorem backward_values (isa : ISA) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps (wordOps isa)) literal)
    (c : Shape) (pid : PID) (m : RegionName→Nat→Word) (terminal : Bool) (scale : ℝ)
    (n : Nat) (i : TileIndex [c.BK,c.BV]) :
    S.lower (sharedOps (wordOps isa)) literal (KernelBackward.values model regularDot (parameters c) pid (input m) scale terminal n i)=
      Reference.reverseCarry (wordOps isa) c (data (wordOps isa) c m Bool.false terminal) (literal scale) pid.z n
        (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val) := by
  induction n with
  | zero=>exact backward_initial isa literal laws c pid m terminal i
  | succ n ih=>
    exact backward_next (wordOps isa) literal laws c pid m terminal scale n _ _ i ih

/-- The native reverse kernel reaches the independent scalar reverse recurrence,
including the optional initial-state gradient, with all source casts retained. -/
theorem backward_target (isa : ISA) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps (wordOps isa)) literal)
    (c : Shape) (scale : ℝ) (terminal initial : Bool) (pid : PID) (m : RegionName→Nat→Word) :
    let a : Call:=⟨c,.backwardState,scale,terminal,initial,pid⟩
    let xs:=data (wordOps isa) c m Bool.false terminal
    (∀ n,n<c.chunks → ∀ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i →
      target isa literal a m "dhistory" (hOffset (parameters c) pid (KernelBackward.time (parameters c) n) i)=
        (wordOps isa).cast (Reference.reverseCarry (wordOps isa) c xs (literal scale) pid.z n
          (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val))) ∧
    (initial=Bool.true → ∀ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i →
      target isa literal a m "initialGradient" (fOffset (parameters c) pid i)=
        (wordOps isa).cast (Reference.reverseCarry (wordOps isa) c xs (literal scale) pid.z c.chunks
          (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val))) := by
  dsimp only
  constructor
  · intro n hn i hi
    change S.lower (sharedOps (wordOps isa)) literal
      ((KernelBackward.memory model regularDot (parameters c) pid (input m) (input m) scale terminal initial
        "dhistory" (hOffset (parameters c) pid (KernelBackward.time (parameters c) n) i)).read .real)=_
    rw [KernelBackward.state_read_history model regularDot (parameters c) pid (input m) (input m) scale terminal initial n i hn hi]
    exact backward_values isa literal laws c pid m terminal scale n i
  · intro hf i hi
    subst initial
    change S.lower (sharedOps (wordOps isa)) literal
      ((KernelBackward.memory model regularDot (parameters c) pid (input m) (input m) scale terminal Bool.true
        "initialGradient" (fOffset (parameters c) pid i)).read .real)=_
    rw [KernelBackward.state_read_final model regularDot (parameters c) pid (input m) (input m) scale terminal i hi]
    exact backward_values isa literal laws c pid m terminal scale c.chunks i

#axiomsClean backward_next
#axiomsClean backward_values
#axiomsClean backward_target
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelNative
