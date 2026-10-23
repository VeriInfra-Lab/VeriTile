import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelNativeGradientModel
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelNative
open VeriTile Triton FP.Structural
open KernelBackward (PID)
open KernelGradient
open KernelReal (parameters)
noncomputable section
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

/-- Scalar input snapshot; every field is read only from an input region. -/
def data (ops : Ops Word) (c : Shape) (m : RegionName→Nat→Word) (seed terminal : Bool) : Inputs Word where
  q:=fun h t e=>m "q" (h*(c.T*c.K)+t*c.K+e)
  k:=fun h t e=>m "k" (h*(c.T*c.K)+t*c.K+e)
  v:=fun h t j=>m "v" (h*(c.T*c.V)+t*c.V+j)
  seed:=fun h e j=>if seed then m "h0" (h*c.K*c.V+e*c.V+j) else ops.zero
  incoming:=fun h t j=>m "incoming" (h*(c.T*c.V)+t*c.V+j)
  finalIncoming:=fun h e j=>if terminal then m "terminal" (h*c.K*c.V+e*c.V+j) else ops.zero

def history (c : Shape) (pid : PID) (m : RegionName→Nat→Word) : Tile Word :=
  fun e j=>m "h" (pid.z*(c.chunks*c.K*c.V)+pid.y*c.K*c.V+e*c.V+j)
def dhistory (c : Shape) (pid : PID) (m : RegionName→Nat→Word) : Tile Word :=
  fun e j=>m "dhistory" (pid.z*(c.chunks*c.K*c.V)+pid.y*c.K*c.V+e*c.V+j)

theorem lower_ite (ops : Ops Word) (literal : ℝ→Word) (p : Prop) [Decidable p] (a b : Term) :
    S.lower (sharedOps ops) literal (if p then a else b)=
      if p then S.lower (sharedOps ops) literal a else S.lower (sharedOps ops) literal b := by split <;> rfl

theorem query_load (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (m : RegionName→Nat→Word) (i : TileIndex [c.BK,64]) :
    S.lower (sharedOps ops) literal (gradQuery model (parameters c) pid (input m) i)=
      qAt ops c (data ops c m Bool.false Bool.false) pid.z (pid.y*64+i.2.1.val) (pid.x*c.BK+i.1.val) := by
  simp [gradQuery,L.load,L.rawLoad,L.zero,model,S.algebra,S.lower,lower_ite,parameters,input,qAt,masked,data,
    laws.zero,and_comm,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

  all_goals rfl

theorem key_load (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (m : RegionName→Nat→Word) (i : TileIndex [64,c.BK]) :
    S.lower (sharedOps ops) literal (gradKey model (parameters c) pid (input m) i)=
      kAt ops c (data ops c m Bool.false Bool.false) pid.z (pid.y*64+i.1.val) (pid.x*c.BK+i.2.1.val) := by
  simp [gradKey,L.load,L.rawLoad,L.zero,model,S.algebra,S.lower,lower_ite,parameters,input,kAt,masked,data,
    laws.zero]

  all_goals rfl

theorem value_load (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (m : RegionName→Nat→Word) (vb : Nat) (i : TileIndex [64,c.BV]) :
    S.lower (sharedOps ops) literal (valueTile model (parameters c) pid (input m) "v" vb i)=
      vAt ops c (data ops c m Bool.false Bool.false) pid.z (pid.y*64+i.1.val) (vb*c.BV+i.2.1.val) := by
  simp [valueTile,L.load,L.rawLoad,L.zero,model,S.algebra,S.lower,lower_ite,parameters,input,vAt,masked,data,
    laws.zero]

  all_goals rfl

theorem incoming_load (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (m : RegionName→Nat→Word) (vb : Nat) (i : TileIndex [64,c.BV]) :
    S.lower (sharedOps ops) literal (valueTile model (parameters c) pid (input m) "incoming" vb i)=
      gAt ops c (data ops c m Bool.false Bool.false) pid.z (pid.y*64+i.1.val) (vb*c.BV+i.2.1.val) := by
  simp [valueTile,L.load,L.rawLoad,L.zero,model,S.algebra,S.lower,lower_ite,parameters,input,gAt,masked,data,
    laws.zero]

  all_goals rfl

theorem history_load (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (m : RegionName→Nat→Word) (vb : Nat) (i : TileIndex [c.BV,c.BK]) :
    S.lower (sharedOps ops) literal (gradHistory model (parameters c) pid (input m) vb i)=
      masked ops c.K c.V (history c pid m) (pid.x*c.BK+i.2.1.val) (vb*c.BV+i.1.val) := by
  simp [gradHistory,L.load,L.rawLoad,L.zero,model,S.algebra,S.lower,lower_ite,parameters,input,masked,history,
    laws.zero,and_comm,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

  all_goals rfl

theorem dhistory_load (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (m : RegionName→Nat→Word) (vb : Nat) (i : TileIndex [c.BK,c.BV]) :
    S.lower (sharedOps ops) literal (gradHistoryGradient model (parameters c) pid (input m) vb i)=
      masked ops c.K c.V (dhistory c pid m) (pid.x*c.BK+i.1.val) (vb*c.BV+i.2.1.val) := by
  simp [gradHistoryGradient,L.load,L.rawLoad,L.zero,model,S.algebra,S.lower,lower_ite,parameters,input,masked,dhistory,
    laws.zero]

  all_goals rfl

theorem rate_eq (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) :
    S.lower (sharedOps ops) literal (KernelBackward.logRate model (parameters c) pid)=logRate ops c pid.z :=
  lower_rate ops literal laws c pid

theorem weight_eq (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (i : Nat) :
    S.lower (sharedOps ops) literal (KernelBackward.weight model (parameters c) pid i)=
      decay ops c pid.z ((i:Int)+1) := by
  change ops.exp2 (ops.mul (ops.integer ((i:Int)+1))
    (S.lower (sharedOps ops) literal (KernelBackward.logRate model (parameters c) pid)))=_
  rw [rate_eq ops literal laws]
  rfl

theorem key_decay (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (i : Nat) :
    S.lower (sharedOps ops) literal (keyDecay model (parameters c) pid i)=
      decay ops c pid.z ((c.length pid.y:Int)-i-1) := by
  change ops.exp2 (ops.mul (ops.integer ((c.length pid.y:Int)-i-1))
    (S.lower (sharedOps ops) literal (KernelBackward.logRate model (parameters c) pid)))=_
  rw [rate_eq ops literal laws]
  rfl

theorem query_decay (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (scale : ℝ) (i : Nat) :
    S.lower (sharedOps ops) literal (queryDecay model (parameters c) pid scale i)=
      ops.cast (ops.mul (decay ops c pid.z ((i:Int)+1)) (literal scale)) := by
  change ops.cast (ops.mul (S.lower (sharedOps ops) literal (KernelBackward.weight model (parameters c) pid i)) (literal scale))=_
  rw [weight_eq ops literal laws]

theorem score_decay (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (scale : ℝ) (i : TileIndex [64,64]) :
    S.lower (sharedOps ops) literal (scoreDecay model (parameters c) pid scale i)=
      ops.mul (causalWeight ops c pid.z i.1.val i.2.1.val) (literal scale) := by
  unfold scoreDecay causalWeight
  split_ifs with h
  · change ops.mul (ops.exp2 (ops.mul (ops.integer ((i.1.val:Int)-i.2.1.val))
      (S.lower (sharedOps ops) literal (KernelBackward.logRate model (parameters c) pid)))) (literal scale)=_
    rw [rate_eq ops literal laws]
    rfl
  · change ops.mul (literal 0) (literal scale)=_
    rw [laws.zero]
    rfl

#axiomsClean query_load
#axiomsClean history_load
#axiomsClean key_decay
#axiomsClean score_decay
#specNonCircular data avoiding [FP.Structural.exec,KernelPacket.reference]
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelNative
