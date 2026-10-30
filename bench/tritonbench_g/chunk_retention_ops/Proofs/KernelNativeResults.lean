import bench.tritonbench_g.chunk_retention_ops.Proofs.KernelNativeAccum
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelNative
open VeriTile Triton FP.Structural
open KernelBackward (PID)
open KernelGradient
open KernelReal (parameters)
noncomputable section
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false

theorem lower_dot (ops : Ops Word) (literal : ℝ→Word) (k : Nat) (a b : Fin k→Term) :
    S.lower (sharedOps ops) literal (.pendingDot k a b)=
      chain ops k
        (fun e=>if he : e<k then S.lower (sharedOps ops) literal (a ⟨e,he⟩) else ops.zero)
        (fun e=>if he : e<k then S.lower (sharedOps ops) literal (b ⟨e,he⟩) else ops.zero) ops.zero :=
  shared_chain ops k _ _ _

theorem score_gradient_eq (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (m : RegionName→Nat→Word) (scale : ℝ) (i : TileIndex [64,64]) :
    S.lower (sharedOps ops) literal (gradScoreGradient model (parameters c) pid scale
      (gradStates model gradientDot (parameters c) pid (input m) scale c.valueBlocks) i)=
      Body.ds ops c (data ops c m Bool.false Bool.false) (literal scale) pid.z pid.y i.1.val i.2.1.val := by
  have hm : ∀ n b,S.lower (sharedOps ops) literal (.binary .mul
      ((gradStates model gradientDot (parameters c) pid (input m) scale n).score i) b)=
      ops.mul (S.lower (sharedOps ops) literal ((gradStates model gradientDot (parameters c) pid (input m) scale n).score i))
        (S.lower (sharedOps ops) literal b) := by intro n b; cases n <;> rfl
  change ops.cast (S.lower (sharedOps ops) literal (.binary .mul _ _))=_
  rw [hm,accumulated_score ops literal laws,score_decay ops literal laws]
  rfl

theorem partial_score_eq (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (m : RegionName→Nat→Word) (scale : ℝ) (i : TileIndex [64,64]) :
    S.lower (sharedOps ops) literal (gradScore model gradientDot (parameters c) pid (input m) scale i)=
      Body.partialScore ops c (data ops c m Bool.false Bool.false) (literal scale) pid.x pid.z pid.y i.1.val i.2.1.val := by
  unfold gradScore
  rw [key_dot_plain]
  change ops.mul (S.lower (sharedOps ops) literal (.pendingDot c.BK _ _))
    (S.lower (sharedOps ops) literal (scoreDecay model (parameters c) pid scale (i.2.1,i.1,())))=_
  rw [lower_dot,score_decay ops literal laws]
  congr 1
  apply chain_congr
  · intro e he
    simp only [he,↓reduceDIte]
    exact key_load ops literal laws c pid m (i.1,⟨e,he⟩,())
  · intro e he
    simp only [he,↓reduceDIte]
    exact query_load ops literal laws c pid m (⟨e,he⟩,i.2.1,())

theorem query_result_eq (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (m : RegionName→Nat→Word) (scale : ℝ) (i : TileIndex [64,c.BK]) :
    S.lower (sharedOps ops) literal (L.cast model (gradQueryResult model gradientDot (parameters c) pid (input m) scale
      (gradStates model gradientDot (parameters c) pid (input m) scale c.valueBlocks) i))=
      Body.dq ops c (data ops c m Bool.false Bool.false) (literal scale) pid.z pid.y (history c pid m)
        i.1.val (pid.x*c.BK+i.2.1.val) := by
  have hm : ∀ n b,S.lower (sharedOps ops) literal (L.mul model
      ((gradStates model gradientDot (parameters c) pid (input m) scale n).query i) b)=
      ops.mul (S.lower (sharedOps ops) literal ((gradStates model gradientDot (parameters c) pid (input m) scale n).query i))
        (S.lower (sharedOps ops) literal b) := by intro n b; cases n <;> rfl
  unfold gradQueryResult
  rw [dot_plain (gradScoreGradient model (parameters c) pid scale
    (gradStates model gradientDot (parameters c) pid (input m) scale c.valueBlocks))
    (gradKey model (parameters c) pid (input m)) i (fun e=>rfl)]
  change ops.cast (S.lower (sharedOps ops) literal (.binary .add _ (.pendingDot 64 _ _)))=_
  rw [lower_live_dot,hm,accumulated_query ops literal laws,query_decay ops literal laws]
  congr 1
  apply chain_congr
  · intro t ht
    simp only [ht,↓reduceDIte]
    exact score_gradient_eq ops literal laws c pid m scale (i.1,⟨t,ht⟩,())
  · intro t ht
    simp only [ht,↓reduceDIte]
    exact key_load ops literal laws c pid m (⟨t,ht⟩,i.2.1,())

theorem key_result_eq (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (m : RegionName→Nat→Word) (scale : ℝ)
    (i : TileIndex [64,c.BK]) (hi : gradKeyActive (parameters c) pid i) :
    S.lower (sharedOps ops) literal (L.cast model (gradKeyResult model gradientDot (parameters c) pid (input m) scale
      (gradStates model gradientDot (parameters c) pid (input m) scale c.valueBlocks) i))=
      Body.dk ops c (data ops c m Bool.false Bool.false) (literal scale) pid.z pid.y (dhistory c pid m)
        i.1.val (pid.x*c.BK+i.2.1.val) := by
  have hm : ∀ n b,S.lower (sharedOps ops) literal (L.mul model
      ((gradStates model gradientDot (parameters c) pid (input m) scale n).key i) b)=
      ops.mul (S.lower (sharedOps ops) literal ((gradStates model gradientDot (parameters c) pid (input m) scale n).key i))
        (S.lower (sharedOps ops) literal b) := by intro n b; cases n <;> rfl
  unfold gradKeyResult L.transpose
  rw [key_result_dot c pid m scale i hi]
  change ops.cast (S.lower (sharedOps ops) literal (.binary .add _ (.materialize (.pendingDot 64 _ _))))=_
  rw [lower_materialized,hm,accumulated_key ops literal laws,key_decay ops literal laws,lower_dot]
  congr 2
  apply chain_congr
  · intro t ht
    simp only [ht,↓reduceDIte]
    exact query_load ops literal laws c pid m (i.2.1,⟨t,ht⟩,())
  · intro t ht
    simp only [ht,↓reduceDIte]
    exact score_gradient_eq ops literal laws c pid m scale (⟨t,ht⟩,i.1,())

theorem value_carry_eq (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (m : RegionName→Nat→Word) (vb : Nat) (i : TileIndex [64,c.BV]) :
    S.lower (sharedOps ops) literal (L.dot gradientDot (gradKey model (parameters c) pid (input m))
      (gradHistoryGradient model (parameters c) pid (input m) vb) i)=
      chain ops c.BK
        (fun e=>kAt ops c (data ops c m Bool.false Bool.false) pid.z (pid.y*64+i.1.val) (pid.x*c.BK+e))
        (fun e=>masked ops c.K c.V (dhistory c pid m) (pid.x*c.BK+e) (vb*c.BV+i.2.1.val)) ops.zero := by
  rw [key_dot_plain,lower_dot]
  apply chain_congr
  · intro e he
    simp only [he,↓reduceDIte]
    exact key_load ops literal laws c pid m (i.1,⟨e,he⟩,())
  · intro e he
    simp only [he,↓reduceDIte]
    exact dhistory_load ops literal laws c pid m vb (⟨e,he⟩,i.2.1,())

theorem value_result_eq (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (m : RegionName→Nat→Word) (scale : ℝ) (vb : Nat) (i : TileIndex [64,c.BV]) :
    S.lower (sharedOps ops) literal (L.cast model (gradValue model gradientDot (parameters c) pid (input m) scale vb i))=
      Body.dvPartial ops c (data ops c m Bool.false Bool.false) (literal scale) pid.x pid.z pid.y (dhistory c pid m)
        i.1.val (vb*c.BV+i.2.1.val) := by
  unfold gradValue
  rw [dot_plain (fun j=>L.cast model (gradScore model gradientDot (parameters c) pid (input m) scale j))
    (valueTile model (parameters c) pid (input m) "incoming" vb) i (fun e=>rfl)]
  change ops.cast (S.lower (sharedOps ops) literal (.binary .add _ (.pendingDot 64 _ _)))=_
  rw [lower_live_dot]
  have hseed : S.lower (sharedOps ops) literal (L.mul model
      (L.dot gradientDot (gradKey model (parameters c) pid (input m))
        (gradHistoryGradient model (parameters c) pid (input m) vb) i)
      (keyDecay model (parameters c) pid i.1.val))=
      ops.mul (S.lower (sharedOps ops) literal (L.dot gradientDot (gradKey model (parameters c) pid (input m))
        (gradHistoryGradient model (parameters c) pid (input m) vb) i))
      (S.lower (sharedOps ops) literal (keyDecay model (parameters c) pid i.1.val)) := by
    rw [key_dot_plain]
    rfl
  rw [hseed,value_carry_eq ops literal laws,key_decay ops literal laws]
  congr 1
  apply chain_congr
  · intro t ht
    simp only [ht,↓reduceDIte]
    change ops.cast (S.lower (sharedOps ops) literal (gradScore model gradientDot (parameters c) pid (input m) scale (i.1,⟨t,ht⟩,())))=_
    rw [partial_score_eq ops literal laws]
  · intro t ht
    simp only [ht,↓reduceDIte]
    exact incoming_load ops literal laws c pid m vb (⟨t,ht⟩,i.2.1,())

#axiomsClean score_gradient_eq
#axiomsClean partial_score_eq
#axiomsClean query_result_eq
#axiomsClean key_result_eq
#axiomsClean value_result_eq
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelNative
