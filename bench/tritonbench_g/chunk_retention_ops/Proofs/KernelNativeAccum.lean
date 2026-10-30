import bench.tritonbench_g.chunk_retention_ops.Proofs.KernelNativeLoads
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelNative
open VeriTile Triton FP.Structural
open KernelBackward (PID)
open KernelGradient
open KernelReal (parameters)
noncomputable section
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

theorem accumulated_query (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (m : RegionName→Nat→Word) (scale : ℝ) (n : Nat) (i : TileIndex [64,c.BK]) :
    S.lower (sharedOps ops) literal ((gradStates model gradientDot (parameters c) pid (input m) scale n).query i)=
      blockChain ops c.BV
        (fun j=>gAt ops c (data ops c m Bool.false Bool.false) pid.z (pid.y*64+i.1.val) j)
        (fun j=>masked ops c.K c.V (history c pid m) (pid.x*c.BK+i.2.1.val) j) n ops.zero := by
  induction n with
  | zero=>exact laws.zero
  | succ n ih=>
    change S.lower (sharedOps ops) literal (.binary .add
      ((gradStates model gradientDot (parameters c) pid (input m) scale n).query i)
      (L.dot gradientDot (valueTile model (parameters c) pid (input m) "incoming" n)
        (gradHistory model (parameters c) pid (input m) n) i))=_
    rw [value_dot_plain m (parameters c) pid "incoming" (by decide),lower_live_dot,ih]
    apply chain_congr
    · intro j hj
      simp only [hj,↓reduceDIte]
      exact incoming_load ops literal laws c pid m n (i.1,⟨j,hj⟩,())
    · intro j hj
      simp only [hj,↓reduceDIte]
      exact history_load ops literal laws c pid m n (⟨j,hj⟩,i.2.1,())

theorem accumulated_key (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (m : RegionName→Nat→Word) (scale : ℝ) (n : Nat) (i : TileIndex [64,c.BK]) :
    S.lower (sharedOps ops) literal ((gradStates model gradientDot (parameters c) pid (input m) scale n).key i)=
      blockChain ops c.BV
        (fun j=>vAt ops c (data ops c m Bool.false Bool.false) pid.z (pid.y*64+i.1.val) j)
        (fun j=>masked ops c.K c.V (dhistory c pid m) (pid.x*c.BK+i.2.1.val) j) n ops.zero := by
  induction n with
  | zero=>exact laws.zero
  | succ n ih=>
    change S.lower (sharedOps ops) literal (.binary .add
      ((gradStates model gradientDot (parameters c) pid (input m) scale n).key i)
      (L.dot gradientDot (valueTile model (parameters c) pid (input m) "v" n)
        (L.transpose (gradHistoryGradient model (parameters c) pid (input m) n)) i))=_
    rw [value_dot_plain m (parameters c) pid "v" (by decide),lower_live_dot,ih]
    apply chain_congr
    · intro j hj
      simp only [hj,↓reduceDIte]
      exact value_load ops literal laws c pid m n (i.1,⟨j,hj⟩,())
    · intro j hj
      simp only [hj,↓reduceDIte]
      exact dhistory_load ops literal laws c pid m n (i.2.1,⟨j,hj⟩,())

theorem accumulated_score (ops : Ops Word) (literal : ℝ→Word) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : PID) (m : RegionName→Nat→Word) (scale : ℝ) (n : Nat) (i : TileIndex [64,64]) :
    S.lower (sharedOps ops) literal ((gradStates model gradientDot (parameters c) pid (input m) scale n).score i)=
      blockChain ops c.BV
        (fun j=>gAt ops c (data ops c m Bool.false Bool.false) pid.z (pid.y*64+i.1.val) j)
        (fun j=>vAt ops c (data ops c m Bool.false Bool.false) pid.z (pid.y*64+i.2.1.val) j) n ops.zero := by
  induction n with
  | zero=>exact laws.zero
  | succ n ih=>
    change S.lower (sharedOps ops) literal (.binary .add
      ((gradStates model gradientDot (parameters c) pid (input m) scale n).score i)
      (L.dot gradientDot (valueTile model (parameters c) pid (input m) "incoming" n)
        (L.transpose (valueTile model (parameters c) pid (input m) "v" n)) i))=_
    rw [value_dot_plain m (parameters c) pid "incoming" (by decide),lower_live_dot,ih]
    apply chain_congr
    · intro j hj
      simp only [hj,↓reduceDIte]
      exact incoming_load ops literal laws c pid m n (i.1,⟨j,hj⟩,())
    · intro j hj
      simp only [hj,↓reduceDIte]
      exact value_load ops literal laws c pid m n (i.2.1,⟨j,hj⟩,())

#axiomsClean accumulated_query
#axiomsClean accumulated_key
#axiomsClean accumulated_score
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelNative
