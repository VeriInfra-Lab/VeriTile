import bench.tritonbench_g.chunk_retention_ops.Proofs.KernelNative
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelNative
open VeriTile Triton FP.Structural
open KernelBackward (Parameters PID Matrix)
open KernelGradient
namespace L
export VeriTile.Bench.Repairs.ChunkLinearAttn (rawLoad load zero cast add mul dot transpose valueBlocks)
end L
namespace St
export VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelStaging
  (fromInput weightedScore gradientDotValues)
end St
noncomputable section
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

def gradientDot : VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput.DotPrimitive Term :=
  ⟨(S.gradientDot (α:=Word)).apply⟩

theorem load_tag (m : RegionName→Nat→Word) (r : RegionName) (hr : r≠"q")
    (base rows cols sr sc ro co : Nat) (i : TileIndex [a,b]) :
    St.fromInput "q" (L.load model (input m) r base rows cols sr sc ro co i)=Bool.false := by
  unfold L.load L.rawLoad
  simp only [model,S.algebra]
  split_ifs <;> simp [St.fromInput,input,L.zero,model,S.algebra,hr,Ne.symm hr]

theorem dot_plain (left : KernelBackward.Matrix Term a k) (right : KernelBackward.Matrix Term k b)
    (i : TileIndex [a,b]) (h : ∀ e,St.fromInput "q" (left (i.1,e,()))=Bool.false) :
    L.dot gradientDot left right i=.pendingDot k (fun e=>left (i.1,e,())) (fun e=>right (e,i.2.1,())) := by
  simp [L.dot,gradientDot,S.gradientDot,St.gradientDotValues,h,ofFloat,toFloat]

theorem dot_materialized (left : KernelBackward.Matrix Term a k) (right : KernelBackward.Matrix Term k b)
    (i : TileIndex [a,b]) (hk : 0<k)
    (hl : St.fromInput "q" (left (i.1,⟨0,hk⟩,()))=Bool.true)
    (hr : St.weightedScore (right (⟨0,hk⟩,i.2.1,()))=Bool.true) :
    L.dot gradientDot left right i=.materialize (.pendingDot k (fun e=>left (i.1,e,())) (fun e=>right (e,i.2.1,()))) := by
  simp [L.dot,gradientDot,S.gradientDot,St.gradientDotValues,hk,hl,hr,ofFloat,toFloat]

theorem value_dot_plain (m : RegionName→Nat→Word) (p : Parameters) (pid : PID)
    (r : RegionName) (hr : r≠"q") (vb : Nat) (right : KernelBackward.Matrix Term p.BV b) (i : TileIndex [p.BT,b]) :
    L.dot gradientDot (valueTile model p pid (input m) r vb) right i=
      .pendingDot p.BV (fun e=>valueTile model p pid (input m) r vb (i.1,e,())) (fun e=>right (e,i.2.1,())) :=
  dot_plain _ _ i (fun e=>load_tag m r hr _ _ _ _ _ _ _ _)

theorem key_dot_plain (m : RegionName→Nat→Word) (p : Parameters) (pid : PID)
    (right : KernelBackward.Matrix Term p.BK b) (i : TileIndex [p.BT,b]) :
    L.dot gradientDot (gradKey model p pid (input m)) right i=
      .pendingDot p.BK (fun e=>gradKey model p pid (input m) (i.1,e,())) (fun e=>right (e,i.2.1,())) :=
  dot_plain _ _ i (fun e=>load_tag m "k" (by decide) _ _ _ _ _ _ _ _)

theorem score_tag (m : RegionName→Nat→Word) (p : Parameters) (pid : PID) (scale : ℝ)
    (n : Nat) (hn : 0<n) (i : TileIndex [p.BT,p.BT]) :
    St.weightedScore (gradScoreGradient model p pid scale (gradStates model gradientDot p pid (input m) scale n) i)=Bool.true := by
  cases n with
  | zero=>omega
  | succ n=>
    change St.weightedScore (.cast (.binary .mul
      (.binary .add ((gradStates model gradientDot p pid (input m) scale n).score i)
        (L.dot gradientDot (valueTile model p pid (input m) "incoming" n)
          (L.transpose (valueTile model p pid (input m) "v" n)) i)) (scoreDecay model p pid scale i)))=_
    rw [value_dot_plain m p pid "incoming" (by decide)]
    rfl

/-- The q×dS dot is the sole active transposed dK result. Its local source tags
force materialization before the final add; no result equality is assumed. -/
theorem key_result_dot (c : Shape) (pid : PID) (m : RegionName→Nat→Word) (scale : ℝ)
    (i : TileIndex [64,c.BK]) (hi : gradKeyActive (KernelReal.parameters c) pid i) :
    let p:=KernelReal.parameters c
    let ds:=gradScoreGradient model p pid scale (gradStates model gradientDot p pid (input m) scale c.valueBlocks)
    L.dot gradientDot (gradQuery model p pid (input m)) ds (i.2.1,i.1,())=
      .materialize (.pendingDot 64 (fun e=>gradQuery model p pid (input m) (i.2.1,e,()))
        (fun e=>ds (e,i.1,()))) := by
  dsimp only
  have hn : 0<c.valueBlocks := by
    have h:=c.value_cover
    have hv:=c.valuePositive
    by_contra hzero
    have hz:c.valueBlocks=0:=by omega
    change c.V≤c.valueBlocks*c.BV at h
    rw [hz] at h
    simp at h
    omega
  have htag : St.fromInput "q" (gradQuery model (KernelReal.parameters c) pid (input m) (i.2.1,⟨0,by change 0<64; decide⟩,()))=Bool.true := by
    have ht : pid.y*64<c.T:=by have hh:=hi.1; change pid.y*64+i.1.val<c.T at hh; omega
    have hk : pid.x*c.BK+i.2.1.val<c.K:=hi.2
    simp [gradQuery,L.load,L.rawLoad,L.zero,model,S.algebra,KernelReal.parameters,ht,hk,input,St.fromInput]
  have hs:=score_tag m (KernelReal.parameters c) pid scale c.valueBlocks hn (⟨0,by change 0<64; decide⟩,i.1,())
  exact dot_materialized _ _ _ (by change 0<64; decide) htag hs

#axiomsClean dot_plain
#axiomsClean score_tag
#axiomsClean key_result_dot
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelNative
