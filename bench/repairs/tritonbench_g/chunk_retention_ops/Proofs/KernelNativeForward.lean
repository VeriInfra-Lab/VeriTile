import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelNative
import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardHNative
import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardONative
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelNative
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open KernelBackward (PID)
open KernelReal (parameters)
open KernelPacket (Call)
open VeriTile.Bench.Repairs.ChunkLinearAttn (hActive hOffset fOffset outputActive outputOffset)
noncomputable section
set_option maxHeartbeats 6000000

/-- Reuse the sibling's separately proved scalar recurrence. Equality of the
complete forward quotation is definitional, so no compiler-result premise is
introduced by the reuse. -/
theorem forward_target (isa : ISA) (literal : ℝ→Word)
    (laws : S.LiteralLaws (sharedOps (wordOps isa)) literal) (c : Shape)
    (seed final : Bool) (pid : PID) (m : RegionName→Nat→Word) :
    let a : Call:=⟨c,.forwardState,0,seed,final,pid⟩
    let s:=KernelForward.state c pid (input m)
    let ops:=sharedOps (wordOps isa)
    let xs:=VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.H.xs ops literal (KernelOutput.shape c) seed s
    (∀ n,n<c.chunks → ∀ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i →
      target isa literal a m "h" (hOffset (parameters c) pid n i)=
        ops.cast (VeriTile.Bench.Repairs.ChunkRetention.Whole001.Reference.history ops (KernelOutput.shape c) xs pid.z n
          (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val))) ∧
    (final=Bool.true → ∀ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i →
      target isa literal a m "ht" (fOffset (parameters c) pid i)=
        VeriTile.Bench.Repairs.ChunkRetention.Whole001.Reference.finalState ops (KernelOutput.shape c) xs pid.z
          (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val)) := by
  dsimp only
  let a : Call:=⟨c,.forwardState,0,seed,final,pid⟩
  obtain ⟨t,ht,hm⟩:=KernelPacket.source model (dotModel a) a (input m)
  obtain ⟨u,hu,hh,hf,_⟩:=VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.H.source_result
    (sharedOps (wordOps isa)) literal laws rfl (KernelOutput.shape c) seed final (a.state (input m))
  have hu' : FP.Structural.exec (engine (VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput.withDot model (dotModel a)))
      a.kernel (a.state (input m))=some u:=hu
  have eq : t=u:=Option.some.inj (ht.symm.trans hu')
  subst u
  constructor
  · intro n hn i hi
    change S.lower (sharedOps (wordOps isa)) literal
      (((KernelPacket.reference model (dotModel a) a (input m)) "h" (hOffset (parameters c) pid n i)).read .real)=_
    rw [←hm]
    exact hh n i hn hi
  · intro flag i hi
    change S.lower (sharedOps (wordOps isa)) literal
      (((KernelPacket.reference model (dotModel a) a (input m)) "ht" (fOffset (parameters c) pid i)).read .real)=_
    rw [←hm]
    exact hf flag i hi

/-- The native forward-O target is the independent scalar FMA chunk contraction
with the genuine history input. -/
theorem output_target (isa : ISA) (literal : ℝ→Word)
    (laws : S.LiteralLaws (sharedOps (wordOps isa)) literal) (c : Shape)
    (scale : ℝ) (pid : PID) (m : RegionName→Nat→Word)
    (i : TileIndex [64,c.BV]) (hi : outputActive (parameters c) pid i) :
    let a : Call:=⟨c,.forwardOutput,scale,Bool.false,Bool.false,pid⟩
    let s:=KernelOutput.state c pid (input m)
    let ops:=sharedOps (wordOps isa)
    target isa literal a m "o" (outputOffset (parameters c) pid i)=
      VeriTile.Bench.Repairs.ChunkRetention.Whole001.Body.output ops (KernelOutput.shape c)
        (VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.O.xs ops literal (KernelOutput.shape c) s)
        (literal scale) pid.z pid.y
        (VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.O.history ops literal (KernelOutput.shape c) s)
        i.1.val (pid.x*c.BV+i.2.1.val) := by
  dsimp only
  let a : Call:=⟨c,.forwardOutput,scale,Bool.false,Bool.false,pid⟩
  obtain ⟨t,ht,hm⟩:=KernelPacket.source model (dotModel a) a (input m)
  obtain ⟨u,hu,hv,_⟩:=VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.O.source_result
    (sharedOps (wordOps isa)) literal laws (KernelOutput.shape c) (a.state (input m)) scale
  have hu' : FP.Structural.exec (engine (VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput.withDot model (dotModel a)))
      a.kernel (a.state (input m))=some u:=hu
  have eq : t=u:=Option.some.inj (ht.symm.trans hu')
  subst u
  change S.lower (sharedOps (wordOps isa)) literal
    (((KernelPacket.reference model (dotModel a) a (input m)) "o" (outputOffset (parameters c) pid i)).read .real)=_
  rw [←hm]
  exact hv i hi

#axiomsClean forward_target
#axiomsClean output_target
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelNative
