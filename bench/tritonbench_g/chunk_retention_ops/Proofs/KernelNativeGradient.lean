import bench.tritonbench_g.chunk_retention_ops.Proofs.KernelNativeResults
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelNative
open VeriTile Triton FP.Structural
open KernelBackward (PID)
open KernelGradient
open KernelReal (parameters)
open KernelPacket (Call)
noncomputable section

/-- Every native Q/K/partial-V output equals an independently declared scalar
FMA contraction. The dK add is separate after its transposed-dot materialization;
the dV result remains per key block and is not replaced by a final reduction. -/
theorem gradient_target (isa : ISA) (literal : ℝ→Word)
    (laws : S.LiteralLaws (sharedOps (wordOps isa)) literal)
    (c : Shape) (scale : ℝ) (pid : PID) (m : RegionName→Nat→Word) (hh : pid.z<c.B*c.H) :
    let a : Call:=⟨c,.backwardGradient,scale,Bool.false,Bool.false,pid⟩
    let xs:=data (wordOps isa) c m Bool.false Bool.false
    (∀ i : TileIndex [64,c.BK],gradKeyActive (parameters c) pid i →
      target isa literal a m "queryGradient" (gradKeyOffset (parameters c) pid i)=
        Body.dq (wordOps isa) c xs (literal scale) pid.z pid.y (history c pid m)
          i.1.val (pid.x*c.BK+i.2.1.val)) ∧
    (∀ i : TileIndex [64,c.BK],gradKeyActive (parameters c) pid i →
      target isa literal a m "keyGradient" (gradKeyOffset (parameters c) pid i)=
        Body.dk (wordOps isa) c xs (literal scale) pid.z pid.y (dhistory c pid m)
          i.1.val (pid.x*c.BK+i.2.1.val)) ∧
    (∀ vb,vb<c.valueBlocks → ∀ i : TileIndex [64,c.BV],gradValueActive (parameters c) pid vb i →
      target isa literal a m "valuePartials" (gradValueOffset (parameters c) pid (c.B*c.H) vb i)=
        Body.dvPartial (wordOps isa) c xs (literal scale) pid.x pid.z pid.y (dhistory c pid m)
          i.1.val (vb*c.BV+i.2.1.val)) := by
  dsimp only
  refine ⟨?_,?_,?_⟩
  · intro i hi
    have hr:=gradient_read_query model gradientDot (parameters c) pid (input m) (input m) scale i hi
    exact (congrArg (fun cell=>S.lower (sharedOps (wordOps isa)) literal (cell.read .real)) hr).trans
      (query_result_eq (wordOps isa) literal laws c pid m scale i)
  · intro i hi
    have hr:=gradient_read_key model gradientDot (parameters c) pid (input m) (input m) scale i hi
    exact (congrArg (fun cell=>S.lower (sharedOps (wordOps isa)) literal (cell.read .real)) hr).trans
      (key_result_eq (wordOps isa) literal laws c pid m scale i hi)
  · intro vb hv i hi
    have hr:=gradient_read_value model gradientDot (parameters c) pid (input m) (input m) scale vb i hh hv hi
    exact (congrArg (fun cell=>S.lower (sharedOps (wordOps isa)) literal (cell.read .real)) hr).trans
      (value_result_eq (wordOps isa) literal laws c pid m scale vb i)

#axiomsClean gradient_target
#specNonCircular data avoiding [run,FP.Structural.exec,target,gradient_target]
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelNative
