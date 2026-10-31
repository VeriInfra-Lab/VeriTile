import bench.tritonbench_g.chunk_gla_simple.Proofs.IOSourceSafety
import VeriTile.Triton.Memory.FlattenR
namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.RealFlat
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkGlaSimple Math Real
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false
set_option maxRecDepth 24000

private theorem fragment (p : Layout) (r : Regions) (scale : ℝ) :
  (VeriTile.Bench.Repairs.ChunkGlaSimple.Program.typed p r scale).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk,VeriTile.Bench.Repairs.ChunkGlaSimple.Program.source_body p r scale,
    VeriTile.Bench.Repairs.ChunkGlaSimple.Program.body]
  simp [VeriTile.Bench.Repairs.ChunkGlaSimple.Program.setup,
    VeriTile.Bench.Repairs.ChunkGlaSimple.Program.addresses,
    VeriTile.Bench.Repairs.ChunkGlaSimple.Program.indices,
    VeriTile.Bench.Repairs.ChunkGlaSimple.Program.times,
    VeriTile.Bench.Repairs.ChunkGlaSimple.Program.loop,
    VeriTile.Bench.Repairs.ChunkGlaSimple.Program.iteration,
    VeriTile.Bench.Repairs.ChunkGlaSimple.Program.tail,
    VeriTile.Bench.Repairs.ChunkGlaSimple.Program.store,
    VeriTile.Bench.Repairs.ChunkGlaSimple.Program.queryPtr,
    VeriTile.Bench.Repairs.ChunkGlaSimple.Program.keyPtr,
    VeriTile.Bench.Repairs.ChunkGlaSimple.Program.historyPtr,
    VeriTile.Bench.Repairs.ChunkGlaSimple.Program.valuePtr,
    VeriTile.Bench.Repairs.ChunkGlaSimple.Program.gatePtr,
    VeriTile.Bench.Repairs.ChunkGlaSimple.Program.gateLoad,
    VeriTile.Bench.Repairs.ChunkGlaSimple.Program.load2,
    VeriTile.Bench.Repairs.ChunkGlaSimple.Program.ptr2,
    VeriTile.Bench.Repairs.ChunkGlaSimple.Program.col,
    VeriTile.Bench.Repairs.ChunkGlaSimple.Program.row,
    VeriTile.Bench.Repairs.ChunkGlaSimple.Program.scaleCol,
    VeriTile.Bench.Repairs.ChunkGlaSimple.Program.scaleRow,
    VeriTile.Bench.Repairs.ChunkGlaSimple.Program.historyBase,
    VeriTile.Bench.Repairs.ChunkGlaSimple.Program.mask2,
    Real.setup,Real.tail,Real.iteration,chunkGlaLoopBody,chunkGlaFront,tripCount,
    StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk]
  all_goals simp [Op.FlattenOk.eq_def]

theorem source_executes (p : Layout) (r : Regions) (scale : ℝ)
    (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ q, q ∉ A.regions → A.extent q=0)
    (s : BlockState) (hu : s.undef=fun _ _=>0)
    (hinj : NoCollision p (launch s)) (hb : Safe.Bounds p (launch s) r A.extent) :
    ∃ sf, exec (A.flattenKernel
      (VeriTile.Bench.Repairs.ChunkGlaSimple.Program.typed p r scale).eraseDType.toAlgKernel)
      (A.flattenState s)=some sf ∧
      (∀ i : TileIndex [p.BT,p.BV], active p (launch s) i.1.val i.2.1.val →
        sf.readMem A.flat (A.addr r.output (outputAddress p (launch s) i.1.val i.2.1.val))=
          expected p (launch s) (inputs r s) scale i.1.val i.2.1.val) ∧
      BlockState.WriteWithin (Flat.footprint A p (launch s) r) (A.flattenState s) sf := by
  let source := (VeriTile.Bench.Repairs.ChunkGlaSimple.Program.typed p r scale).eraseDType.toAlgKernel
  obtain ⟨t,ran,values,frame⟩ := VeriTile.Bench.Repairs.ChunkGlaSimple.Execution.region_executes
    p r scale s hinj
  have safety := VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.IOSourceSafety.source_safe
    p r scale s A.extent hb
  have fragment : source.FlattenOk := fragment p r scale
  have he := A.execR_flatten hd hcov .triv source s safety fragment hu
  simp only [execR_triv] at he
  dsimp only [source] at he
  rw [ran,Option.map_some] at he
  let sf := A.flattenState t
  have ho (i : TileIndex [p.BT,p.BV]) (hi : active p (launch s) i.1.val i.2.1.val) :
      sf.readMem A.flat (A.addr r.output (outputAddress p (launch s) i.1.val i.2.1.val)) =
        expected p (launch s) (inputs r s) scale i.1.val i.2.1.val := by
    rw [A.flattenState_readMem hd t (r := r.output) (by
      by_contra hn
      have bad := (hb.tail.output i.1 i.2.1 hi)
      rw [hcov _ hn] at bad
      exact Nat.not_lt_zero _ bad) (o := outputAddress p (launch s) i.1.val i.2.1.val) (by
      have := hb.tail.output i.1 i.2.1 hi
      simpa [outputAddress,row,column] using this)]
    exact values i hi
  have hf := Flat.flatten_frame A p (launch s) r s t frame
  exact ⟨sf,he,ho,hf⟩
#axiomsClean source_executes
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.RealFlat
