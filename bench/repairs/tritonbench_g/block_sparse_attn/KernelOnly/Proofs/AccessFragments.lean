import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Proofs.LocalExecution
import bench.repairs.tritonbench_g.chunk_retention.Proofs.PureAccessSafety
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile Triton FP.Structural
open ChunkRetention.Whole001
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs Bool.false

/-- Coordinates of the actual source loads and stores. This proof-only record
will be derived from public finite IO windows, not assumed by a specification. -/
structure AccessBounds (p : Parameters) (s : State (Replicas α)) (bounds : RegionName→Nat) : Prop where
  rowFirst : rowPointer p s<bounds "rows"
  rowLast : rowPointer p s+1<bounds "rows"
  column : ∀ slot,first p s≤slot → slot<last p s →
    ((layout p s*p.colStride : Nat)+slot : Int).toNat<bounds "cols"
  query : ∀ slot,first p s≤slot → slot<last p s → ∀ block,block<p.blocks → ∀ i : TileIndex [16,16],
    (p.evenM=Bool.true ∨ row s i.1<p.length) → qPointer p s i+block*16*p.qd<bounds "Q"
  key : ∀ slot,first p s≤slot → slot<last p s → ∀ block,block<p.blocks →
    ∀ i : TileIndex [16,16],
    (p.evenN=Bool.true ∨ (i.2.1.val : Int)+csrColumn p s slot*16<p.length) →
    kPointer p s i+(csrColumn p s slot*16*p.kn).toNat+block*16*p.kd<bounds "K"
  value : ∀ slot,first p s≤slot → slot<last p s → ∀ i : TileIndex [16,16],
    (p.evenN=Bool.true ∨ (i.1.val : Int)+csrColumn p s slot*16<p.length) →
    vPointer p s i+(csrColumn p s slot*16*p.vn).toNat<bounds "V"
  output : ∀ i : TileIndex [16,16],row s i.1<p.length → outputOffset p s i<bounds "out"

namespace SourceSafety

theorem prefix_safe (a : Arithmetic α) (p : Parameters) (s : State (Replicas α))
    (bounds : RegionName→Nat) (ints : RowInt32 p s) (backed : AccessBounds p s bounds) :
    Safety.traceSafeList a.model bounds (prelude p) (bindScale (broadcast (a.argument p.scale)) s) := by
  have rows0 := ints.1
  have rows1 := ints.2
  dsimp only [first,last,rowPointer,layout,head] at rows0 rows1
  have h0 := backed.rowFirst
  have h1 := backed.rowLast
  dsimp only [rowPointer,layout,head] at h0 h1
  simp [prelude,source,declared,ComputeKernel.surfaceBody,bindScale,
    AccessSafety.singleton_safe,AccessSafety.pair_safe,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,numeric,bop,natLt,numericLt,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,Option.bind,
    integralFloorDiv,integralMod,Int32Bits.valuesValid,rows0,rows1,h0,h1]

theorem visit_prefix_safe (a : Arithmetic α) (p : Parameters)
    (initial current : State (Replicas α)) (carry : Carry α) (slot : Int)
    (h : Runtime a p initial carry current) (ints : ColumnInt32 p initial slot)
    (bounds : RegionName→Nat) (backed : AccessBounds p initial bounds)
    (lo : first p initial≤slot) (hi : slot<last p initial) :
    Safety.traceSafeList a.model bounds ((body p).take 3)
      (current.setReg "col_idx_idx" .int [] (fun _=>slot)) := by
  have cols := ints
  dsimp only [ColumnInt32,csrColumn] at cols
  have bound := backed.column slot lo hi
  simp [body,source,declared,ComputeKernel.surfaceBody,
    AccessSafety.singleton_safe,AccessSafety.pair_safe,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,numeric,bop,natLt,numericLt,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,Option.bind,
    integralFloorDiv,integralMod,Arithmetic.model,Int32Bits.valuesValid,
    h.scale,h.memory,h.layout,cols,bound,Broadcast.leftIndex,Broadcast.rightIndex]
  simpa only [Int.natCast_mul] using bound

theorem channel_safe (a : Arithmetic α) (native : Bool) (p : Parameters)
    (initial current : State (Replicas α)) (carry : Carry α) (slot : Int)
    (scores : Matrix α) (block : Nat)
    (h : ChannelInvariant a p initial carry (csrColumn p initial slot) scores current)
    (bounds : RegionName→Nat) (backed : AccessBounds p initial bounds)
    (lo : first p initial≤slot) (hi : slot<last p initial) (hb : block<p.blocks) :
    Safety.traceSafeList a.model bounds (channelBody native p)
      (current.setReg "score_block" .nat [] (fun _=>block)) := by
  have hq := backed.query slot lo hi block hb
  have hk := backed.key slot lo hi block hb
  cases native <;> cases em : p.evenM <;> cases en : p.evenN
  all_goals
    simp [channelBody,compiledScoreBody,scoreBody,body,source,declared,em,en,
      carryNode,fp,ref2,ComputeKernel.surfaceBody,
      AccessSafety.singleton_safe,AccessSafety.pair_safe,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,numeric,bop,natLt,numericLt,
      ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,Option.bind,
      integralFloorDiv,integralMod,Arithmetic.model,
      h.scale,h.memory,h.qptrs,h.kptrs,h.rows,h.keys,h.length,h.start,h.scores,
      Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    constructor
    · intro i j unit
      cases unit
      try intro active
      exact hq (i,j,PUnit.unit) (by simp_all)
    · intro i j unit
      cases unit
      try intro active
      simpa only [Int.natCast_mul] using hk (i,j,PUnit.unit) (by simp_all)

end SourceSafety
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
