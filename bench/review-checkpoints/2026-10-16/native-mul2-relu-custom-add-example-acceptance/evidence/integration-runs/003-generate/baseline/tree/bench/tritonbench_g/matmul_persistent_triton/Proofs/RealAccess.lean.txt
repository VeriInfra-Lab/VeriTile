import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealLoads
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealStore
import VeriTile.Triton.Memory.TraceProofs

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealAccess
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

/-- Allocation bounds only on original logical input/output cells. No masked
tail lane is required to name a valid allocated address. -/
structure Bounds (p : Parameters) (bounds : RegionBounds) : Prop where
  a : ∀ r k, r<p.M → k<p.K → r*p.sam+k*p.sak < bounds p.A
  b : ∀ k c, k<p.K → c<p.N → k*p.sbk+c*p.sbn < bounds p.B
  c : ∀ r c, r<p.M → c<p.N → p.scm*r+p.scn*c < bounds p.C

theorem a_safe (p : Parameters) (bounds : RegionBounds) (s : BlockState)
    (rows : Fin p.BM → Nat) (ki : Nat) (hM : 0<p.M) (hb : Bounds p bounds)
    (hp : s.regs .ptr [p.BM,p.BK] "a_ptrs" = some (RealLoads.aAddresses p rows ki))
    (hk : s.regs .int [] "ki" = some (Tile.scalar (ki : Int)))
    (hl : s.regs .nat [p.BK] "offs_k_for_mask" = some (Tile.vec Fin.val)) :
    (RealLoads.aLoad p).SafeAt bounds s := by
  simp [RealLoads.aLoad,RealLoads.remaining,Op.SafeAt.eq_def,
    MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,Op.PointerAddressesSafeOn,
    Op.MemorySafe,MaskOpt.Active,evalOp.eq_def,hp,hk,hl,bind,Option.bind,
    Tile.mat,Tile.vec,Tile.scalar,Tile.remap,Tile.expandDim,TileShape.dropInsertedIndex,
    Tile.bop,Tile.cop,Broadcast.leftIndex,Broadcast.rightIndex,
    NumericDType.mul,NumericDType.sub,ComparableDType.lt,RealLoads.aAddresses,
    -Tile.scalar_eta]
  intro i k hmask
  exact hb.a _ _ (RealTiles.clamp_bound _ _ hM)
    ((RealMath.signed_tail_mask p.K p.BK ki k.val).mp hmask)

theorem b_safe (p : Parameters) (bounds : RegionBounds) (s : BlockState)
    (columns : Fin p.BN → Nat) (ki : Nat) (hN : 0<p.N) (hb : Bounds p bounds)
    (hp : s.regs .ptr [p.BK,p.BN] "b_ptrs" = some (RealLoads.bAddresses p columns ki))
    (hk : s.regs .int [] "ki" = some (Tile.scalar (ki : Int)))
    (hl : s.regs .nat [p.BK] "offs_k_for_mask" = some (Tile.vec Fin.val)) :
    (RealLoads.bLoad p).SafeAt bounds s := by
  simp [RealLoads.bLoad,RealLoads.remaining,Op.SafeAt.eq_def,
    MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,Op.PointerAddressesSafeOn,
    Op.MemorySafe,MaskOpt.Active,evalOp.eq_def,hp,hk,hl,bind,Option.bind,
    Tile.mat,Tile.vec,Tile.scalar,Tile.remap,Tile.expandDim,TileShape.dropInsertedIndex,
    Tile.bop,Tile.cop,Broadcast.leftIndex,Broadcast.rightIndex,
    NumericDType.mul,NumericDType.sub,ComparableDType.lt,RealLoads.bAddresses,
    -Tile.scalar_eta]
  intro k j hmask
  exact hb.b _ _ ((RealMath.signed_tail_mask p.K p.BK ki k.val).mp hmask)
    (RealTiles.clamp_bound _ _ hN)

def output (p : Parameters) : Stmt :=
  .store .real [p.BM,p.BN] (.ptr (.ref .ptr [p.BM,p.BN] "c_ptrs"))
    (.ref .real [p.BM,p.BN] "c") (.mask (.ref .bool [p.BM,p.BN] "c_mask"))

local instance : Inhabited Stmt := ⟨.assign .nat [] "" (.constNat 0)⟩

theorem output_matches_source (p : Parameters) : (RealStore.code p)[5]! = output p := by
  simp only [RealStore.code,RealProgram.body,RealSource.kernel,
    Parameters.source,TypedSource.kernel,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,
    List.map_cons,List.map_nil,FloatDType.toTileDType]
  rfl

theorem output_safe (p : Parameters) (bounds : RegionBounds) (s : BlockState)
    (r c : Nat) (values : Fin p.BM → Fin p.BN → ℝ) (hb : Bounds p bounds) :
    Stmt.TraceSafe bounds (output p) (RealStore.prepared p s r c values) := by
  simp [output,Stmt.TraceSafe,Op.SafeAt.eq_def,MemAccess.SafeAt,MaskOpt.SafeAt,
    MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,Op.PointerAddressesSafeOn,
    Op.MemorySafe,MaskOpt.Active,RealStore.prepared,RealStore.offset,RealStore.active,
    evalOp.eq_def,bind,Option.bind,Tile.mat]
  intro i j hr hc
  exact hb.c _ _ hr hc

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealAccess
