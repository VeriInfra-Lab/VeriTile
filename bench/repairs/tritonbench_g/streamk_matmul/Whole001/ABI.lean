import bench.repairs.tritonbench_g.streamk_matmul.Whole001.Native
namespace VeriTile.Bench.Repairs.StreamK.Whole001
namespace ABI
open Source Plan Protocol
set_option maxHeartbeats 8000000

/-- Source arithmetic envelope, derived from the checked wrapper guards by
`Checked.Envelope.valid`. These are not allocator or ordinary Tensor laws. -/
structure Valid (p : Parameters) : Prop where
  dimensions : p.h.m < 2^31 ∧ p.h.n < 2^31 ∧ p.h.k < 2^31
  blocks : p.h.bm < 2^30 ∧ p.h.bn < 2^30 ∧ p.h.bk < 2^30
  cdivM : p.h.m+p.h.bm-1 < 2^31
  cdivN : p.h.n+p.h.bn-1 < 2^31
  cdivK : p.h.k+p.h.bk-1 < 2^31
  programs : p.h.programs < 2^31
  iter : (select p.h).iterations < 2^31
  groupWidth : 8*VeriTile.Bench.TritonBenchG.StreamK.ceilDiv p.h.n p.h.bn < 2^31
  tiledRows : VeriTile.Bench.TritonBenchG.StreamK.ceilDiv p.h.m p.h.bm*p.h.bm < 2^63
  tiledCols : VeriTile.Bench.TritonBenchG.StreamK.ceilDiv p.h.n p.h.bn*p.h.bn < 2^63
  streamWork : (select p.h).streamTiles*(select p.h).iterations < 2^63
  /-- The un-clipped next-tile boundary is evaluated before `minimum stop`. -/
  nextBoundary : (select p.h).streamTiles*(select p.h).iterations+
    (select p.h).iterations < 2^63
  strides : p.strideAM < 2^63 ∧ p.strideAK < 2^63 ∧
    p.strideBK < 2^63 ∧ p.strideBN < 2^63
  advanceA : ((select p.h).iterations+1)*p.h.bk*p.strideAK < 2^63
  advanceB : ((select p.h).iterations+1)*p.h.bk*p.strideBK < 2^63
  pointerA :
    (VeriTile.Bench.TritonBenchG.StreamK.ceilDiv p.h.m p.h.bm*p.h.bm)*p.strideAM+
      (((select p.h).iterations+1)*p.h.bk)*p.strideAK < 2^63
  pointerB :
    (((select p.h).iterations+1)*p.h.bk)*p.strideBK+
      (VeriTile.Bench.TritonBenchG.StreamK.ceilDiv p.h.n p.h.bn*p.h.bn)*p.strideBN < 2^63
  pointerC :
    (VeriTile.Bench.TritonBenchG.StreamK.ceilDiv p.h.m p.h.bm*p.h.bm)*p.h.n+
      VeriTile.Bench.TritonBenchG.StreamK.ceilDiv p.h.n p.h.bn*p.h.bn < 2^63
  scratch : p.h.programs*p.h.bm*p.h.bn < 2^63

/-- Positive P is used verbatim; nonpositive user P selects ordinary blocking,
exactly as the source host conditional. -/
def programs (user : Int) : Nat := user.toNat

theorem programs_nonpositive (user : Int) (h : user ≤ 0) : programs user=0 := by
  simp [programs,Int.toNat_eq_zero.mpr h]

theorem programs_positive (user : Int) (h : 0 < user) : (programs user : Int)=user := by
  exact Int.toNat_of_nonneg (by omega)

theorem selected_edges_representable (p : Parameters) (valid : Valid p)
    (hs : 0 < (select p.h).streamTiles) (pr : Nat) (hp : pr ≤ p.h.programs) :
    edge ((select p.h).streamTiles*(select p.h).iterations/p.h.programs)
      ((select p.h).streamTiles*(select p.h).iterations%p.h.programs) pr < 2^63 := by
  have hm:=edge_mono ((select p.h).streamTiles*(select p.h).iterations/p.h.programs)
    ((select p.h).streamTiles*(select p.h).iterations%p.h.programs) pr p.h.programs hp
  have hl:=edge_last ((select p.h).streamTiles*(select p.h).iterations) p.h.programs (selected_active p.h hs).1
  have hb:=valid.streamWork
  omega

theorem next_boundary_representable (p : Parameters) (valid : Valid p)
    (current : Nat)
    (hc : current ≤ (select p.h).streamTiles*(select p.h).iterations) :
    current+((select p.h).iterations-current%(select p.h).iterations) < 2^63 := by
  have hb := valid.nextBoundary
  omega

/-- Same-stream ordered kernel completion is the ordinary CUDA launch model.
No CTA spin, lock, residency, cross-program value flag, or fairness premise is
present: each source while loop decreases stop-start and all other loops are
finite, as proved by Protocol.chunks and represented by finite event lists. -/
def completionBudget (p : Parameters) (ops : Instructions α) : Nat :=
  (firstWave p ops).length+(reducePairs p).length+(fullWave p ops).length

#axiomsClean programs_nonpositive
#axiomsClean programs_positive
#axiomsClean selected_edges_representable
#axiomsClean next_boundary_representable
#specNonCircular Valid avoiding [Source.call,Schedule.call,Memory.run,Returned.value,Native.Numerical]
end ABI
end VeriTile.Bench.Repairs.StreamK.Whole001
