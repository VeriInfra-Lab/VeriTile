import bench.repairs.tritonbench_g.int4_matmul.IO
import bench.repairs.tritonbench_g.int4_matmul.Proofs.Kernel.AccessSafety

namespace VeriTile.Bench.Repairs.Int4Matmul.IOProof
open VeriTile.Triton FP.Structural KernelExecution KernelSource
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

theorem source_bounds (x : Raw.Arguments) (geometry : Raw.Geometry x) (t : Tile)
    (member : t∈Host.sourceTiles) (o : Ops α) (rest : Algebra α) (lanes : List Lane)
    (v : (IO.programIO x t o rest lanes).Inputs) (bounds : RegionName → Nat)
    (inside : (IO.programIO x t o rest lanes).InBounds bounds v)
    (s : RegisterState (SourceGraph.Node α)) : SourceSafety.Bounds (Raw.parameters x t) s bounds := by
  constructor
  · intro k _ i live
    rw [Raw.aOffset_dense x geometry]
    have live' := live
    rw [Raw.inner_kIndex] at live'
    have cap := rowMajor_bound x.shape.m x.shape.k
      (inputRow (Raw.parameters x t) s i.1) (kIndex t (s.pids 1) k i.2.1.val)
      (Nat.mod_lt _ geometry.shape.mpos) live'
    exact inside.2.1 0 (⟨_,cap⟩,()) (by trivial)
  · intro k _ i live
    have offsets := Raw.packed_offsets x geometry t member s k i live
    rw [offsets.1]
    have live' := live
    rw [Raw.inner_kIndex] at live'
    have cap := active_addresses x.shape geometry.shape 0
      (inputColumn (Raw.parameters x t) s i.2.1) (kIndex t (s.pids 1) k i.1.val)
      geometry.shape.mpos (Nat.mod_lt _ geometry.shape.npos) live'
    exact inside.2.1 1 (⟨_,cap.2.1⟩,()) (by trivial)
  · intro k _ i live
    have offsets := Raw.packed_offsets x geometry t member s k i live
    rw [offsets.2.1]
    have live' := live
    rw [Raw.inner_kIndex] at live'
    have cap := active_addresses x.shape geometry.shape 0
      (inputColumn (Raw.parameters x t) s i.2.1) (kIndex t (s.pids 1) k i.1.val)
      geometry.shape.mpos (Nat.mod_lt _ geometry.shape.npos) live'
    exact inside.2.1 2 (⟨_,cap.2.2.1⟩,()) (by trivial)
  · intro k _ i live
    have offsets := Raw.packed_offsets x geometry t member s k i live
    rw [offsets.2.2]
    have live' := live
    rw [Raw.inner_kIndex] at live'
    have cap := active_addresses x.shape geometry.shape 0
      (inputColumn (Raw.parameters x t) s i.2.1) (kIndex t (s.pids 1) k i.1.val)
      geometry.shape.mpos (Nat.mod_lt _ geometry.shape.npos) live'
    exact inside.2.1 3 (⟨_,cap.2.2.2.1⟩,()) (by trivial)
  · intro i live
    have hr : rowBlock (Raw.parameters x t) s*t.bm+i.1.val<x.shape.m := live.1
    have hc : columnBlock (Raw.parameters x t) s*t.bn+i.2.1.val<x.shape.n := live.2
    have cap := inside.2.2.1 0 (⟨_,hr⟩,⟨_,hc⟩,()) (by trivial)
    change (rowBlock (Raw.parameters x t) s*t.bm+i.1.val)*x.shape.n+
      (columnBlock (Raw.parameters x t) s*t.bn+i.2.1.val)<bounds "out" at cap
    simpa only [outputOffset,Raw.parameters,Nat.one_mul,Nat.mul_comm] using cap

theorem invocation_safe (x : Raw.Arguments) (geometry : Raw.Geometry x) (t : Tile)
    (member : t∈Host.sourceTiles) (o : Ops α) (rest : Algebra α) (lanes : List Lane)
    (v : (IO.programIO x t o rest lanes).Inputs) (bounds : RegionName → Nat)
    (inside : (IO.programIO x t o rest lanes).InBounds bounds v)
    (p : IO.Program) (s : RegisterState (SourceGraph.Node α)) :
    (IO.invocation x t o rest p).Safe bounds s :=
  SourceSafety.complete o rest (Raw.parameters x t) _ bounds
    (source_bounds x geometry t member o rest lanes v bounds inside _)

#axiomsClean invocation_safe
end
end VeriTile.Bench.Repairs.Int4Matmul.IOProof
