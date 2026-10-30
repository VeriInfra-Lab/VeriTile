import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.NativeBytes
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelBinding

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.WordTarget
open Kernel.Native
noncomputable section

/-! Independent FP32 graph, using entry-memory words. Scalar primitive models
and finite dot instruction trees are explicit. No source execution is used
to compute a target, and no FP32 operation is equated to real arithmetic. -/
def forward (native : Scalar FP32Word) (p : DotTrees) (s : Shape) (b : Bases)
    (entry : NativeBytes.Bytes) (tile : Tile) (row col : Nat) : FP32Word :=
  forwardLane native.ops (Kernel.plans s p) s (currentInputs native.ops s b (NativeBytes.load entry)) tile row col
def query (native : Scalar FP32Word) (p : DotTrees) (s : Shape) (b : Bases)
    (entry : NativeBytes.Bytes) (tile : Tile) (row col : Nat) : FP32Word :=
  queryLane native.ops (Kernel.plans s p) s (currentInputs native.ops s b (NativeBytes.load entry)) tile row col
def key (native : Scalar FP32Word) (p : DotTrees) (s : Shape) (b : Bases)
    (entry : NativeBytes.Bytes) (tile : Tile) (row col : Nat) : FP32Word :=
  (keyValueLane native.ops (Kernel.plans s p) s (currentInputs native.ops s b (NativeBytes.load entry)) tile row col 0).1
def value (native : Scalar FP32Word) (p : DotTrees) (s : Shape) (b : Bases)
    (entry : NativeBytes.Bytes) (tile : Tile) (row col : Nat) : FP32Word :=
  (keyValueLane native.ops (Kernel.plans s p) s (currentInputs native.ops s b (NativeBytes.load entry)) tile row 0 col).2

end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.WordTarget
