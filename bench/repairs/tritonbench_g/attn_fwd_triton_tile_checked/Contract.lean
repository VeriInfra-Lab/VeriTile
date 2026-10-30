import bench.repairs.tritonbench_g.attn_fwd_triton.Contract

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked
open AttnFwdTriton
open Checked

def staticTiles (g : Geometry) : Bool := decide (128 * HostShape.blockD g.headDim ≤ 1048576)
def launchChecks (g : Geometry) (qs ks : TensorMetadata) : Bool :=
  Checked.launchChecks g qs ks && staticTiles g

/-- Exact additional live Python branch, after the original checks and before
the unchanged signed-address envelope. Empty calls never reach this branch. -/
theorem launch_iff (g : Geometry) (qs ks : TensorMetadata) :
    launchChecks g qs ks=Bool.true ↔ Checked.launchChecks g qs ks=Bool.true ∧
      ¬(128 * HostShape.blockD g.headDim > 1048576) := by
  simp [launchChecks,staticTiles]

theorem inherited (g : Geometry) (qs ks : TensorMetadata)
    (h : launchChecks g qs ks=Bool.true) : Checked.launchChecks g qs ks=Bool.true :=
  ((launch_iff g qs ks).mp h).1

/-- Complete shapes of source vectors, broadcasts, score/P tiles, dots,
accumulator, output cast and pointers. Transposition does not change numel. -/
def tileShapes (BD : Nat) : List (List Nat) :=
  [[128],[64],[BD],[128,BD],[BD,64],[64,BD],[128,64]]

theorem all_tiles_fit (g : Geometry) (h : staticTiles g=Bool.true) :
    ∀ shape ∈ tileShapes (HostShape.blockD g.headDim), shape.prod ≤ 1048576 := by
  have hb : 128 * HostShape.blockD g.headDim ≤ 1048576 := by simpa [staticTiles] using h
  intro shape hs
  simp only [tileShapes,List.mem_cons,List.not_mem_nil,or_false] at hs
  rcases hs with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;>
    simp only [List.prod_cons,List.prod_nil,Nat.mul_one] <;> omega

theorem admitted_tiles (g : Geometry) (qs ks : TensorMetadata)
    (h : launchChecks g qs ks=Bool.true) :
    ∀ shape ∈ tileShapes (HostShape.blockD g.headDim), shape.prod ≤ 1048576 :=
  all_tiles_fit g (by simpa [staticTiles] using ((launch_iff g qs ks).mp h).2)

#axiomsClean launch_iff
#axiomsClean all_tiles_fit
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked
