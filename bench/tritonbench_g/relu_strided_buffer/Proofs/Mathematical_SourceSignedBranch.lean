import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_SourceBranch

/-!
The signed source branch retains the source's tile/index assignments. The
ordinary block-pointer primitive is lowered to its signed element-offset
formula and a logical boundary mask; both pointer register names retain the
resulting address vectors. No conversion to Nat occurs before the full signed
base + logical_index * stride sum. This is a local explicit lowering, not a
claim that the existing Nat-stride BlockPtr type directly supports signed stride.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SourceSignedBranch
open VeriTile.Triton SignedView SignedTile SourceBranch

structure SignedViews where
  inputView : View
  outputView : View
  n : Nat
  B : Nat

def dynamicIndex (B : Nat) : Op .nat [B] :=
  .add .nat .scalarL (.ref .nat [] "offset0") (.arange B)

def dynamicAddress (v : View) (B : Nat) : Op .nat [B] :=
  .castIntToNat (.add .int .scalarL (.constInt v.base)
    (.mul .int .scalarR (.castNatToInt (dynamicIndex B)) (.constInt v.stride)))

def dynamicMask (n B : Nat) : Op .bool [B] :=
  .lt .nat .scalarR (dynamicIndex B) (.constNat n)

def addressValue (v : View) (start B : Nat) : Tile .nat [B] :=
  ⟨fun i => v.address (start + i.1.val)⟩

theorem dynamicIndex_eval (B start : Nat) (s : BlockState)
    (h : s.regs .nat [] "offset0" = some (Tile.scalar start)) :
    evalOp (dynamicIndex B) s = some (⟨fun i => start + i.1.val⟩ : Tile .nat [B]) := by
  simp only [dynamicIndex, evalOp, h, bind, Option.bind]
  rfl

theorem dynamicAddress_eval (v : View) (B start : Nat) (s : BlockState)
    (h : s.regs .nat [] "offset0" = some (Tile.scalar start)) :
    evalOp (dynamicAddress v B) s = some (addressValue v start B) := by
  simp only [dynamicAddress, evalOp, dynamicIndex_eval B start s h, bind, Option.bind]
  rfl

theorem dynamicMask_eval (n B start : Nat) (s : BlockState)
    (h : s.regs .nat [] "offset0" = some (Tile.scalar start)) :
    evalOp (dynamicMask n B) s =
      some (⟨fun i => decide (start + i.1.val < n)⟩ : Tile .bool [B]) := by
  simp only [dynamicMask, evalOp, dynamicIndex_eval B start s h, bind, Option.bind]
  rfl

def SignedViews.indexState (c : SignedViews) (T : Nat) (s : BlockState) : BlockState :=
  (s.setReg "tile_id0" .nat [] (Tile.scalar T)).setReg "offset0" .nat [] (Tile.scalar (T * c.B))

def SignedViews.pointerState (c : SignedViews) (T : Nat) (s : BlockState) : BlockState :=
  (c.indexState T s).setReg "in0_bptr" .nat [c.B] (addressValue c.inputView (T * c.B) c.B)

noncomputable def SignedViews.storeState (c : SignedViews) (T : Nat) (s : BlockState) : BlockState :=
  (afterValue c.inputView c.n (T * c.B) c.B (c.pointerState T s)).setReg
    "out0_bptr" .nat [c.B] (addressValue c.outputView (T * c.B) c.B)

noncomputable def SignedViews.sourceFinal (c : SignedViews) (T : Nat) (s : BlockState) : BlockState :=
  scatter32 c.outputView c.n (T * c.B) c.B
    (resultTile c.inputView c.n (T * c.B) c.B (c.pointerState T s)) (c.storeState T s)

theorem pointer_load (v : View) (n start B : Nat) (s : BlockState)
    (hp : s.regs .nat [B] "in0_bptr" = some (addressValue v start B))
    (ho : s.regs .nat [] "offset0" = some (Tile.scalar start)) :
    evalOp (.castFloat .fp32 .fp32
      (.load .fp32 (.region v.region (.ref .nat [B] "in0_bptr"))
        (.mask (dynamicMask n B)) .unspecified)) s = some (loadedTile v n start B s) := by
  rw [cast32_identity]
  have he : evalOp (.load .fp32 (.region v.region (.ref .nat [B] "in0_bptr"))
      (.mask (dynamicMask n B)) .unspecified) s = evalOp (load32 v n start B) s := by
    simp only [load32, evalOp, hp, dynamicMask_eval n B start s ho,
      addressOp_eval, maskOp_eval, addressValue]
  exact he.trans (load32_eval v n start B s)

theorem sourceFinal_obs (c : SignedViews) (T : Nat) (s : BlockState) :
    ObsEq (c.sourceFinal T s) (finalState c.inputView c.outputView c.n (T * c.B) c.B s) := by
  unfold SignedViews.sourceFinal finalState
  rw [resultTile_obs c.inputView c.n (T * c.B) c.B
    (show ObsEq (c.pointerState T s) s from ⟨rfl, rfl⟩)]
  apply scatter32_obs
  exact ⟨rfl, rfl⟩

theorem sourceFinal_control (c : SignedViews) (T : Nat) (s : BlockState) :
    (c.sourceFinal T s).regs .nat [] "pid" = s.regs .nat [] "pid" ∧
    (c.sourceFinal T s).regs .nat [] "num_ctas" = s.regs .nat [] "num_ctas" ∧
    (c.sourceFinal T s).pids = s.pids ∧
    (c.sourceFinal T s).numPids = s.numPids := by
  have h := scatter32_control c.outputView c.n (T * c.B) c.B
    (resultTile c.inputView c.n (T * c.B) c.B (c.pointerState T s)) (c.storeState T s)
  simp only [SignedViews.sourceFinal, h.1, h.2.1, h.2.2]
  simp [SignedViews.storeState, afterValue, afterLoad, SignedViews.pointerState, SignedViews.indexState]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SourceSignedBranch
