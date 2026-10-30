import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_SignedTileBlockPtr

/-!
Selected-FP32 source branch, retaining every source index and block-pointer
register assignment. Natural bases/strides are the domain of the existing
BlockPtr representation. This module bridges the actual statements to the
frozen signed-memory tile result by observation of memory and padding state.
It uses the current mathematical FP32 evaluator, not native IEEE semantics.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SourceBranch
open VeriTile.Triton SignedView SignedTile

structure NatViews where
  input : RegionName
  output : RegionName
  inputBase : Nat
  outputBase : Nat
  inputStride : Nat
  outputStride : Nat
  n : Nat
  B : Nat

def NatViews.inputView (c : NatViews) : View := ⟨c.input, c.inputBase, c.inputStride⟩
def NatViews.outputView (c : NatViews) : View := ⟨c.output, c.outputBase, c.outputStride⟩

def ObsEq (s t : BlockState) : Prop := s.mem = t.mem ∧ s.undef = t.undef

theorem ObsEq.refl (s : BlockState) : ObsEq s s := ⟨rfl, rfl⟩
theorem ObsEq.symm {s t : BlockState} (h : ObsEq s t) : ObsEq t s :=
  ⟨h.1.symm, h.2.symm⟩
theorem ObsEq.trans {s t u : BlockState} (h : ObsEq s t) (h' : ObsEq t u) : ObsEq s u :=
  ⟨h.1.trans h'.1, h.2.trans h'.2⟩

theorem setReg_obs (s : BlockState) (name : RegName) (dtype : TileDType)
    (shape : TileShape) (value : Tile dtype shape) : ObsEq (s.setReg name dtype shape value) s :=
  ⟨rfl, rfl⟩

def ptrValue (region : RegionName) (base stride n start B : Nat) : Tile .blockPtr [B] :=
  ⟨fun _ => { region := region, baseOffset := base, parentShape := [n], blockShape := [B], strides := [stride], offsets := [start] }⟩

def dynamicPtr (region : RegionName) (base stride n B : Nat) : Op .blockPtr [B] :=
  .makeBlockPtrDynOffsets region (.constNat base) [n] [B] [stride] [.ref .nat [] "offset0"]

theorem dynamicPtr_eval (region : RegionName) (base stride n start B : Nat)
    (s : BlockState) (h : s.regs .nat [] "offset0" = some (Tile.scalar start)) :
    evalOp (dynamicPtr region base stride n B) s = some (ptrValue region base stride n start B) := by
  simp only [dynamicPtr, evalOp, h, List.mapM_cons, List.mapM_nil, bind, Option.bind,
    Option.pure_def]
  rfl

def NatViews.perTile (c : NatViews) : List Stmt :=
  [.assign .nat [] "tile_id0" (.ref .nat [] "tile_id"),
   .assign .nat [] "offset0" (.mul .nat .nil (.ref .nat [] "tile_id0") (.constNat c.B)),
   .assign .blockPtr [c.B] "in0_bptr" (dynamicPtr c.input c.inputBase c.inputStride c.n c.B),
   .assign .fp32 [c.B] "in0" (.castFloat .fp32 .fp32
     (.load .fp32 (.blockPtr (.ref .blockPtr [c.B] "in0_bptr") [0]) .none .unspecified)),
   .assign .fp32 [c.B] "out0" (reluOp c.B),
   .assign .blockPtr [c.B] "out0_bptr" (dynamicPtr c.output c.outputBase c.outputStride c.n c.B),
   .store .fp32 [c.B] (.blockPtr (.ref .blockPtr [c.B] "out0_bptr") [0])
     (.castFloat .fp32 .fp32 (.ref .fp32 [c.B] "out0")) .none]

def NatViews.indexState (c : NatViews) (T : Nat) (s : BlockState) : BlockState :=
  (s.setReg "tile_id0" .nat [] (Tile.scalar T)).setReg "offset0" .nat [] (Tile.scalar (T * c.B))

def NatViews.pointerState (c : NatViews) (T : Nat) (s : BlockState) : BlockState :=
  (c.indexState T s).setReg "in0_bptr" .blockPtr [c.B]
    (ptrValue c.input c.inputBase c.inputStride c.n (T * c.B) c.B)

noncomputable def NatViews.storeState (c : NatViews) (T : Nat) (s : BlockState) : BlockState :=
  (afterValue c.inputView c.n (T * c.B) c.B (c.pointerState T s)).setReg
    "out0_bptr" .blockPtr [c.B]
    (ptrValue c.output c.outputBase c.outputStride c.n (T * c.B) c.B)

noncomputable def NatViews.sourceFinal (c : NatViews) (T : Nat) (s : BlockState) : BlockState :=
  scatter32 c.outputView c.n (T * c.B) c.B
    (resultTile c.inputView c.n (T * c.B) c.B (c.pointerState T s)) (c.storeState T s)

theorem pointer_load (region : RegionName) (base stride n start B : Nat) (s : BlockState)
    (h : s.regs .blockPtr [B] "in0_bptr" = some (ptrValue region base stride n start B)) :
    evalOp (.castFloat .fp32 .fp32
      (.load .fp32 (.blockPtr (.ref .blockPtr [B] "in0_bptr") [0]) .none .unspecified)) s =
      some (loadedTile ⟨region, base, stride⟩ n start B s) := by
  rw [cast32_identity]
  have he : evalOp (.load .fp32 (.blockPtr (.ref .blockPtr [B] "in0_bptr") [0])
      .none .unspecified) s = evalOp (natBlockLoad region base stride n start B) s := by
    simp only [natBlockLoad, evalOp, h, natBlockPtr_eval, ptrValue]
  exact he.trans ((natBlockLoad_eq region base stride n start B s).trans
    (load32_eval ⟨region, base, stride⟩ n start B s))

theorem pointer_store (region : RegionName) (base stride n start B : Nat) (s : BlockState)
    (values : Tile .fp32 [B])
    (hp : s.regs .blockPtr [B] "out0_bptr" = some (ptrValue region base stride n start B))
    (hv : s.regs .fp32 [B] "out0" = some values) :
    stepStmt (.store .fp32 [B] (.blockPtr (.ref .blockPtr [B] "out0_bptr") [0])
      (.castFloat .fp32 .fp32 (.ref .fp32 [B] "out0")) .none) s =
      some (scatter32 ⟨region, base, stride⟩ n start B values s) := by
  have he : stepStmt (.store .fp32 [B] (.blockPtr (.ref .blockPtr [B] "out0_bptr") [0])
      (.castFloat .fp32 .fp32 (.ref .fp32 [B] "out0")) .none) s =
      stepStmt (natBlockStore region base stride n start B) s := by
    simp only [natBlockStore, stepStmt, evalOp_ref, hp, natBlockPtr_eval, ptrValue]
  exact he.trans ((natBlockStore_eq region base stride n start B s).trans
    ((store_step ⟨region, base, stride⟩ n start B s).trans
      (store32_step ⟨region, base, stride⟩ n start B "out0" values s hv)))

theorem sourceTile_exec (c : NatViews) (T : Nat) (s : BlockState)
    (ht : s.regs .nat [] "tile_id" = some (Tile.scalar T)) :
    stepStmts c.perTile s = some (c.sourceFinal T s) := by
  have hindex : stepStmts
      [.assign .nat [] "tile_id0" (.ref .nat [] "tile_id"),
       .assign .nat [] "offset0" (.mul .nat .nil (.ref .nat [] "tile_id0") (.constNat c.B))] s =
      some (c.indexState T s) := by
    simp only [stepStmts, stepStmt, evalOp_ref, ht, bind, Option.bind,
      evalOp_mul, evalOp_constNat, BlockState.setReg_same]
    rfl
  have hptr : stepStmt (.assign .blockPtr [c.B] "in0_bptr"
      (dynamicPtr c.input c.inputBase c.inputStride c.n c.B)) (c.indexState T s) =
      some (c.pointerState T s) := by
    simp only [stepStmt]
    rw [dynamicPtr_eval _ _ _ _ (T * c.B) _ _ (by simp [NatViews.indexState])]
    rfl
  have hload : stepStmt (.assign .fp32 [c.B] "in0" (.castFloat .fp32 .fp32
      (.load .fp32 (.blockPtr (.ref .blockPtr [c.B] "in0_bptr") [0]) .none .unspecified)))
      (c.pointerState T s) = some (afterLoad c.inputView c.n (T * c.B) c.B (c.pointerState T s)) := by
    simp only [stepStmt]
    have he := pointer_load c.input c.inputBase c.inputStride c.n (T * c.B) c.B
      (c.pointerState T s) (by simp [NatViews.pointerState])
    exact congrArg (fun value : Option (Tile .fp32 [c.B]) =>
      value.bind (fun value => some ((c.pointerState T s).setReg "in0" .fp32 [c.B] value))) he
  have houtptr : stepStmt (.assign .blockPtr [c.B] "out0_bptr"
      (dynamicPtr c.output c.outputBase c.outputStride c.n c.B))
      (afterValue c.inputView c.n (T * c.B) c.B (c.pointerState T s)) =
      some (c.storeState T s) := by
    simp only [stepStmt]
    rw [dynamicPtr_eval _ _ _ _ (T * c.B) _ _
      (by simp [afterValue, afterLoad, NatViews.pointerState, NatViews.indexState])]
    rfl
  have hstore := pointer_store c.output c.outputBase c.outputStride c.n (T * c.B) c.B
    (c.storeState T s) (resultTile c.inputView c.n (T * c.B) c.B (c.pointerState T s))
    (by simp [NatViews.storeState]) (by simp [NatViews.storeState, afterValue])
  rw [show c.perTile =
      [.assign .nat [] "tile_id0" (.ref .nat [] "tile_id"),
       .assign .nat [] "offset0" (.mul .nat .nil (.ref .nat [] "tile_id0") (.constNat c.B))] ++
      c.perTile.drop 2 by rfl]
  simp only [stepStmts.append_some hindex]
  simp only [NatViews.perTile, List.drop, stepStmts, hptr, hload, value_step,
    houtptr, hstore]
  rfl

theorem resultTile_obs (input : View) (n start B : Nat) {s t : BlockState}
    (h : ObsEq s t) : resultTile input n start B s = resultTile input n start B t := by
  ext i
  simp only [resultTile, loadedTile]
  congr 1
  split
  · simp only [BlockState.readMemValue, BlockState.readMemAs, h.1]
  · exact BlockState.paddingValue_congr s t .fp32 .unspecified _ _ h.2

theorem scatter32_obs (output : View) (n start B : Nat) (values : Tile .fp32 [B])
    {s t : BlockState} (h : ObsEq s t) :
    ObsEq (scatter32 output n start B values s) (scatter32 output n start B values t) := by
  unfold scatter32
  generalize TileShape.allIndices [B] = lanes
  induction lanes generalizing s t with
  | nil => exact h
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : start + i.1.val < n
    · simp only [hi, if_true]
      apply ih
      constructor
      · simp only [BlockState.writeMemTyped, BlockState.writeMemAs, h.1]
      · exact h.2
    · simpa only [hi, if_false] using ih h

theorem scatter32_control (output : View) (n start B : Nat) (values : Tile .fp32 [B])
    (s : BlockState) :
    (scatter32 output n start B values s).regs = s.regs ∧
    (scatter32 output n start B values s).pids = s.pids ∧
    (scatter32 output n start B values s).numPids = s.numPids := by
  unfold scatter32
  generalize TileShape.allIndices [B] = lanes
  induction lanes generalizing s with
  | nil => exact ⟨rfl, rfl, rfl⟩
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : start + i.1.val < n
    · simpa only [hi, if_true, BlockState.writeMemTyped, BlockState.writeMemAs] using
        ih (s.writeMemTyped .fp32 output.region (output.address (start + i.1.val)) (values.data i))
    · simpa only [hi, if_false] using ih s

theorem tile_obs_congr (input output : View) (n start B : Nat) {s t : BlockState}
    (h : ObsEq s t) :
    ObsEq (finalState input output n start B s) (finalState input output n start B t) := by
  unfold finalState
  rw [resultTile_obs input n start B h]
  apply scatter32_obs
  exact h

theorem sourceFinal_obs (c : NatViews) (T : Nat) (s : BlockState) :
    ObsEq (c.sourceFinal T s) (finalState c.inputView c.outputView c.n (T * c.B) c.B s) := by
  unfold NatViews.sourceFinal finalState
  rw [resultTile_obs c.inputView c.n (T * c.B) c.B
    (show ObsEq (c.pointerState T s) s from ⟨rfl, rfl⟩)]
  apply scatter32_obs
  exact ⟨rfl, rfl⟩

theorem sourceFinal_control (c : NatViews) (T : Nat) (s : BlockState) :
    (c.sourceFinal T s).regs .nat [] "pid" = s.regs .nat [] "pid" ∧
    (c.sourceFinal T s).regs .nat [] "num_ctas" = s.regs .nat [] "num_ctas" ∧
    (c.sourceFinal T s).pids = s.pids ∧
    (c.sourceFinal T s).numPids = s.numPids := by
  have h := scatter32_control c.outputView c.n (T * c.B) c.B
    (resultTile c.inputView c.n (T * c.B) c.B (c.pointerState T s)) (c.storeState T s)
  simp only [NatViews.sourceFinal, h.1, h.2.1, h.2.2]
  simp [NatViews.storeState, afterValue, afterLoad, NatViews.pointerState, NatViews.indexState]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SourceBranch
