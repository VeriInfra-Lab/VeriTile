import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_SignedView

/-!
Complete selected-FP32 tile value body for `relu_strided_buffer.py`.
The body loads the signed input view, keeps both source element-type casts,
computes `where(x > 0, x, 0)` and stores through the signed output view.
The source block-pointer address and boundary mask are lowered using SignedView.
Program/grid indexing and wrapper metadata are separate composition obligations.

The existing `.fp32` evaluator uses a WithBot-real carrier. The theorems below
are exact in that evaluator; they do not claim a native IEEE implementation.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SignedTile
open VeriTile.Triton
open SignedView

noncomputable def relu32 (x : TileCarrier .fp32) : TileCarrier .fp32 :=
  if ComparableDType.fp32.gt x (some 0) then x else some 0

theorem relu32_some (x : ℝ) : relu32 (some x) = some (max 0 x) := by
  unfold relu32 ComparableDType.gt
  change (if decide ((0 : WithBot ℝ) < (x : WithBot ℝ)) then some x else some 0) = _
  simp only [decide_eq_true_eq]
  by_cases hx : 0 < x
  · simp [hx, max_eq_right (le_of_lt hx)]
  · simp [hx, max_eq_left (le_of_not_gt hx)]

def zero32 : Op .fp32 [] := .castFloat .real .fp32 (.const 0)

def reluOp (B : Nat) : Op .fp32 [B] :=
  .where (.gt .fp32 .scalarR (.ref .fp32 [B] "in0") zero32)
    (.ref .fp32 [B] "in0") (zero32.broadcast [B])

def tileStore (output : View) (n start B : Nat) : Stmt :=
  .store .fp32 [B] (.region output.region (addressOp output start B))
    (.castFloat .fp32 .fp32 (.ref .fp32 [B] "out0"))
    (.mask (maskOp n start B))

def tileBody (input output : View) (n start B : Nat) : List Stmt :=
  [.assign .fp32 [B] "in0" (.castFloat .fp32 .fp32 (load32 input n start B)),
   .assign .fp32 [B] "out0" (reluOp B),
   tileStore output n start B]

noncomputable def loadedTile (input : View) (n start B : Nat)
    (s : BlockState) : Tile .fp32 [B] :=
  ⟨fun i => if start + i.1.val < n then
    s.readMemValue .fp32 input.region (input.address (start + i.1.val))
    else s.paddingValue .fp32 .unspecified input.region
      (input.address (start + i.1.val))⟩

noncomputable def resultTile (input : View) (n start B : Nat)
    (s : BlockState) : Tile .fp32 [B] :=
  ⟨fun i => relu32 ((loadedTile input n start B s).data i)⟩

noncomputable def afterLoad (input : View) (n start B : Nat)
    (s : BlockState) : BlockState :=
  s.setReg "in0" .fp32 [B] (loadedTile input n start B s)

noncomputable def afterValue (input : View) (n start B : Nat)
    (s : BlockState) : BlockState :=
  (afterLoad input n start B s).setReg "out0" .fp32 [B] (resultTile input n start B s)

noncomputable def finalState (input output : View) (n start B : Nat)
    (s : BlockState) : BlockState :=
  scatter32 output n start B (resultTile input n start B s) (afterValue input n start B s)

theorem cast32_identity {shape : TileShape} (e : Op .fp32 shape) (s : BlockState) :
    evalOp (.castFloat .fp32 .fp32 e) s = evalOp e s := by
  rw [evalOp_castFloat]
  change (evalOp e s >>= fun value => some value) = evalOp e s
  cases evalOp e s <;> rfl

theorem zero32_eval (s : BlockState) :
    evalOp zero32 s = some (Tile.scalar (some 0) : Tile .fp32 []) := by
  exact (evalOp_castFloat .real .fp32 (.const 0) s).trans (by
    change (evalOp (Op.const (0 : ℝ)) s).bind
      (fun value => some (⟨value.data⟩ : Tile .fp32 [])) = _
    rw [evalOp_const]
    rfl)

theorem reluOp_eval (B : Nat) (s : BlockState) (input : Tile .fp32 [B])
    (hin : s.regs .fp32 [B] "in0" = some input) :
    evalOp (reluOp B) s = some (⟨fun i => relu32 (input.data i)⟩ : Tile .fp32 [B]) := by
  simp only [reluOp, evalOp, zero32_eval, hin, bind, Option.bind]
  rfl

theorem load_step (input : View) (n start B : Nat) (s : BlockState) :
    stepStmt (.assign .fp32 [B] "in0" (.castFloat .fp32 .fp32 (load32 input n start B))) s =
      some (afterLoad input n start B s) := by
  have he : evalOp (.castFloat .fp32 .fp32 (load32 input n start B)) s =
      some (loadedTile input n start B s) :=
    (cast32_identity (load32 input n start B) s).trans (load32_eval input n start B s)
  simp only [stepStmt]
  exact congrArg (fun value : Option (Tile .fp32 [B]) =>
    value.bind (fun value => some (s.setReg "in0" .fp32 [B] value))) he

theorem value_step (input : View) (n start B : Nat) (s : BlockState) :
    stepStmt (.assign .fp32 [B] "out0" (reluOp B)) (afterLoad input n start B s) =
      some (afterValue input n start B s) := by
  have hin : (afterLoad input n start B s).regs .fp32 [B] "in0" =
      some (loadedTile input n start B s) := by simp [afterLoad]
  simp only [stepStmt, reluOp_eval B _ _ hin, bind, Option.bind]
  rfl

/-- Both explicit casts remain in the source body. Identity is proved in the
selected mathematical FP32 evaluator, rather than erased from the body. -/
theorem store_step (output : View) (n start B : Nat) (s : BlockState) :
    stepStmt (tileStore output n start B) s =
      stepStmt (store32 output n start B "out0") s := by
  simp only [tileStore, store32, stepStmt]
  congr 1
  exact cast32_identity (.ref .fp32 [B] "out0") s

theorem tile_exec (input output : View) (n start B : Nat) (s : BlockState) :
    stepStmts (tileBody input output n start B) s =
      some (finalState input output n start B s) := by
  have hout : (afterValue input n start B s).regs .fp32 [B] "out0" =
      some (resultTile input n start B s) := by simp [afterValue]
  simp only [tileBody, stepStmts, load_step, value_step,
    store_step, store32_step output n start B "out0" _ _ hout]
  rfl

/-- The composable tile contract derives output from the same call's initial
input cells. No output-value callback or complete-output premise occurs. -/
theorem tile_success (input output : View) {n capacity start B : Nat}
    (hout : output.Valid n capacity) (hstride : output.stride ≠ 0) (s : BlockState) :
    ∃ final, stepStmts (tileBody input output n start B) s = some final ∧
      (∀ i : TileIndex [B], start + i.1.val < n →
        final.mem output.region (output.address (start + i.1.val)) =
          storedCell32 (relu32 (s.readMemValue .fp32 input.region
            (input.address (start + i.1.val))))) ∧
      (∀ r a, (r ≠ output.region ∨ ∀ i : TileIndex [B],
        start + i.1.val < n → a ≠ output.address (start + i.1.val)) →
        final.mem r a = s.mem r a) := by
  refine ⟨finalState input output n start B s, tile_exec input output n start B s, ?_, ?_⟩
  · intro i hi
    have h := scatter32_readback output hout hstride (resultTile input n start B s)
      (afterValue input n start B s) i hi
    simpa [finalState, resultTile, loadedTile, hi] using h
  · intro r a ha
    have h := scatter32_frame output n start B (resultTile input n start B s)
      (afterValue input n start B s) r a ha
    simpa [finalState, afterValue, afterLoad] using h

/-- Bounds of every active load and store use ordinary signed-view validity.
They are independent of numerical values and introduce no artificial size cap. -/
theorem active_accesses_bounded (input output : View) {n inputCapacity outputCapacity start B : Nat}
    (hin : input.Valid n inputCapacity) (hout : output.Valid n outputCapacity)
    (i : TileIndex [B]) (hi : start + i.1.val < n) :
    input.address (start + i.1.val) < inputCapacity ∧
      output.address (start + i.1.val) < outputCapacity :=
  ⟨address_lt input hin hi, address_lt output hout hi⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SignedTile
