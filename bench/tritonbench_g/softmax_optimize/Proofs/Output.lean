import bench.tritonbench_g.softmax_optimize.Proofs.Input
import bench.tritonbench_g.softmax_optimize.Proofs.Mathematical_RealStoreFrame

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.Output
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SoftmaxOptimize
open SignedTransport Primitive Input
set_option maxHeartbeats 4000000
noncomputable section

def exponential {B : Nat} (input : Tile .real [B]) (shift : WithBot ℝ) : Tile .real [B] :=
  ⟨fun i => WithBot.realExp (WithBot.realSub (input.data i) shift)⟩

def quotient {B : Nat} (input : Tile .real [B]) (shift denominator : WithBot ℝ) : Tile .real [B] :=
  ⟨fun i => WithBot.realDiv ((exponential input shift).data i) denominator⟩

def arithmetic (B : Nat) : List Stmt :=
  [.assign .real [B] "e"
    (.exp (.sub .real .scalarR (.ref .real [B] "inp") (.ref .real [] "m"))),
   .assign .real [B] "out"
    (.div .real .scalarR (.ref .real [B] "e") (.ref .real [] "z"))]

def arithmeticState {B : Nat} (s : BlockState) (input : Tile .real [B])
    (shift denominator : WithBot ℝ) : BlockState :=
  (s.setReg "e" .real [B] (exponential input shift)).setReg
    "out" .real [B] (quotient input shift denominator)

theorem arithmetic_run {B : Nat} (s : BlockState) (input : Tile .real [B])
    (shift denominator : WithBot ℝ)
    (hi : s.regs .real [B] "inp" = some input)
    (hm : s.regs .real [] "m" = some (Tile.scalar shift))
    (hz : s.regs .real [] "z" = some (Tile.scalar denominator)) :
    stepStmts (arithmetic B) s = some (arithmeticState s input shift denominator) := by
  unfold arithmetic
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := exponential input shift) (by
    simp only [evalOp.eq_def, hi, hm, Option.bind_eq_bind, Option.bind_some]
    rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := quotient input shift denominator) (by
    simp only [evalOp.eq_def, BlockState.setReg_ne_name, BlockState.setReg_same, hz,
      ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
    rfl))]
  rw [stepStmts.nil]
  rfl

def scatter {B : Nat} (s : BlockState) (Y : RegionName) (offsets : TileIndex [B] → Nat)
    (values : Tile .real [B]) (mask : TileIndex [B] → Bool) : BlockState :=
  (TileShape.allIndices [B]).foldl (fun state i =>
    if mask i then state.writeMem Y (offsets i) ((values.data i).unbotD 0) else state) s

theorem store_full_run {B : Nat} (s : BlockState) (Y : RegionName)
    (offsets : TileIndex [B] → Nat) (values : Tile .real [B])
    (hp : s.regs .ptr [B] "output_ptrs" = some ⟨fun i => (Y, offsets i)⟩)
    (hv : s.regs .real [B] "out" = some values) :
    stepStmt (.store .real [B] (.ptr (.ref .ptr [B] "output_ptrs"))
      (.ref .real [B] "out") .none) s =
      some (scatter s Y offsets values (fun _ => Bool.true)) := by
  unfold stepStmt
  simp only [evalOp_ref, hp, hv, Option.bind_eq_bind, Option.bind_some, if_true]
  apply congrArg some
  apply List.foldl_ext
  intro state i hi
  simp only [scatter, Bool.true_eq, ↓reduceIte, BlockState.writeMemTyped_real,
    FloatDType.real_storeValue]

theorem store_tail_run {B : Nat} (s : BlockState) (Y : RegionName)
    (offsets : TileIndex [B] → Nat) (values : Tile .real [B]) (mask : Tile .bool [B])
    (hp : s.regs .ptr [B] "output_ptrs" = some ⟨fun i => (Y, offsets i)⟩)
    (hv : s.regs .real [B] "out" = some values)
    (hm : s.regs .bool [B] "mask" = some mask) :
    stepStmt (.store .real [B] (.ptr (.ref .ptr [B] "output_ptrs"))
      (.ref .real [B] "out") (.mask (.ref .bool [B] "mask"))) s =
      some (scatter s Y offsets values mask.data) := by
  unfold stepStmt
  simp only [evalOp_ref, hp, hv, hm, Option.bind_eq_bind, Option.bind_some, Option.map_some]
  apply congrArg some
  apply List.foldl_ext
  intro state i hi
  split
  · rw [BlockState.writeMemTyped_real, FloatDType.real_storeValue]
  · rfl

theorem scatter_frame {B : Nat} (s : BlockState) (Y : RegionName)
    (offsets : TileIndex [B] → Nat) (values : Tile .real [B]) (mask : TileIndex [B] → Bool)
    (region : RegionName) (address : Nat)
    (h : ∀ i, mask i → ¬ (Y = region ∧ offsets i = address)) :
    (scatter s Y offsets values mask).mem region address = s.mem region address :=
  masked_scatter_raw_frame _ s Y region offsets (fun i => (values.data i).unbotD 0) mask address
    (fun i _ hi => h i hi)

theorem quotient_real {B : Nat} (input : Tile .real [B]) (shift total x : ℝ)
    (i : TileIndex [B]) (hi : input.data i = (x : WithBot ℝ)) :
    ((quotient input shift (Real.exp (-shift) * total)).data i).unbotD 0 = Real.exp x / total := by
  change (WithBot.realDiv (WithBot.realExp (WithBot.realSub (input.data i) (shift : WithBot ℝ)))
    ((Real.exp (-shift) * total : ℝ) : WithBot ℝ)).unbotD 0 = _
  rw [hi]
  exact pass2_store_value x shift total

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.Output
