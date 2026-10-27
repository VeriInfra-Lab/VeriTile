import bench.repairs.tritonbench_g.softmax_optimize.Proofs.OutputBodies

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.OutputProperties
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SoftmaxOptimize
open SignedTransport Input Output OutputBodies
noncomputable section
set_option maxHeartbeats 4000000

def metadata (s : BlockState) := (s.regs, s.pids, s.numPids, s.undef)

theorem scatter_metadata {B : Nat} (s : BlockState) (Y : RegionName)
    (offsets : TileIndex [B] → Nat) (values : Tile .real [B]) (mask : TileIndex [B] → Bool) :
    metadata (scatter s Y offsets values mask) = metadata s := by
  unfold scatter
  generalize TileShape.allIndices [B] = items
  induction items generalizing s with
  | nil => rfl
  | cons head tail ih =>
    rw [List.foldl_cons, ih]
    split <;> rfl

theorem scatter_regs {B : Nat} (s : BlockState) (Y : RegionName)
    (offsets : TileIndex [B] → Nat) (values : Tile .real [B]) (mask : TileIndex [B] → Bool) :
    (scatter s Y offsets values mask).regs = s.regs :=
  congrArg Prod.fst (scatter_metadata s Y offsets values mask)

def kept (name : RegName) : Prop := name ∈ ["pid_m", "N", "prev_multiple", "m", "z"]

theorem full_keeps (s : BlockState) (Y X : RegionName) (N B start row : Nat)
    (shift denominator : WithBot ℝ) (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hn : kept name) :
    (fullResult s Y X N B start row shift denominator).regs dtype shape name = s.regs dtype shape name := by
  unfold fullResult
  rw [scatter_regs]
  simp only [kept, List.mem_cons, List.not_mem_nil, or_false] at hn
  rcases hn with rfl | rfl | rfl | rfl | rfl <;>
    simp only [fullBeforeStore, arithmeticState, fullLoaded, prefixIntState,
      BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]

theorem tail_keeps (s : BlockState) (Y X : RegionName) (N B start row : Nat)
    (shift denominator : WithBot ℝ) (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hn : kept name) :
    (tailResult s Y X N B start row shift denominator).regs dtype shape name = s.regs dtype shape name := by
  unfold tailResult
  rw [scatter_regs]
  simp only [kept, List.mem_cons, List.not_mem_nil, or_false] at hn
  rcases hn with rfl | rfl | rfl | rfl | rfl <;>
    simp only [tailBeforeStore, arithmeticState, tailLoaded, maskedPrefix, prefixIntState,
      BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]

theorem full_metadata (s : BlockState) (Y X : RegionName) (N B start row : Nat)
    (shift denominator : WithBot ℝ) :
    (fullResult s Y X N B start row shift denominator).pids = s.pids ∧
    (fullResult s Y X N B start row shift denominator).numPids = s.numPids ∧
    (fullResult s Y X N B start row shift denominator).undef = s.undef := by
  have h := congrArg Prod.snd (scatter_metadata
    (fullBeforeStore s Y X N B start row shift denominator) Y
    (fun i => row * N + (start + i.1.val))
    (quotient (fullLoad s X N B start row) shift denominator) (fun _ => Bool.true))
  exact ⟨congrArg Prod.fst h, congrArg (fun p => p.2.1) h, congrArg (fun p => p.2.2) h⟩

theorem tail_metadata (s : BlockState) (Y X : RegionName) (N B start row : Nat)
    (shift denominator : WithBot ℝ) :
    (tailResult s Y X N B start row shift denominator).pids = s.pids ∧
    (tailResult s Y X N B start row shift denominator).numPids = s.numPids ∧
    (tailResult s Y X N B start row shift denominator).undef = s.undef := by
  have h := congrArg Prod.snd (scatter_metadata
    (tailBeforeStore s Y X N B start row shift denominator) Y
    (fun i => row * N + (start + i.1.val))
    (quotient (tailLoad s X N B start row) shift denominator) (tailMask N B start).data)
  exact ⟨congrArg Prod.fst h, congrArg (fun p => p.2.1) h, congrArg (fun p => p.2.2) h⟩

theorem offsets_injective (N B start row : Nat) :
    Function.Injective (fun i : TileIndex [B] => row * N + (start + i.1.val)) := by
  intro a b h
  exact BlockState.tileIndex1d_base_offset_injective (row * N + start)
    (by simpa only [Nat.add_assoc] using h)

theorem full_readback (s : BlockState) (Y X : RegionName) (N B start row : Nat)
    (shift denominator : WithBot ℝ) (i : TileIndex [B]) :
    (fullResult s Y X N B start row shift denominator).readMem Y (row * N + (start + i.1.val)) =
      ((quotient (fullLoad s X N B start row) shift denominator).data i).unbotD 0 := by
  unfold fullResult scatter
  rw [BlockState.scatter_readback_masked_nd _ _ _ _ (offsets_injective N B start row) i]
  rfl

theorem tail_readback (s : BlockState) (Y X : RegionName) (N B start row : Nat)
    (shift denominator : WithBot ℝ) (i : TileIndex [B]) (hi : start + i.1.val < N) :
    (tailResult s Y X N B start row shift denominator).readMem Y (row * N + (start + i.1.val)) =
      ((quotient (tailLoad s X N B start row) shift denominator).data i).unbotD 0 := by
  unfold tailResult scatter
  rw [BlockState.scatter_readback_masked_nd _ _ _ _ (offsets_injective N B start row) i]
  simp only [tailMask, hi, decide_true, ↓reduceIte]

theorem full_frame (s : BlockState) (Y X : RegionName) (N B start row : Nat)
    (shift denominator : WithBot ℝ) (region : RegionName) (address : Nat)
    (outside : ∀ i : TileIndex [B], ¬ (Y = region ∧ row * N + (start + i.1.val) = address)) :
    (fullResult s Y X N B start row shift denominator).mem region address = s.mem region address :=
  scatter_frame _ _ _ _ _ region address (fun i _ => outside i)

theorem tail_frame (s : BlockState) (Y X : RegionName) (N B start row : Nat)
    (shift denominator : WithBot ℝ) (region : RegionName) (address : Nat)
    (outside : ∀ i : TileIndex [B], start + i.1.val < N →
      ¬ (Y = region ∧ row * N + (start + i.1.val) = address)) :
    (tailResult s Y X N B start row shift denominator).mem region address = s.mem region address :=
  scatter_frame _ _ _ _ _ region address
    (fun i hi => outside i (by simpa only [tailMask, decide_eq_true_eq] using hi))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.OutputProperties
