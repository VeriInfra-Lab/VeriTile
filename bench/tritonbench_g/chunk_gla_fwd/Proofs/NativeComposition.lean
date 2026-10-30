import bench.tritonbench_g.chunk_gla_fwd.Proofs.NativeInstructions

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions
open VeriTile Triton FP.Structural
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000

def isaTF32Step {α : Type} (I : ISA α) (k row col block : Nat)
    (a b : Fin k → α) (acc : α) : α :=
  I.mmaTF32 ⟨row%16,Nat.mod_lt _ (by decide)⟩ ⟨col%8,Nat.mod_lt _ (by decide)⟩
    (fun e => if h : block*8+e.val<k then a ⟨block*8+e.val,h⟩ else I.literal (some .fp32) .real 0)
    (fun e => if h : block*8+e.val<k then b ⟨block*8+e.val,h⟩ else I.literal (some .fp32) .real 0) acc

theorem tf32_step_eval {α : Type} (I : ISA α) (k row col block : Nat)
    (a b : Fin k → Graph) (acc : Graph) :
    (tf32Step k row col block a b acc).eval I=
      isaTF32Step I k row col block (fun j => (a j).eval I) (fun j => (b j).eval I) (acc.eval I) := by
  simp only [tf32Step,Graph.eval,isaTF32Step]
  congr 1 <;> funext j <;> split <;> rfl

theorem tf32_fold_eval {α : Type} (I : ISA α) (k row col : Nat)
    (a b : Fin k → Graph) (blocks : List Nat) (seed : Graph) :
    (blocks.foldl (fun acc block => tf32Step k row col block a b acc) seed).eval I=
      blocks.foldl (fun acc block => isaTF32Step I k row col block
        (fun j => (a j).eval I) (fun j => (b j).eval I) acc) (seed.eval I) := by
  induction blocks generalizing seed with
  | nil => rfl
  | cons block blocks ih => simpa only [List.foldl_cons,tf32_step_eval] using ih (tf32Step k row col block a b seed)

theorem tf32_complete_chain {α : Type} (I : ISA α) (k row col : Nat)
    (a b : Fin k → Graph) (seed : Graph) :
    (tf32Dot k row col a b seed).eval I=
      (List.range ((k+7)/8)).foldl (fun acc block => isaTF32Step I k row col block
        (fun j => (a j).eval I) (fun j => (b j).eval I) acc) (seed.eval I) :=
  tf32_fold_eval I k row col a b _ seed

theorem ieee_fold_eval {α : Type} (I : ISA α) (k : Nat) (a b : Fin k → Graph)
    (steps : List (Fin k)) (seed : Graph) :
    (steps.foldl (fun acc e => Graph.fma (a e) (b e) acc) seed).eval I=
      steps.foldl (fun acc e => I.fma ((a e).eval I) ((b e).eval I) acc) (seed.eval I) := by
  induction steps generalizing seed with
  | nil => rfl
  | cons e steps ih => simpa only [List.foldl_cons,Graph.eval] using ih (.fma (a e) (b e) seed)

theorem ieee_complete_chain {α : Type} (I : ISA α) (k : Nat)
    (a b : Fin k → Graph) (seed : Graph) :
    (ieeeDot k a b seed).eval I=
      (List.finRange k).foldl (fun acc e => I.fma ((a e).eval I) ((b e).eval I) acc) (seed.eval I) :=
  ieee_fold_eval I k a b _ seed

/-- Native M/N block ownership is independent of warp/register placement.
Small V uses only the in-range result lanes of an 8-column instruction tile. -/
theorem matrix_lane_owner (m n row col : Nat) (hr : row<m) (hc : col<n) :
    row/16<(m+15)/16 ∧ col/8<(n+7)/8 ∧ row%16<16 ∧ col%8<8 ∧
      row/16*16+row%16=row ∧ col/8*8+col%8=col := by omega

theorem matrix_lane_unique (blockM blockN laneM laneN row col : Nat)
    (hm : laneM<16) (hn : laneN<8)
    (hr : blockM*16+laneM=row) (hc : blockN*8+laneN=col) :
    blockM=row/16 ∧ blockN=col/8 ∧ laneM=row%16 ∧ laneN=col%8 := by omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions
