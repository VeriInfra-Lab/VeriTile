import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.ActualRow
import bench.tritonbench_g.rms_norm_triton.Proofs.Observed

/-! The current graph reads epsilon from its actual FP32 scalar argument.
Native epsilon is an input word; no cast-shaped literal is used as its marker.
The historical graph interpretation remains available for old model theorems. -/
namespace VeriTile.Bench.Repairs.RmsNormTriton.RuntimeObserved
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RmsNormTriton Whole Observed
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem epsilon_word (M : NativeModel α) (profile : ReductionFamily) (lane : Nat) :
    interpret M profile lane (.parameter "eps") = M.epsilon := rfl

theorem source_graph_lowering (M : NativeModel α) (profile : ReductionFamily)
    (c : Layout) (xs ws : Nat → α) (i : Nat) :
    interpret M profile i (ActualRow.rowValue graphAlgebra c (.parameter "eps") (fun j => .input (xs j)) (fun j => .input (ws j)) i) =
      reference M profile c.N c.B xs ws i := by
  simp only [interpret,ActualRow.rowValue,ActualRow.inputTile,graphAlgebra,graphSum,lower,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,reference,id_eq]
  have squares : (fun j : Fin c.B => multiplyInfo M
      (lower M profile i (if j.val < c.N then .input (xs j.val) else .literal (some .fp32) .real 0))
      (lower M profile i (if j.val < c.N then .input (xs j.val) else .literal (some .fp32) .real 0))) =
      (fun j : Fin c.B => multiplyInfo M
        ⟨if j.val < c.N then xs j.val else M.literal 0,none,none,Bool.false⟩
        ⟨if j.val < c.N then xs j.val else M.literal 0,none,none,Bool.false⟩) := by
    funext j
    split_ifs <;> rfl
  norm_num only
  rw [squares]

def entryGraph (entry : Memory α) : Memory (Term α) := fun allocation offset => .input (entry allocation offset)

/-- The physical output address determines the logical lane/root. This matters
for native butterfly FMA contractions; no uniform rounded row sum is assumed. -/
def projectMemory (M : NativeModel α) (profile : ReductionFamily) (N base : Nat)
    (memory : Memory (Term α)) : Memory α := fun allocation offset =>
  interpret M profile ((offset-base)%N) (memory allocation offset)


#axiomsClean epsilon_word
#axiomsClean source_graph_lowering
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsNormTriton.RuntimeObserved
