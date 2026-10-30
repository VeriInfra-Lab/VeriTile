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

/-- Ordered recurrence for the actual integer call argument N, including
its native conversion and the compiler's integer-one specialization. -/
def reference (M : NativeModel α) (profile : ReductionFamily) (N B : Nat)
    (xs ws : Nat → α) (i : Nat) : α :=
  let inputs : Fin B → Lowered α := fun j =>
    let x := if j.val < N then xs j.val else M.literal 0
    ⟨x,none,none,Bool.false⟩
  let squares := fun j => multiplyInfo M (inputs j) (inputs j)
  let total := reduceInfo M profile B i squares
  let variance := divideInfo M total (naturalInfo M N)
  let radicand := addEpsilonInfo M variance
  let root : Lowered α := ⟨M.sqrt radicand.value,none,none,Bool.false⟩
  let rrms := divideInfo M (literalInfo M 1) root
  (multiplyInfo M (multiplyInfo M ⟨xs i,none,none,Bool.false⟩ rrms)
    ⟨ws i,none,none,Bool.false⟩).value


/-- The actual runtime-integer source retains the specialized singleton FMA. -/
theorem reference_singleton (M : NativeModel α) (profile : ReductionFamily) (xs ws : Nat → α) :
    reference M profile 1 1 xs ws 0 =
      M.mul (M.mul (M.div (M.literal 1) (M.sqrt (M.fma (xs 0) (xs 0) M.epsilon))) (xs 0)) (ws 0) := by
  simp [reference,naturalInfo,reduceInfo,divideInfo,literalInfo,multiplyInfo,addEpsilonInfo]

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


#axiomsClean reference_singleton
#axiomsClean epsilon_word
#axiomsClean source_graph_lowering
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsNormTriton.RuntimeObserved
