import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.FPBackwardStore
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
theorem scatterMemoryReference {ι : Type} (l : List ι) (r : RegionName) (off : ι → Nat) (v : ι → Cell α) (s : State α) :
    (l.foldl (fun t i => t.write r (off i) (v i)) s).mem=
      l.foldl (fun m i => writeMemory m r (off i) (v i)) s.mem := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih => simp only [List.foldl_cons,ih]; rfl
@[simp] theorem scatterUndef {ι : Type} (l : List ι) (r : RegionName) (off : ι → Nat) (v : ι → Cell α) (s : State α) :
    (l.foldl (fun t i => t.write r (off i) (v i)) s).undef=s.undef := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih => exact (ih _).trans rfl

def backwardBodyResult (P : Primitives α) (p : BackwardParameters) (initial current : State α)
    (c j : Nat) (outerMem : Memory α) (st : BackwardInner α p) : State α :=
  backwardAdvancePointers p initial
    (backwardStoreResult P p initial
      (backwardGradientResult P p initial (backwardScoreResult P p initial current c j outerMem st) c j outerMem st)
      c j outerMem st) j

theorem backwardBodyExecution (rest : Algebra α) (P : Primitives α) (p : BackwardParameters)
    (initial current : State α) (c j : Nat) (outerMem : Memory α) (st : BackwardInner α p)
    (h : BackwardInnerInvariant p initial c j outerMem st current) :
    run (backend rest P) (backwardInnerBody p) (current.setReg "start_m" .nat [] (fun _ => j*p.blockM))=
      some (backwardBodyResult P p initial current c j outerMem st) := by
  let a := backwardScoreResult P p initial current c j outerMem st
  let b := backwardGradientResult P p initial a c j outerMem st
  let d := backwardStoreResult P p initial b c j outerMem st
  have hs := backwardScoreExecution rest P p initial current c j outerMem st h
  have hg := backwardGradientExecution rest P p initial a c j outerMem st
    (backwardScoreInvariant P p initial current c j outerMem st h)
    (by simp [a,backwardScoreResult]) (by simp [a,backwardScoreResult])
  have hd := backwardStoreExecution rest P p initial b c j outerMem st
    (by simp [b,a,backwardGradientResult,backwardScoreResult,h.memory])
    (by simp [b,a,backwardGradientResult,backwardScoreResult,h.dqPtr])
    (by simp [b,a,backwardGradientResult,backwardScoreResult,h.key])
    (by simp [b,a,backwardGradientResult,backwardScoreResult])
    (by simp [b,a,backwardGradientResult,backwardScoreResult,h.dk])
    (by simp [b,backwardGradientResult])
  have hp := backwardAdvanceExecution rest P p initial d j
    (by simp [d,b,a,backwardStoreResult,backwardStoreRegisters,backwardGradientResult,backwardScoreResult,h.dqDeadPtrs])
    (by simp [d,b,a,backwardStoreResult,backwardStoreRegisters,backwardGradientResult,backwardScoreResult,h.qPtr])
    (by simp [d,b,a,backwardStoreResult,backwardStoreRegisters,backwardGradientResult,backwardScoreResult,h.doPtr])
    (by simp [d,b,a,backwardStoreResult,backwardStoreRegisters,backwardGradientResult,backwardScoreResult,h.dqPtr])
  have hsplit : backwardInnerBody p=(backwardInnerBody p).take 6 ++
      ((backwardInnerBody p).drop 6).take 6 ++ ((backwardInnerBody p).drop 12).take 4 ++
      (backwardInnerBody p).drop 16 := rfl
  rw [hsplit,run_append,run_append,run_append,hs]
  dsimp only [a] at hg
  dsimp only [b,a] at hd
  dsimp only [d,b,a] at hp
  simp only [Option.bind_some,hg,hd]
  exact hp

theorem backwardBodyInvariant (P : Primitives α) (p : BackwardParameters)
    (initial current : State α) (c j : Nat) (outerMem : Memory α) (st : BackwardInner α p)
    (h : BackwardInnerInvariant p initial c j outerMem st current) :
    BackwardInnerInvariant p initial c (j+1) outerMem
      (backwardInnerAdvance P p initial (backwardRead p initial outerMem p.k c)
        (backwardRead p initial outerMem p.v c) c j st)
      (backwardBodyResult P p initial current c j outerMem st) := by
  constructor
  · constructor
    · constructor <;> simp [backwardBodyResult,backwardAdvancePointers,backwardStoreResult,backwardStoreRegisters,
        backwardGradientResult,backwardScoreResult,h.undef,h.pids,h.hz,h.z,h.h,h.strideZ,h.strideH,h.dqBase]
    all_goals simp [backwardBodyResult,backwardAdvancePointers,backwardStoreResult,backwardStoreRegisters,
      backwardGradientResult,backwardScoreResult,backwardInnerAdvance,scatterMemoryReference,h.memory,h.keyPtr,h.valuePtr,h.dkPtr,h.dvPtr]
  all_goals simp [backwardBodyResult,backwardAdvancePointers,backwardStoreResult,backwardStoreRegisters,
    backwardGradientResult,backwardScoreResult,h.lo,h.rows,h.cols,h.deltaPtr,h.maxPtr,h.key,h.value]

#axiomsClean backwardBodyExecution
#axiomsClean backwardBodyInvariant
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
