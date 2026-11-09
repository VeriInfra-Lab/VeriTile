import bench.tritonbench_g.bmm_chunk_bwd.Proofs.SignedLoop
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.Mathematical_RealStep
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.Mathematical_RealSetup

/-! Execute the complete dynamic contraction loop from the actual setup.
The invariant relates pointers and the accumulator to initial memory, keeps
all other registers and state fields, and counts the source's ceiling-divided
iterations. Partial final blocks are included through the actual load masks.
No loop execution or precomputed product is assumed. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealLoop
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 300000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def dPointers (s₀ : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (c : Nat) : Tile .ptr [BM, BCS] :=
  Tile.mat fun i k => (p.Dout, RealSetup.dBase s₀ p +
    (RealSetup.rows s₀ BM BN p i * p.dColumn + (c * BCS + k.val) * p.dRow))

def aPointers (s₀ : BlockState) (BN BCS : Nat) (p : RealProgram.Parameters)
    (c : Nat) : Tile .ptr [BCS, BN] :=
  Tile.mat fun k j => (p.A, RealSetup.aBase s₀ p +
    ((c * BCS + k.val) * p.aRow + RealSetup.columns s₀ BN p j * p.aColumn))

def product (s₀ : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (i : Fin BM) (j : Fin BN) (k : Nat) : ℝ :=
  s₀.readMem p.Dout (RealSetup.dBase s₀ p + (RealSetup.rows s₀ BM BN p i * p.dColumn + k * p.dRow)) *
  s₀.readMem p.A (RealSetup.aBase s₀ p + (k * p.aRow + RealSetup.columns s₀ BN p j * p.aColumn))

def carry (s₀ : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (c : Nat) (i : Fin BM) (j : Fin BN) : ℝ :=
  if RealSetup.rows s₀ BM BN p i < p.chunk ∧ RealSetup.columns s₀ BN p j < p.K then
    RealMath.partialSum (RealSetup.limit s₀ p) BCS c (product s₀ BM BN p i j)
  else 0

def writes : List RegName := ["cs", "__veritile_load_0", "dout", "__veritile_load_1", "a", "acc", "dout_ptrs", "a_ptrs"]

structure Holds (s₀ : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (c : Nat) (s : BlockState) : Prop where
  count : c ≤ RealMath.blocks (RealSetup.limit s₀ p) BCS
  mem : s.mem = s₀.mem
  undef : s.undef = s₀.undef
  pids : s.pids = s₀.pids
  numPids : s.numPids = s₀.numPids
  fixed : ∀ dtype shape name, name ∉ writes → s.regs dtype shape name =
    (RealSetup.result s₀ BM BN BCS p).regs dtype shape name
  dout : s.regs .ptr [BM, BCS] "dout_ptrs" = some (dPointers s₀ BM BN BCS p c)
  a : s.regs .ptr [BCS, BN] "a_ptrs" = some (aPointers s₀ BN BCS p c)
  acc : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry s₀ BM BN BCS p c i j))

theorem initial (s₀ : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters) :
    Holds s₀ BM BN BCS p 0 (RealSetup.result s₀ BM BN BCS p) := by
  refine ⟨Nat.zero_le _, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [RealSetup.result, RealSetup.arguments, BlockState.setReg]
  · simp [RealSetup.result, RealSetup.arguments, BlockState.setReg]
  · simp [RealSetup.result, RealSetup.arguments, BlockState.setReg]
  · simp [RealSetup.result, RealSetup.arguments, BlockState.setReg]
  · intros; rfl
  · simp [RealSetup.result, dPointers, RealSetup.dPointers]
  · simp [RealSetup.result, aPointers, RealSetup.aPointers]
  · simp [RealSetup.result, carry, RealMath.partial_zero]

theorem d_advance (s₀ : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters) (c : Nat) :
    RealStep.advance (dPointers s₀ BM BN BCS p c) (BCS * p.dRow) =
      dPointers s₀ BM BN BCS p (c + 1) := by
  apply Tile.ext
  rintro ⟨i, k, ⟨⟩⟩
  simp only [RealStep.advance, dPointers, Tile.mat, Prod.mk.injEq, true_and]
  ring

theorem a_advance (s₀ : BlockState) (BN BCS : Nat) (p : RealProgram.Parameters) (c : Nat) :
    RealStep.advance (aPointers s₀ BN BCS p c) (BCS * p.aRow) =
      aPointers s₀ BN BCS p (c + 1) := by
  apply Tile.ext
  rintro ⟨k, j, ⟨⟩⟩
  simp only [RealStep.advance, aPointers, Tile.mat, Prod.mk.injEq, true_and]
  ring

theorem carry_succ (s₀ s : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (c : Nat) (hm : s.mem = s₀.mem) (i : Fin BM) (j : Fin BN) :
    carry s₀ BM BN BCS p c i j +
      RealDot.value s (dPointers s₀ BM BN BCS p c) (aPointers s₀ BN BCS p c)
        (RealSetup.rows s₀ BM BN p) (RealSetup.columns s₀ BN p) p.chunk p.K
        (RealSetup.limit s₀ p) c i j = carry s₀ BM BN BCS p (c + 1) i j := by
  simp only [carry, RealDot.value]
  split
  · rw [RealMath.partial_succ]
    congr 1
    apply Finset.sum_congr rfl
    intro k _
    simp only [dPointers, aPointers, Tile.mat, product, BlockState.readMem, hm]
  · simp


theorem iteration (s₀ : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (c : Nat) (s : BlockState) (hc : c < RealMath.blocks (RealSetup.limit s₀ p) BCS)
    (h : Holds s₀ BM BN BCS p c s) :
    ∃ t, stepStmts (RealLoads.code BM BN BCS p) (s.setReg "cs" .int [] (Tile.scalar (c : Int))) = some t ∧
      Holds s₀ BM BN BCS p (c + 1) t := by
  let entry := s.setReg "cs" .int [] (Tile.scalar (c : Int))
  have hr : entry.regs .int [BM] "offs_m" = some (Tile.vec fun i => (RealSetup.rows s₀ BM BN p i : Int)) := by
    have hv := h.fixed .int [BM] "offs_m" (by decide)
    simpa [entry, RealSetup.result] using hv
  have hn : entry.regs .int [BN] "offs_n" = some (Tile.vec fun j => (RealSetup.columns s₀ BN p j : Int)) := by
    have hv := h.fixed .int [BN] "offs_n" (by decide)
    simpa [entry, RealSetup.result] using hv
  have hk : entry.regs .int [BCS] "offs_cs" = some (Tile.vec fun k => (k.val : Int)) := by
    have hv := h.fixed .int [BCS] "offs_cs" (by decide)
    simpa [entry, RealSetup.result] using hv
  have hl : entry.regs .int [] "chunk_size_limit" = some (Tile.scalar (RealSetup.limitInt s₀ p)) := by
    have hv := h.fixed .int [] "chunk_size_limit" (by decide)
    simpa [entry, RealSetup.result] using hv
  have he := RealStep.executes BM BN BCS entry p (dPointers s₀ BM BN BCS p c)
    (aPointers s₀ BN BCS p c) (RealSetup.rows s₀ BM BN p) (RealSetup.columns s₀ BN p)
    (RealSetup.limitInt s₀ p) c (carry s₀ BM BN BCS p c)
    (by simpa [entry] using h.dout) (by simpa [entry] using h.a)
    hr hn hk hl (by simp [entry]) (by simpa [entry] using h.acc)
  refine ⟨_, he, Nat.succ_le_of_lt hc, h.mem, h.undef, h.pids, h.numPids, ?_, ?_, ?_, ?_⟩
  · intro dtype shape name hname
    have hx : name ∉ ["__veritile_load_0", "dout", "__veritile_load_1", "a", "acc", "dout_ptrs", "a_ptrs"] := by
      have hs := hname
      simp only [writes, List.mem_cons, not_or] at hs
      simpa only [List.mem_cons, not_or] using hs.2
    rw [RealStep.register_frame _ _ _ _ _ _ _ _ _ _ _ _ hx]
    have hcs : name ≠ "cs" := by
      have hs := hname
      simp only [writes, List.mem_cons, not_or] at hs
      exact hs.1
    simpa [entry, BlockState.setReg, hcs] using h.fixed dtype shape name hname
  · simp [RealStep.result, d_advance]
  · simp [RealStep.result, a_advance]
  · simp only [RealStep.result, RealSetup.limitInt_toNat, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, BlockState.setReg_same]
    congr 1
    apply Tile.ext
    rintro ⟨i,j,⟨⟩⟩
    exact congrArg some (carry_succ s₀ entry BM BN BCS p c h.mem i j)


/-- The source's dynamic contraction loop, with its actual register bound. -/
def statement (BM BN BCS : Nat) (p : RealProgram.Parameters) : Stmt :=
  .forRangeInt "cs" (.constInt 0) (RealProgram.loopBound BCS) (.constNat 1)
    (RealLoads.code BM BN BCS p)

def stop (s₀ : BlockState) (BCS : Nat) (p : RealProgram.Parameters) : Int :=
  (RealSetup.limitInt s₀ p + ((BCS : Int) - 1)).tdiv BCS

@[simp] theorem stop_count (s₀ : BlockState) (BCS : Nat) (p : RealProgram.Parameters) :
    (stop s₀ BCS p).toNat = RealMath.blocks (RealSetup.limit s₀ p) BCS := by
  simp only [stop, IntegerIndices.signed_blocks, RealSetup.limitInt_toNat]

theorem bound_eval (s₀ : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters) :
    evalOp (RealProgram.loopBound BCS) (RealSetup.result s₀ BM BN BCS p) =
      some (Tile.scalar (stop s₀ BCS p)) := by
  rw [RealProgram.loopBound, eval_truncDiv]
  simp only [evalOp.eq_def, RealSetup.chunk_limit, bind, Option.bind_some]
  rfl

theorem executes (s₀ : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters) :
    ∃ t, stepStmt (statement BM BN BCS p) (RealSetup.result s₀ BM BN BCS p) = some t ∧
      Holds s₀ BM BN BCS p (RealMath.blocks (RealSetup.limit s₀ p) BCS) t := by
  have resolve : stepStmt (statement BM BN BCS p) (RealSetup.result s₀ BM BN BCS p) =
      stepForRangeIntAux "cs" 0 (stop s₀ BCS p) 1
        (RealLoads.code BM BN BCS p) (RealSetup.result s₀ BM BN BCS p) := by
    rw [statement, stepForRangeIntAux.forRangeInt_unfold, bound_eval s₀ BM BN BCS p]
    simp only [evalOp.eq_def, bind, Option.bind_some, Tile.scalar]
  rw [resolve]
  have h := SignedLoop.executes "cs" (stop s₀ BCS p) (RealLoads.code BM BN BCS p)
    (Holds s₀ BM BN BCS p)
    (fun c s hc hp => iteration s₀ BM BN BCS p c s (by simpa only [stop_count] using hc) hp)
    0 (RealSetup.result s₀ BM BN BCS p) (Nat.zero_le _) (initial s₀ BM BN BCS p)
  simpa only [Int.natCast_zero, stop_count] using h

/-- Readout is a sum over the initial memory, including a partial last block. -/
theorem full_sum (s₀ s : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (hBCS : 0 < BCS)
    (h : Holds s₀ BM BN BCS p (RealMath.blocks (RealSetup.limit s₀ p) BCS) s) :
    s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some
      (if RealSetup.rows s₀ BM BN p i < p.chunk ∧ RealSetup.columns s₀ BN p j < p.K then
        ∑ k ∈ Finset.range (RealSetup.limit s₀ p), product s₀ BM BN p i j k else 0)) := by
  rw [h.acc]
  simp only [carry, RealMath.partial_complete _ _ hBCS]

/-- The canonical source reaches its entire prologue and signed loop for
either value of the retired selector. No pre-populated register is assumed. -/
theorem prefix_executes (swapped : Bool) (s₀ : BlockState) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) :
    ∃ t, stepStmts
      (RealProgram.realPart (RealProgram.parts swapped BM BN BCS p).before ++
        [.forRangeInt "cs" (.constInt 0) (RealProgram.loopBound BCS) (.constNat 1)
          (RealProgram.realPart (RealProgram.parts swapped BM BN BCS p).body)]) s₀ = some t ∧
      Holds s₀ BM BN BCS p (RealMath.blocks (RealSetup.limit s₀ p) BCS) t := by
  obtain ⟨t, he, ht⟩ := executes s₀ BM BN BCS p
  refine ⟨t, ?_, ht⟩
  rw [RealSetup.original_candidate_setup, RealLoads.matches_source]
  rw [stepStmts.append_some (RealSetup.executes s₀ BM BN BCS p)]
  dsimp only [statement] at he
  rw [stepStmts.cons_some he]
  exact stepStmts.nil

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealLoop
