import bench.tritonbench_g.bmm_chunk_bwd.Proofs.Mathematical_RealSetup

/-! The complete post-loop source and real execution of its residual branch.
Load and store share the same active-lane predicate. Inactive residual loads
retain undef, and the source addition order yields carry+residual
in the real interpretation. The final output store is recorded here but its
execution and memory frame are separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealResidual
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 18000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def rowOffsets (BM : Nat) : Op .int [BM] :=
  .add .int .scalarL (.mul .int .nil (.ref .int [] "pid_m") (.constInt BM)) (.castNatToInt (.arange BM))

def columnOffsets (BN : Nat) : Op .int [BN] :=
  .add .int .scalarL (.mul .int .nil (.ref .int [] "pid_n") (.constInt BN)) (.castNatToInt (.arange BN))

def basePointer (region : RegionName) (chunk batch row head : Nat) : Op .ptr [] :=
  .ptrAdd .nil (.ptrBase region) (.castIntToNat
    (.add .int .nil
      (.add .int .nil (.mul .int .nil (.ref .int [] "pid_b") (.castNatToInt (.constNat batch)))
        (.mul .int .nil (.mul .int .nil (.ref .int [] "pid_c") (.castNatToInt (.constNat chunk))) (.castNatToInt (.constNat row))))
      (.mul .int .nil (.ref .int [] "pid_h") (.castNatToInt (.constNat head)))))

def pointers (BM BN : Nat) (base : RegName) (row column : Nat) : Op .ptr [BM, BN] :=
  .ptrAdd .scalarL (.ref .ptr [] base) (.castIntToNat
    (.add .int Broadcast.nil.consL.consR
      (.mul .int .scalarR (.expandDim ⟨1, by simp⟩ (.ref .int [BM] "offs_m")) (.castNatToInt (.constNat row)))
      (.mul .int .scalarR (.expandDim ⟨0, by simp⟩ (.ref .int [BN] "offs_n")) (.castNatToInt (.constNat column)))))

def mask (BM BN K : Nat) : Op .bool [BM, BN] :=
  .boolAnd Broadcast.nil.consL.consR
    (.lt .int .scalarR (.expandDim ⟨1, by simp⟩ (.ref .int [BM] "offs_m"))
      (.ref .int [] "chunk_size_limit"))
    (.lt .int .scalarR (.expandDim ⟨0, by simp⟩ (.ref .int [BN] "offs_n")) (.castNatToInt (.constNat K)))

/-- The retired selector does not introduce a second residual order. -/
def addition (_retiredResidualOrder : Bool) (BM BN : Nat) : Op .real [BM, BN] :=
  .add .real (.consSame (.consSame .nil)) (.ref .real [BM, BN] "acc") (.ref .real [BM, BN] "res")

def residual (swapped : Bool) (BM BN : Nat) (p : RealProgram.Parameters) : List Stmt :=
  [.assign .ptr [] "res_ptr" (basePointer p.Res p.chunk p.rBatch p.rRow p.rHead),
   .assign .ptr [BM, BN] "res_ptrs" (pointers BM BN "res_ptr" p.rRow p.rColumn),
   .assign .real [BM, BN] "__veritile_load_2" (.load .real (.ptr (.ref .ptr [BM, BN] "res_ptrs")) (.mask (mask BM BN p.K))),
   .assign .real [BM, BN] "res" (.ref .real [BM, BN] "__veritile_load_2"),
   .assign .real [BM, BN] "acc" (addition swapped BM BN)]

def code (swapped : Bool) (BM BN : Nat) (p : RealProgram.Parameters) : List Stmt :=
  [.assign .int [BM] "offs_m" (rowOffsets BM),
   .assign .int [BN] "offs_n" (columnOffsets BN),
   .ifThen (.constBool p.hasResidual) (residual swapped BM BN p),
   .assign .real [BM, BN] "db" (.ref .real [BM, BN] "acc"),
   .assign .ptr [] "db_ptr" (basePointer p.Db p.chunk p.oBatch p.oRow p.oHead),
   .assign .ptr [BM, BN] "db_ptrs" (pointers BM BN "db_ptr" p.oRow p.oColumn),
   .store .real [BM, BN] (.ptr (.ref .ptr [BM, BN] "db_ptrs"))
     (.ref .real [BM, BN] "db") (.mask (mask BM BN p.K))]

theorem matches_source (swapped : Bool) (BM BN BCS : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (RealProgram.parts swapped BM BN BCS p).after = code swapped BM BN p := by
  cases swapped
  all_goals
    simp [code, residual, addition, basePointer, pointers, rowOffsets, columnOffsets, mask,
      RealProgram.realPart, RealProgram.parts, RealProgram.source,
      TypedSource.specialize, VeriTile.Bench.Repairs.BmmChunkBwd.Kernels.family,
      VeriTile.Bench.Repairs.BmmChunkBwd.Kernels.sourceKernel,
      VeriTile.Bench.Repairs.BmmChunkBwd.Kernels.dotDType, Source.intRange?, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    try norm_num


def baseValue (chunk batch row head pb pc ph : Nat) : Nat :=
  pb * batch + pc * chunk * row + ph * head

def pointerValues (region : RegionName) (BM BN base row column : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) : Tile .ptr [BM, BN] :=
  Tile.mat fun i j => (region, base + (rows i * row + columns j * column))

theorem base_eval (s : BlockState) (region : RegionName) (chunk batch row head pb pc ph : Nat)
    (hb : s.regs .int [] "pid_b" = some (Tile.scalar (pb : Int)))
    (hc : s.regs .int [] "pid_c" = some (Tile.scalar (pc : Int)))
    (hh : s.regs .int [] "pid_h" = some (Tile.scalar (ph : Int))) :
    evalOp (basePointer region chunk batch row head) s =
      some (Tile.scalar (region, baseValue chunk batch row head pb pc ph)) := by
  simp [basePointer, evalOp.eq_def, hb, hc, hh, bind, Option.bind]
  apply Tile.ext
  intro i
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul, baseValue, Nat.zero_add]
  simp only [← Int.natCast_add, ← Int.natCast_mul, Int.toNat_natCast]

theorem pointers_eval (s : BlockState) (region : RegionName) (BM BN base row column : Nat)
    (baseReg : RegName) (rows : Fin BM → Nat) (columns : Fin BN → Nat)
    (hb : s.regs .ptr [] baseReg = some (Tile.scalar (region, base)))
    (hr : s.regs .int [BM] "offs_m" = some (Tile.vec fun i => (rows i : Int)))
    (hn : s.regs .int [BN] "offs_n" = some (Tile.vec fun j => (columns j : Int))) :
    evalOp (pointers BM BN baseReg row column) s =
      some (pointerValues region BM BN base row column rows columns) := by
  simp [pointers, evalOp.eq_def, hb, hr, hn, bind, Option.bind]
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Tile.mat, Tile.vec, Tile.expandDim,
    Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex,
    NumericDType.add, NumericDType.mul, ← Int.natCast_add, ← Int.natCast_mul,
    Int.toNat_natCast, pointerValues]

theorem mask_eval (s : BlockState) (BM BN K : Nat) (length : Int)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat)
    (hr : s.regs .int [BM] "offs_m" = some (Tile.vec fun i => (rows i : Int)))
    (hn : s.regs .int [BN] "offs_n" = some (Tile.vec fun j => (columns j : Int)))
    (hl : s.regs .int [] "chunk_size_limit" = some (Tile.scalar length)) :
    evalOp (mask BM BN K) s =
      some (Tile.mat fun i j => decide (rows i < length.toNat ∧ columns j < K)) := by
  simp [mask, evalOp.eq_def, hr, hn, hl, bind, Option.bind]
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, Tile.mat, Tile.scalar,
    ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex]
  congr 2 <;> simp only [Int.ofNat_lt]

def residualValue {BM BN : Nat} (s : BlockState) (ptrs : Tile .ptr [BM, BN])
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (length K : Nat) (i : Fin BM) (j : Fin BN) : ℝ :=
  if rows i < length ∧ columns j < K then
    s.readMem (ptrs.data (i,j,PUnit.unit)).1 (ptrs.data (i,j,PUnit.unit)).2
  else s.undef (ptrs.data (i,j,PUnit.unit)).1 (ptrs.data (i,j,PUnit.unit)).2

def residualValues {BM BN : Nat} (s : BlockState) (ptrs : Tile .ptr [BM, BN])
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (length K : Nat) : Tile .real [BM, BN] :=
  Tile.mat fun i j => some (residualValue s ptrs rows columns length K i j)

/-- The residual has no explicit `other`: masked values retain source undef.
The eventual store uses this same predicate, so those lanes are not written. -/
theorem residual_load (s : BlockState) (BM BN K : Nat) (length : Int)
    (ptrs : Tile .ptr [BM, BN]) (rows : Fin BM → Nat) (columns : Fin BN → Nat)
    (hp : s.regs .ptr [BM, BN] "res_ptrs" = some ptrs)
    (hr : s.regs .int [BM] "offs_m" = some (Tile.vec fun i => (rows i : Int)))
    (hn : s.regs .int [BN] "offs_n" = some (Tile.vec fun j => (columns j : Int)))
    (hl : s.regs .int [] "chunk_size_limit" = some (Tile.scalar length)) :
    evalOp (.load .real (.ptr (.ref .ptr [BM, BN] "res_ptrs")) (.mask (mask BM BN K))) s =
      some (residualValues s ptrs rows columns length.toNat K) := by
  rw [evalOp.eq_def]
  simp only [evalOp_ref, hp, mask_eval s BM BN K length rows columns hr hn hl,
    bind, Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [residualValues, residualValue, Tile.mat, decide_eq_true_eq]
  split_ifs <;> simp only [BlockState.readMemValue_real]


theorem addition_eval (swapped : Bool) (s : BlockState) (BM BN : Nat)
    (carry res : Fin BM → Fin BN → ℝ)
    (ha : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry i j)))
    (hr : s.regs .real [BM, BN] "res" = some (Tile.mat fun i j => some (res i j))) :
    evalOp (addition swapped BM BN) s =
      some (Tile.mat fun i j => some (carry i j + res i j)) := by
  cases swapped
  all_goals
    simp [addition, evalOp.eq_def, ha, hr, bind, Option.bind]
    apply Tile.ext
    rintro ⟨i,j,⟨⟩⟩
    simp [Tile.bop, Tile.mat, Broadcast.leftIndex, Broadcast.rightIndex,
      NumericDType.add, _root_.add_comm]

def residualResult (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pb pc ph : Nat) (length : Int) (rows : Fin BM → Nat) (columns : Fin BN → Nat)
    (carry : Fin BM → Fin BN → ℝ) : BlockState :=
  let base := baseValue p.chunk p.rBatch p.rRow p.rHead pb pc ph
  let ptrs := pointerValues p.Res BM BN base p.rRow p.rColumn rows columns
  let t := s.setReg "res_ptr" .ptr [] (Tile.scalar (p.Res, base))
  let t := t.setReg "res_ptrs" .ptr [BM, BN] ptrs
  let t := t.setReg "__veritile_load_2" .real [BM, BN] (residualValues s ptrs rows columns length.toNat p.K)
  let t := t.setReg "res" .real [BM, BN] (residualValues s ptrs rows columns length.toNat p.K)
  t.setReg "acc" .real [BM, BN] (Tile.mat fun i j => some
    (carry i j + residualValue s ptrs rows columns length.toNat p.K i j))

/-- Execute the source residual branch without replacing inactive loads by
zero. The retired selector uses this same current source order. -/
theorem residual_executes (swapped : Bool) (s : BlockState) (BM BN : Nat)
    (p : RealProgram.Parameters) (pb pc ph : Nat) (length : Int)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (carry : Fin BM → Fin BN → ℝ)
    (hb : s.regs .int [] "pid_b" = some (Tile.scalar (pb : Int)))
    (hc : s.regs .int [] "pid_c" = some (Tile.scalar (pc : Int)))
    (hh : s.regs .int [] "pid_h" = some (Tile.scalar (ph : Int)))
    (hr : s.regs .int [BM] "offs_m" = some (Tile.vec fun i => (rows i : Int)))
    (hn : s.regs .int [BN] "offs_n" = some (Tile.vec fun j => (columns j : Int)))
    (hl : s.regs .int [] "chunk_size_limit" = some (Tile.scalar length))
    (ha : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry i j))) :
    stepStmts (residual swapped BM BN p) s =
      some (residualResult s BM BN p pb pc ph length rows columns carry) := by
  let base := baseValue p.chunk p.rBatch p.rRow p.rHead pb pc ph
  let ptrs := pointerValues p.Res BM BN base p.rRow p.rColumn rows columns
  let s₁ := s.setReg "res_ptr" .ptr [] (Tile.scalar (p.Res, base))
  let s₂ := s₁.setReg "res_ptrs" .ptr [BM, BN] ptrs
  let sl := s₂.setReg "__veritile_load_2" .real [BM, BN] (residualValues s ptrs rows columns length.toNat p.K)
  let s₃ := sl.setReg "res" .real [BM, BN] (residualValues s ptrs rows columns length.toNat p.K)
  have hp := base_eval s p.Res p.chunk p.rBatch p.rRow p.rHead pb pc ph hb hc hh
  have hps : evalOp (pointers BM BN "res_ptr" p.rRow p.rColumn) s₁ = some ptrs := by
    apply pointers_eval s₁ p.Res BM BN base p.rRow p.rColumn "res_ptr" rows columns
    · simp [s₁]
    · simpa [s₁] using hr
    · simpa [s₁] using hn
  have hload : evalOp (.load .real (.ptr (.ref .ptr [BM, BN] "res_ptrs"))
      (.mask (mask BM BN p.K))) s₂ = some (residualValues s ptrs rows columns length.toNat p.K) := by
    apply residual_load s₂ BM BN p.K length ptrs rows columns
    · simp [s₂]
    · simpa [s₂, s₁] using hr
    · simpa [s₂, s₁] using hn
    · simpa [s₂, s₁] using hl
  have hadd : evalOp (addition swapped BM BN) s₃ = some (Tile.mat fun i j => some
      (carry i j + residualValue s ptrs rows columns length.toNat p.K i j)) := by
    apply addition_eval swapped s₃ BM BN carry (residualValue s ptrs rows columns length.toNat p.K)
    · simpa [s₃, sl, s₂, s₁] using ha
    · simp [s₃, residualValues]
  dsimp only [s₃, sl, s₂, s₁, ptrs, base] at hps hload hadd
  simp only [residual, stepStmts, stepStmt, hp, hps, hload, hadd,
    evalOp_ref, BlockState.setReg_same, bind, Option.bind_some]
  rfl


/-- Residual evaluation changes registers only. -/
theorem residual_frame (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pb pc ph : Nat) (length : Int) (rows : Fin BM → Nat) (columns : Fin BN → Nat)
    (carry : Fin BM → Fin BN → ℝ) :
    {residualResult s BM BN p pb pc ph length rows columns carry with regs := s.regs} = s := by
  cases s
  rfl

theorem residual_register_frame (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pb pc ph : Nat) (length : Int) (rows : Fin BM → Nat) (columns : Fin BN → Nat)
    (carry : Fin BM → Fin BN → ℝ) (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hn : name ∉ ["res_ptr", "res_ptrs", "__veritile_load_2", "res", "acc"]) :
    (residualResult s BM BN p pb pc ph length rows columns carry).regs dtype shape name =
      s.regs dtype shape name := by
  simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at hn
  rcases hn with ⟨hb, hp, hl, hr, ha⟩
  simp [residualResult, BlockState.setReg, hb, hp, hl, hr, ha]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealResidual
