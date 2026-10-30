import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexCorrespondence
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplex
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

/-- Proof-only extraction of the actual signed loop; no alternate source. -/
def forwardLoop (p : ForwardLayout) (length : Nat) : List Stmt :=
  match ((forwardBody p length)[10]? : Option Stmt) with
  | some (.forRangeInt _ _ _ _ body) => body
  | _ => []

theorem forward_decomposition (p : ForwardLayout) (length : Nat) :
    forwardBody p length = forwardPrefix p length ++
      [.forRangeInt "t" (.constInt 0) (.ref .int [] "length") (.constNat 1)
        (forwardLoop p length)] := by
  simp only [forwardPrefix, forwardLoop, forwardBody, Kernels.complexForwardTyped,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, List.map_cons, List.map_nil,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
  simp only [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  rfl

/-- Only the registers used by the next iteration are related. The two
programs retain their distinct signed/natural index registers. -/
structure ForwardRelated (p : ForwardLayout) (length : Nat) (actual historical : BlockState) : Prop where
  memory : actual.mem = historical.mem
  pids : actual.pids = historical.pids
  lengthReg : actual.regs .int [] "length" = some (Tile.scalar (length : Int))
  batchReg : actual.regs .int [] "batch_size" = some (Tile.scalar (p.batch : Int))
  dimReg : actual.regs .int [] "dim" = some (Tile.scalar (p.dim : Int))
  columns : ∃ c : Tile .nat [p.B],
    actual.regs .int [p.B] "col_offsets" = some ⟨fun i => (c.data i : Int)⟩ ∧
    historical.regs .nat [p.B] "col_offsets" = some c
  mask : ∃ m : Tile .bool [p.B],
    actual.regs .bool [p.B] "mask" = some m ∧ historical.regs .bool [p.B] "mask" = some m
  realState : ∃ v : Tile .real [p.B],
    actual.regs .real [p.B] "s_real" = some v ∧ historical.regs .real [p.B] "s_real" = some v
  imagState : ∃ v : Tile .real [p.B],
    actual.regs .real [p.B] "s_imag" = some v ∧ historical.regs .real [p.B] "s_imag" = some v
  realTransition : ∃ v : Tile .real [p.B],
    actual.regs .real [p.B] "lambda_real" = some v ∧ historical.regs .real [p.B] "lambda_real" = some v
  imagTransition : ∃ v : Tile .real [p.B],
    actual.regs .real [p.B] "lambda_imag" = some v ∧ historical.regs .real [p.B] "lambda_imag" = some v

theorem prelude_related (p : ForwardLayout) (length : Nat) (u : BlockState) :
    ForwardRelated p length (forwardPrelude p length u).val (unsignedPrelude p u).val := by
  have hi := forwardPrelude_indices p length u
  refine ⟨rfl, rfl, forwardPrelude_length p length u, hi.1, hi.2.1, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine ⟨⟨columns p u⟩, hi.2.2.1, ?_⟩
    simp only [unsignedPrelude, BlockState.setReg, ↓reduceDIte, ↓reduceIte, castTile_self]
    rfl
  · refine ⟨⟨fun i => decide (columns p u i < p.batch * p.dim)⟩, hi.2.2.2, ?_⟩
    simp only [unsignedPrelude, BlockState.setReg, ↓reduceDIte, ↓reduceIte, castTile_self]
    rfl
  · obtain ⟨v, hv⟩ : ∃ v, (unsignedPrelude p u).val.regs .real [p.B] "s_real" = some v := by
      simp only [unsignedPrelude, BlockState.setReg, ↓reduceDIte, ↓reduceIte, castTile_self]
      exact ⟨_, rfl⟩
    exact ⟨v, (prelude_initial_real p length u).trans hv, hv⟩
  · obtain ⟨v, hv⟩ : ∃ v, (unsignedPrelude p u).val.regs .real [p.B] "s_imag" = some v := by
      simp only [unsignedPrelude, BlockState.setReg, ↓reduceDIte, ↓reduceIte, castTile_self]
      exact ⟨_, rfl⟩
    exact ⟨v, (prelude_initial_imag p length u).trans hv, hv⟩
  · obtain ⟨v, hv⟩ : ∃ v, (unsignedPrelude p u).val.regs .real [p.B] "lambda_real" = some v := by
      simp only [unsignedPrelude, BlockState.setReg, ↓reduceDIte, ↓reduceIte, castTile_self]
      exact ⟨_, rfl⟩
    exact ⟨v, (prelude_transition_real p length u).trans hv, hv⟩
  · obtain ⟨v, hv⟩ : ∃ v, (unsignedPrelude p u).val.regs .real [p.B] "lambda_imag" = some v := by
      simp only [unsignedPrelude, BlockState.setReg, ↓reduceDIte, ↓reduceIte, castTile_self]
      exact ⟨_, rfl⟩
    exact ⟨v, (prelude_transition_imag p length u).trans hv, hv⟩

#axiomsClean forward_decomposition
#axiomsClean prelude_related
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplex
