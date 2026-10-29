import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceBackwardCorrespondence
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

/-- Proof-only extraction of the actual signed loop; no alternate source. -/
def forwardLoop (p : BackwardLayout) (length : Nat) : List Stmt :=
  match ((forwardBody p length)[9]? : Option Stmt) with
  | some (.forRangeInt _ _ _ _ body) => body
  | _ => []

def backwardPost (p : BackwardLayout) (length : Nat) : List Stmt :=
  (forwardBody p length).drop 10

theorem forward_decomposition (p : BackwardLayout) (length : Nat) :
    forwardBody p length = forwardPrefix p length ++
      [.forRangeInt "i" (.constInt 0) (.ref .int [] "length") (.constNat 1)
        (forwardLoop p length)] ++ backwardPost p length := by
  simp only [backwardPost, forwardPrefix, forwardLoop, forwardBody, Kernels.backwardTyped,
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
structure BackwardRelated (p : BackwardLayout) (length : Nat) (actual historical : BlockState) : Prop where
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
  carry : ∃ v : Tile .real [p.B],
    actual.regs .real [p.B] "grad_s" = some v ∧ historical.regs .real [p.B] "grad_s" = some v
  gradient : ∃ v : Tile .real [p.B],
    actual.regs .real [p.B] "grad_Lambda" = some v ∧ historical.regs .real [p.B] "grad_Lambda" = some v
  transition : ∃ v : Tile .real [p.B],
    actual.regs .real [p.B] "Lambda" = some v ∧ historical.regs .real [p.B] "Lambda" = some v

theorem prelude_related (p : BackwardLayout) (length : Nat) (u : BlockState) :
    BackwardRelated p length (forwardPrelude p length u).val (unsignedPrelude p u).val := by
  have hi := forwardPrelude_indices p length u
  refine ⟨rfl, rfl, forwardPrelude_length p length u, hi.1, hi.2.1, ?_, ?_, ?_, ?_, ?_⟩
  · refine ⟨⟨columns p u⟩, hi.2.2.1, ?_⟩
    simp only [unsignedPrelude, BlockState.setReg, ↓reduceDIte, ↓reduceIte, castTile_self]
    rfl
  · refine ⟨⟨fun i => decide (columns p u i < p.batch * p.dim)⟩, hi.2.2.2, ?_⟩
    simp only [unsignedPrelude, BlockState.setReg, ↓reduceDIte, ↓reduceIte, castTile_self]
    rfl
  · obtain ⟨v, hv⟩ : ∃ v, (unsignedPrelude p u).val.regs .real [p.B] "grad_s" = some v := by
      simp only [unsignedPrelude, BlockState.setReg, ↓reduceDIte, ↓reduceIte, castTile_self]
      exact ⟨_, rfl⟩
    exact ⟨v, (prelude_carry p length u).trans hv, hv⟩
  · obtain ⟨v, hv⟩ : ∃ v, (unsignedPrelude p u).val.regs .real [p.B] "grad_Lambda" = some v := by
      simp only [unsignedPrelude, BlockState.setReg, ↓reduceDIte, ↓reduceIte, castTile_self]
      exact ⟨_, rfl⟩
    exact ⟨v, (prelude_gradient p length u).trans hv, hv⟩
  · obtain ⟨v, hv⟩ : ∃ v, (unsignedPrelude p u).val.regs .real [p.B] "Lambda" = some v := by
      simp only [unsignedPrelude, BlockState.setReg, ↓reduceDIte, ↓reduceIte, castTile_self]
      exact ⟨_, rfl⟩
    exact ⟨v, (prelude_transition p length u).trans hv, hv⟩

#axiomsClean forward_decomposition
#axiomsClean prelude_related
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceBackward
