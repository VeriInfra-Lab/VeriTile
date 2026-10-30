import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_SourceChecks
import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_Domain
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Semantics.Eval
import VeriTile.Triton.Semantics.ProgramRewrite

/-! Operational proof of the complete source, assembled from source prefixes.
The initial-load prefix is a phase lemma, not a complete-kernel certificate. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Execution
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def source (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) : Kernel :=
  (original f r stride M N eps p block hBlock).eraseDType.toAlgKernel

def baseStmt (r : RegionName) (name : RegName) (stride : Nat) : Stmt :=
  .assign .ptr [] name (.ptrAdd .nil (.ptrBase r)
    (.mul .nat .nil (.ref .nat [] "row") (.constNat stride)))

/-- Exact original prefix through the initial masked load, including the
optional source pointer bindings. It retains arbitrary initial registers. -/
def loadPrefix (f : Flags) (r : Regions) (stride : Strides) (N block : Nat) : List Stmt :=
  [.assign .nat [] "row" (.programId 0),
   baseStmt r.x "X" stride.x,
   baseStmt r.y "Y" stride.y,
   .ifThen (.constBool f.residual) [baseStmt r.residual "RESIDUAL" stride.residual],
   .ifThen (.constBool f.residualOut) [baseStmt r.residualOut "RESIDUAL_OUT" stride.residualOut],
   .ifThen (.constBool f.x1) [baseStmt r.x1 "X1" stride.x1],
   .ifThen (.constBool f.w1) [baseStmt r.y1 "Y1" stride.y1],
   .assign .nat [] "BLOCK_N_arg" (.constNat block),
   .assign .nat [block] "cols" (.arange block),
   .assign .real [block] "x" (.load .real
     (.ptr (.ptrAdd .scalarL (.ref .ptr [] "X") (.ref .nat [block] "cols")))
     (.maskOther (.lt .nat .scalarR (.ref .nat [block] "cols") (.constNat N))
       (.broadcast (.const 0.0) [block])))]

theorem source_loadPrefix (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) :
    (source f r stride M N eps p block hBlock).body.take 10 = loadPrefix f r stride N block := by
  have hproj : (original f r stride M N eps p block hBlock).toAlgorithm? =
      .ok (original f r stride M N eps p block hBlock).toAlgKernel := rfl
  simp only [source, ComputeKernel.eraseDType, hproj, ComputeKernel.toAlgKernel_fromKernelBody]
  simp only [original, loadPrefix, baseStmt, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, Random.i32ArangeSize, Region.cast,
    List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero]
  rfl

def ptrState (f : Flags) (r : Regions) (stride : Strides) (block : Nat)
    (u : BlockState) : BlockState :=
  let s := u.setReg "row" .nat [] (Tile.scalar u.pid)
  let s := s.setReg "X" .ptr [] (Tile.scalar (r.x, u.pid * stride.x))
  let s := s.setReg "Y" .ptr [] (Tile.scalar (r.y, u.pid * stride.y))
  let s := if f.residual then s.setReg "RESIDUAL" .ptr []
    (Tile.scalar (r.residual, u.pid * stride.residual)) else s
  let s := if f.residualOut then s.setReg "RESIDUAL_OUT" .ptr []
    (Tile.scalar (r.residualOut, u.pid * stride.residualOut)) else s
  let s := if f.x1 then s.setReg "X1" .ptr []
    (Tile.scalar (r.x1, u.pid * stride.x1)) else s
  let s := if f.w1 then s.setReg "Y1" .ptr []
    (Tile.scalar (r.y1, u.pid * stride.y1)) else s
  let s := s.setReg "BLOCK_N_arg" .nat [] (Tile.scalar block)
  s.setReg "cols" .nat [block] (Tile.vec (fun j => j.val))

def loaded (r : RegionName) (stride N block : Nat) (u : BlockState) : Tile .real [block] :=
  ⟨fun j => if j.1.val < N then some (u.readMem r (u.pid * stride + j.1.val)) else some 0.0⟩

def loadState (f : Flags) (r : Regions) (stride : Strides) (N block : Nat)
    (u : BlockState) : BlockState :=
  (ptrState f r stride block u).setReg "x" .real [block] (loaded r.x stride.x N block u)

theorem loadPrefix_exec (f : Flags) (r : Regions) (stride : Strides) (N block : Nat)
    (u : BlockState) :
    stepStmts (loadPrefix f r stride N block) u = some (loadState f r stride N block u) := by
  rcases f with ⟨rms, residual, residualOut, bias, dropout, dropoutMask, rowscale, x1, w1, b1⟩
  cases residual <;> cases residualOut <;> cases x1 <;> cases w1
  all_goals simp [loadPrefix, baseStmt, loadState, ptrState, loaded,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd,
    NumericDType.mul, ComparableDType.lt]

theorem source_initial_load (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (u : BlockState) :
    stepStmts ((source f r stride M N eps p block hBlock).body.take 10) u =
      some (loadState f r stride N block u) := by
  rw [source_loadPrefix, loadPrefix_exec]

/-- No memory field is changed before the first optional dropout mask store. -/
theorem loadState_mem (f : Flags) (r : Regions) (stride : Strides) (N block : Nat)
    (u : BlockState) : (loadState f r stride N block u).mem = u.mem := by
  rcases f with ⟨rms, residual, residualOut, bias, dropout, dropoutMask, rowscale, x1, w1, b1⟩
  cases residual <;> cases residualOut <;> cases x1 <;> cases w1 <;> rfl

theorem loadState_row (f : Flags) (r : Regions) (stride : Strides) (N block : Nat)
    (u : BlockState) :
    (loadState f r stride N block u).regs .nat [] "row" = some (Tile.scalar u.pid) := by
  rcases f with ⟨rms, residual, residualOut, bias, dropout, dropoutMask, rowscale, x1, w1, b1⟩
  cases residual <;> cases residualOut <;> cases x1 <;> cases w1
  all_goals simp [loadState, ptrState]

theorem loadState_cols (f : Flags) (r : Regions) (stride : Strides) (N block : Nat)
    (u : BlockState) :
    (loadState f r stride N block u).regs .nat [block] "cols" =
      some (Tile.vec (fun j => j.val)) := by
  simp [loadState, ptrState]

theorem loadState_x (f : Flags) (r : Regions) (stride : Strides) (N block : Nat)
    (u : BlockState) :
    (loadState f r stride N block u).regs .real [block] "x" =
      some (loaded r.x stride.x N block u) := by
  simp [loadState]

theorem loaded_active (r : RegionName) (stride N block : Nat) (u : BlockState)
    (j : TileIndex [block]) (hj : j.1.val < N) :
    (loaded r stride N block u).data j = some (u.readMem r (u.pid * stride + j.1.val)) := by
  simp [loaded, hj]

theorem source_after_load (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (u : BlockState) :
    exec (source f r stride M N eps p block hBlock) u =
      stepStmts ((source f r stride M N eps p block hBlock).body.drop 10)
        (loadState f r stride N block u) := by
  unfold exec
  conv_lhs => rw [← List.take_append_drop 10 (source f r stride M N eps p block hBlock).body]
  rw [ProgramRewrite.append, source_initial_load, Option.bind_some]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Execution
