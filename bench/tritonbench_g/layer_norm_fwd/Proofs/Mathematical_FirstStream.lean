import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_Dropout

/-! Composition of the first twelve actual source statements. This covers
initial pointers, the complete first input stream, and its optional mask store;
second input, residual, normalization and affine output phases follow it. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Execution
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def loadedValues (r : RegionName) (stride N block : Nat) (u : BlockState) : Fin block → ℝ :=
  fun j => if j.val < N then u.readMem r (u.pid * stride + j.val) else 0

theorem loaded_realTile (r : RegionName) (stride N block : Nat) (u : BlockState) :
    loaded r stride N block u = realTile (loadedValues r stride N block u) := by
  ext j
  by_cases h : j.1.val < N <;> simp [loaded, realTile, loadedValues, h] <;> norm_num

def firstScaledValues (f : Flags) (r : Regions) (stride : Strides) (N block : Nat)
    (u : BlockState) : Fin block → ℝ :=
  fun j => if f.rowscale then loadedValues r.x stride.x N block u j * u.readMem r.rowscale u.pid
    else loadedValues r.x stride.x N block u j

def firstScaledState (f : Flags) (r : Regions) (stride : Strides) (N block : Nat)
    (u : BlockState) : BlockState :=
  scaleState f.rowscale r.rowscale u.pid block (loadedValues r.x stride.x N block u)
    (loadState f r stride N block u)

def firstStreamState (f : Flags) (r : Regions) (stride : Strides) (N block : Nat)
    (p : ℝ) (u : BlockState) : BlockState :=
  dropoutState f.dropout f.dropoutMask r.seeds r.dropoutMask u.pid N block p
    (firstScaledValues f r stride N block u) (firstScaledState f r stride N block u)

theorem firstScaledState_x (f : Flags) (r : Regions) (stride : Strides) (N block : Nat)
    (u : BlockState) :
    (firstScaledState f r stride N block u).regs .real [block] "x" =
      some (realTile (firstScaledValues f r stride N block u)) := by
  have hx := loadState_x f r stride N block u
  rw [loaded_realTile] at hx
  rw [firstScaledState, scaleState_x _ _ _ _ _ _ hx]
  simp only [BlockState.readMem, loadState_mem, firstScaledValues]
  rfl

theorem source_firstStreamPrefix (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) :
    (source f r stride M N eps p block hBlock).body.take 12 =
      loadPrefix f r stride N block ++ scalePhase f.rowscale r.rowscale block ++
        dropoutPhase f.dropout f.dropoutMask r.seeds r.dropoutMask N block p := by
  have hproj : (original f r stride M N eps p block hBlock).toAlgorithm? =
      .ok (original f r stride M N eps p block hBlock).toAlgKernel := rfl
  simp only [source, ComputeKernel.eraseDType, hproj, ComputeKernel.toAlgKernel_fromKernelBody]
  simp only [original, loadPrefix, baseStmt, scalePhase, dropoutPhase, keepOp, randomOp,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, Random.i32ArangeSize, Region.cast,
    List.take_succ_cons, List.take_zero]
  rfl

theorem source_firstStream_exec (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (u : BlockState) :
    stepStmts ((source f r stride M N eps p block hBlock).body.take 12) u =
      some (firstStreamState f r stride N block p u) := by
  have hx := loadState_x f r stride N block u
  rw [loaded_realTile] at hx
  have hs := scalePhase_exec f.rowscale r.rowscale u.pid block
    (loadedValues r.x stride.x N block u) (loadState f r stride N block u)
    (loadState_row f r stride N block u) hx
  rw [source_firstStreamPrefix]
  simp only [ProgramRewrite.append, loadPrefix_exec, Option.bind_some]
  rw [hs, Option.bind_some]
  apply dropoutPhase_exec _ _ _ _ _ _ _ _ _ _ hBlock
  · rw [scaleState_row]
    exact loadState_row f r stride N block u
  · rw [scaleState_cols]
    exact loadState_cols f r stride N block u
  · exact firstScaledState_x f r stride N block u

theorem firstScaledState_mem (f : Flags) (r : Regions) (stride : Strides) (N block : Nat)
    (u : BlockState) : (firstScaledState f r stride N block u).mem = u.mem := by
  rw [firstScaledState, scaleState_mem, loadState_mem]

theorem firstScaledState_seed (f : Flags) (r : Regions) (stride : Strides) (N block : Nat)
    (u : BlockState) :
    (firstScaledState f r stride N block u).readMemValue .int r.seeds u.pid =
      u.readMemValue .int r.seeds u.pid := by
  simp only [BlockState.readMemValue, BlockState.readMemTyped, firstScaledState_mem]

/-- Independent mathematical first-stream value, over the original memory
snapshot; the source's pre-scaling padding is included. -/
theorem firstStreamState_x (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (p : ℝ) (u : BlockState) :
    (firstStreamState f r stride N block p u).regs .real [block] "x" =
      some (realTile (fun j => Formula.stream f (Formula.paddedInputs (snapshot r stride u) N)
        M p Bool.false u.pid j.val)) := by
  rw [firstStreamState, dropoutState_x _ _ _ _ _ _ _ _ _ _
    (firstScaledState_x f r stride N block u), firstScaledState_seed]
  rfl

theorem firstStreamState_mem (f : Flags) (r : Regions) (stride : Strides) (N block : Nat)
    (p : ℝ) (u : BlockState) (hN : N ≤ block) (region : RegionName) (offset : Nat) :
    (firstStreamState f r stride N block p u).mem region offset =
      if f.dropout && f.dropoutMask &&
          decide (region = r.dropoutMask ∧ Contract.inRow (u.pid*N) N offset) then
        MemCell.of .bool (Formula.keep (u.readMemValue .int r.seeds u.pid) p (offset-u.pid*N))
      else u.mem region offset := by
  cases hd : f.dropout <;> cases hm : f.dropoutMask
  all_goals simp [firstStreamState, dropoutState, dropoutBase, hd, hm,
    Memory.scatter_cell _ _ _ _ hN, firstScaledState_seed, firstScaledState_mem,
    Bool.and_eq_true]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Execution
