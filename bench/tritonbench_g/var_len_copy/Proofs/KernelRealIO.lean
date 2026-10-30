import bench.tritonbench_g.var_len_copy.Proofs.KernelRealInitialize

namespace VeriTile.Bench.Repairs.VarLenCopy.KernelReal
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.VarLenCopy

/-- Successful complete signed-loop execution copies precisely the actual
source span from the initial source buffer and preserves every other cell. -/
theorem source_io (s : BlockState) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int)) :
    ∃ final, exec (source os ol ns src dst B) s = some final ∧
      BlockState.WriteWithin (footprint dst newStart length B) s final ∧
      (∀ n, n < length.toNat →
        final.readMem dst (newStart + n) = s.readMem src (oldStart + n)) ∧
      final.pids = s.pids ∧ final.numPids = s.numPids ∧ final.undef = s.undef := by
  obtain ⟨final,he,hf⟩ := source_run s os ol ns src dst B length oldStart newStart hB hsd hL hO hN
  refine ⟨final,he,hf.frame,?_,hf.pids,hf.numPids,hf.undef⟩
  intro n hn
  have decomp : n / B * B + n % B = n := by
    simpa only [Nat.mul_comm, Nat.add_comm] using Nat.mod_add_div n B
  have within : n / B * B < length.toNat := by omega
  have round := (Math.loop_test length hB).mp within
  have active : ((n / B * B + n % B : Nat) : Int) < length := by omega
  simpa only [decomp] using hf.values (n / B) (n % B) round (Nat.mod_lt n hB) active

/-- Empty and negative signed lengths execute no stores at any address. -/
theorem source_nonpositive (s : BlockState) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst) (hle : length ≤ 0)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int)) :
    ∃ final, exec (source os ol ns src dst B) s = some final ∧ final.mem = s.mem := by
  obtain ⟨final,he,hf,_,_,_,_⟩ := source_io s os ol ns src dst B length oldStart newStart hB hsd hL hO hN
  refine ⟨final,he,?_⟩
  funext r o
  symm
  apply hf r o
  simp [footprint, Int.toNat_eq_zero.mpr hle]

/-- Nonpositive lengths leave every memory cell unchanged, even when the
unused signed starts are negative. No finiteness or padding condition is used. -/
theorem source_empty (s : BlockState) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (hB : 0 < B) (hle : length ≤ 0)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length) :
    ∃ final, exec (source os ol ns src dst B) s = some final ∧ final.mem = s.mem := by
  simp [exec,source,Source.kernel,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    NumericDType.eraseDType,ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType,Region.cast,Except.bind,bind]
  simp only [stepStmts,stepStmt,evalOp.eq_def,BlockState.setReg_same,
    BlockState.setReg_ne_name,ne_eq,String.reduceEq,not_false_eq_true,
    Option.bind_eq_bind,Option.bind,Option.pure_def]
  simp only [Tile.scalar_data,Tile.uop,Tile.bop,Int.toNat_natCast]
  simp only [ite_true, BlockState.setReg_readMemValue, Int.toNat_natCast]
  simp only [Region.cast] at hL ⊢
  have hpid : (Int.ofNat (s.pids 0)).toNat = s.pids 0 := rfl
  rw [hpid,hL]
  simp only [stepForRangeIntAux.step_ge (Nat.ne_of_gt hB) hle]
  exact ⟨_,rfl,rfl⟩

end VeriTile.Bench.Repairs.VarLenCopy.KernelReal

#guard_msgs (drop info) in
#auditModuleAxioms
