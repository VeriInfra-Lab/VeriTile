import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.KernelExecution
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

namespace VeriTile.Bench.Repairs.DequantizeMatmul.KernelResult
open VeriTile.Triton FP.Structural KernelSource KernelExecution
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

abbrev Lane (p : Parameters) := TileIndex [p.BK, p.BN]
abbrev Program := Nat × Nat
def row (p : Parameters) (pid : Program) (i : Lane p) := pid.1 * p.BK + i.1.val
def col (p : Parameters) (pid : Program) (i : Lane p) := pid.2 * p.BN + i.2.1.val
def active (p : Parameters) (pid : Program) (i : Lane p) : Prop :=
  row p pid i < p.K ∧ col p pid i < p.N
instance (p : Parameters) (pid : Program) (i : Lane p) : Decidable (active p pid i) :=
  inferInstanceAs (Decidable (_ ∧ _))
def offset (p : Parameters) (pid : Program) (i : Lane p) := row p pid i * p.N + col p pid i

/-- Independent source-order scalar formula: signed promotion, one FP32
multiply and an already typed FP32 store. No arithmetic identity is imposed. -/
def dequantized {α : Type} (M : Algebra α) (q : Int) (scale : α) : α :=
  M.binary (some .fp32) .real .mul (M.fromInt (some .fp32) q) (M.fp32Load scale)

def word {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) (i : Lane p) : α :=
  let pid := (s.pids 0, s.pids 1)
  dequantized M
    (if active p pid i then (s.mem p.b (offset p pid i)).read .int else 0)
    (if col p pid i < p.N then (s.mem p.scale (col p pid i)).read .real
      else M.literal (some .fp32) .real 0)

def written {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.BK, p.BN]).foldl (fun t i =>
    if active p (s.pids 0, s.pids 1) i then
      t.write p.output (offset p (s.pids 0, s.pids 1) i) (.mk .real (word M p s i))
    else t) s

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    (execution M p s).val.mem = (written M p s).mem := by
  have hz : (0.0 : ℝ) = 0 := by norm_num
  simp [execution, written, word, dequantized, row, col, active, offset,
    masked_scatter_memory, Broadcast.leftIndex, Broadcast.rightIndex, hz,
    -Nat.cast_mul, -Nat.cast_add,
    ← Int.natCast_mul, ← Int.natCast_add, Int.ofNat_lt, Int.toNat_natCast]

theorem offset_bound (p : Parameters) (pid : Program) (i : Lane p) (hi : active p pid i) :
    offset p pid i < p.K * p.N := by
  have h := Nat.mul_le_mul_right p.N hi.1
  have hn := hi.2
  simp only [Nat.succ_mul] at h
  dsimp [offset]
  omega

theorem offset_coordinates (p : Parameters) (pid : Program) (i : Lane p) (hi : active p pid i) :
    offset p pid i / p.N = row p pid i ∧ offset p pid i % p.N = col p pid i := by
  have hn : 0 < p.N := Nat.zero_lt_of_lt hi.2
  constructor
  · simp [offset, Nat.mul_add_div hn, Nat.mul_comm, Nat.div_eq_of_lt hi.2]
  · simp [offset, Nat.add_mod, Nat.mod_eq_of_lt hi.2]

theorem offset_injective (p : Parameters) (pid : Program) :
    ∀ i j : Lane p, active p pid i → active p pid j → offset p pid i = offset p pid j → i = j := by
  intro i j hi hj he
  have eri := offset_coordinates p pid i hi
  have erj := offset_coordinates p pid j hj
  have er := congrArg (· / p.N) he
  have ec := congrArg (· % p.N) he
  dsimp only at er ec
  rw [eri.1, erj.1] at er
  rw [eri.2, erj.2] at ec
  apply Prod.ext
  · apply Fin.ext
    dsimp [row] at er
    omega
  · apply Prod.ext
    · apply Fin.ext
      dsimp [col] at ec
      omega
    · exact Subsingleton.elim _ _

theorem written_word {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (i : Lane p) (hi : active p (s.pids 0, s.pids 1) i) :
    (written M p s).mem p.output (offset p (s.pids 0, s.pids 1) i) = .mk .real (word M p s i) :=
  masked_readback_active _ _ _ _ _ i hi (offset_injective p (s.pids 0, s.pids 1))

theorem written_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (r : RegionName) (o : Nat)
    (h : r ≠ p.output ∨ ∀ i : Lane p, active p (s.pids 0, s.pids 1) i → o ≠ offset p (s.pids 0, s.pids 1) i) :
    (written M p s).mem r o = s.mem r o := by
  apply State.masked_scatter_frame
  exact h.imp id (fun h i _ hi => h i hi)

#axiomsClean execution_memory
#axiomsClean written_word
#axiomsClean written_frame
#specNonCircular dequantized avoiding [KernelExecution.execution, KernelExecution.exec, KernelSource.kernel]
end
end VeriTile.Bench.Repairs.DequantizeMatmul.KernelResult
