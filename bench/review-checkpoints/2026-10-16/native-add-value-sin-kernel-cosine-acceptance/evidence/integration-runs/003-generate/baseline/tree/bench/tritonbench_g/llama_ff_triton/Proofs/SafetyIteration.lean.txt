import bench.tritonbench_g.llama_ff_triton.Proofs.Real
import VeriTile.Triton.Memory.TraceProofs

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.AccessSafety
open VeriTile.Triton
set_option maxHeartbeats 600000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- Every lane of an unmasked source load must belong to its allocation. -/
theorem load_safe (bounds : RegionBounds) (s : BlockState) (name : RegName)
    (d : TileDType) (p : Tile .ptr sh)
    (hp : s.regs .ptr sh name = some p)
    (hb : ∀ i, (p.data i).2 < bounds (p.data i).1) :
    Op.SafeAt bounds s (.load d (.ptr (.ref .ptr sh name)) .none) := by
  simp only [Op.SafeAt.eq_def, true_and, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe.eq_def,
    MaskOpt.Active]
  intro ptrs he i _
  rw [evalOp_ref, hp] at he
  cases he
  exact hb i

/-- Exact access safety of a complete source iteration, including the FP8
raw loads and decoder assignments. Numerical values and decoder success are
unrestricted: failure does not create fictitious later accesses. -/
theorem iteration (fp8 : Bool) (bounds : RegionBounds) (s : BlockState)
    (BM BN BK sak sw1k sw3k srms : Nat)
    (ap : Tile .ptr [BM, BK]) (p1 p3 : Tile .ptr [BK, BN]) (rp : Tile .ptr [1, BK])
    (ha : s.regs .ptr [BM, BK] "a_ptrs" = some ap)
    (h1 : s.regs .ptr [BK, BN] "w1_ptrs" = some p1)
    (h3 : s.regs .ptr [BK, BN] "w3_ptrs" = some p3)
    (hr : s.regs .ptr [1, BK] "rms_w_ptrs" = some rp)
    (ba : ∀ i, (ap.data i).2 < bounds (ap.data i).1)
    (b1 : ∀ i, (p1.data i).2 < bounds (p1.data i).1)
    (b3 : ∀ i, (p3.data i).2 < bounds (p3.data i).1)
    (br : ∀ i, (rp.data i).2 < bounds (rp.data i).1) :
    Stmt.TraceSafeList bounds (RealSource.loop fp8 BM BN BK sak sw1k sw3k srms) s := by
  cases fp8
  all_goals
    simp only [RealSource.loop, ffLoopBody, List.map_cons, List.map_nil,
      List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero,
      List.cons_append, List.nil_append, List.flatMap_cons, List.flatMap_nil,
      RealSource.weightLoad, Bool.false_and, Bool.true_and, Bool.false_eq_true,
      Bool.or_eq_true, beq_iff_eq, String.reduceEq, false_or, or_false, ↓reduceIte,
      Stmt.eraseDType, Op.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
      IntegralDType.eraseDType, ComparableDType.eraseDType, FloatDType.toTileDType]
    apply TraceProofs.assign_cons
    · exact load_safe bounds s "a_ptrs" .real ap ha ba
    · intro atile
      apply TraceProofs.assign_cons
      · simp [Op.SafeAt.eq_def]
      · intro sumtile
        apply TraceProofs.assign_cons
        · exact load_safe bounds _ "rms_w_ptrs" _ rp (by simpa using hr) br
        · intro rtile
          repeat' (apply TraceProofs.assign_cons; (solve | simp [Op.SafeAt.eq_def]); intro)
          apply TraceProofs.assign_cons
          · exact load_safe bounds _ "w1_ptrs" _ p1 (by simpa using h1) b1
          · intro w1tile
            repeat' (apply TraceProofs.assign_cons; (solve | simp [Op.SafeAt.eq_def]); intro)
            apply TraceProofs.assign_cons
            · exact load_safe bounds _ "w3_ptrs" _ p3 (by simpa using h3) b3
            · intro w3tile
              apply TraceProofs.all_safe
              simp [Stmt.TraceSafe, Op.SafeAt.eq_def]

/-- The complete prologue performs no memory accesses. -/
theorem prologue (bounds : RegionBounds) (s : BlockState) (A W1 W3 RMS : RegionName)
    (M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms : Nat) :
    Stmt.TraceSafeList bounds
      (ffPrologue A W1 W3 RMS M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms) s := by
  apply TraceProofs.all_safe
  simp [ffPrologue, Stmt.TraceSafe, Op.SafeAt.eq_def]

/-- Explicit bounds for the actual ceil-K traversal. The four inputs are
unmasked, including wrapped inactive M/N lanes and all padded K lanes. -/
def Reads (bounds : RegionBounds) (s : BlockState) (A W1 W3 RMS : RegionName)
    (M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms : Nat) : Prop :=
  (∀ t < cdiv K BK, ∀ (i : Fin BM) (e : Fin BK),
    rowIndex s M N BM BN i * sam + (t * BK + e.val) * sak < bounds A) ∧
  (∀ t < cdiv K BK, ∀ (e : Fin BK) (j : Fin BN),
    (t * BK + e.val) * sw1k + colIndex s N BN j * sw1n < bounds W1) ∧
  (∀ t < cdiv K BK, ∀ (e : Fin BK) (j : Fin BN),
    (t * BK + e.val) * sw3k + colIndex s N BN j * sw3n < bounds W3) ∧
  (∀ t < cdiv K BK, ∀ e : Fin BK, (t * BK + e.val) * srms < bounds RMS)

/-- Numerical recurrence invariant supplies the same actual source pointers;
this lemma uses only those four pointer fields. -/
theorem iteration_of_invariant (fp8 : Bool) (bounds : RegionBounds)
    (A W1 W3 RMS : RegionName) (initial s : BlockState)
    (M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms t : Nat)
    (ht : t < cdiv K BK)
    (h : ffInvariant A W1 W3 RMS initial M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms (cdiv K BK) t s)
    (hb : Reads bounds initial A W1 W3 RMS M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms) :
    Stmt.TraceSafeList bounds (RealSource.loop fp8 BM BN BK sak sw1k sw3k srms)
      (s.setReg "_i" .nat [] (Tile.scalar t)) := by
  obtain ⟨_, _, _, _, _, _, _, ha, h1, h3, hr, _, _⟩ := h
  apply iteration fp8 bounds _ BM BN BK sak sw1k sw3k srms _ _ _ _
    (by simpa using ha) (by simpa using h1) (by simpa using h3) (by simpa using hr)
  · intro i
    simpa [Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hb.1 t ht i.1 i.2.1
  · intro i
    simpa [Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hb.2.1 t ht i.1 i.2.1
  · intro i
    simpa [Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hb.2.2.1 t ht i.1 i.2.1
  · intro i
    simpa [Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hb.2.2.2 t ht i.2.1


#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.AccessSafety
