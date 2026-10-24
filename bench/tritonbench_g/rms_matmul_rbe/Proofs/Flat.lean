import bench.tritonbench_g.rms_matmul_rbe.Proofs.Safety
import VeriTile.Triton.Memory.KernelCore

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.FlatIO
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem flattenOk (input : TypedSource.Storage) (fp8 outputFP8 variant : Bool)
    (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (start : Int) (rbe : Bool) (theta EPS : ℝ) :
    (TypedSource.kernel input fp8 outputFP8 variant X W RMS OUT M N K sxb sxm sxk swk swn srms
      sob som son BM BN BK start rbe theta EPS).eraseDType.toAlgKernel.FlattenOk := by
  rw [ComputeKernel.toAlgKernel, RealSource.projection]
  cases fp8 <;> cases variant <;>
    simp [Kernel.FlattenOk, RealSource.body, RealSource.loop, RealSource.tail,
      RealSource.weightLoad, RealSource.ceilDivOp, rrmPrologue, rrmLoopBody, rrmStoreTail,
      StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def,
      Stmt.eraseDType, Op.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, VeriTile.Triton.eraseDType,
      NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
      FloatDType.toTileDType, TileShape.eraseAxis]

/-- Transport the entire raw-cell frame, including cells outside every
allocation and cells whose stored dtype differs from the output dtype. -/
theorem frame (A : FlatAlloc) (s t : BlockState) (OUT : RegionName)
    (M N BM BN sob som son : Nat)
    (hf : ∀ r o, (r ≠ OUT ∨ ∀ i : TileIndex [BM, BN], active s M N BM BN i →
      o ≠ outOffset s N BM BN sob som son i) → t.mem r o = s.mem r o)
    (r : RegionName) (off : Nat)
    (hn : r ≠ A.flat ∨ ∀ i : TileIndex [BM, BN], active s M N BM BN i →
      off ≠ A.addr OUT (outOffset s N BM BN sob som son i)) :
    (A.flattenState t).mem r off = (A.flattenState s).mem r off := by
  change (if r = A.flat then A.readFlat t off else MemCell.real 0) =
    (if r = A.flat then A.readFlat s off else MemCell.real 0)
  by_cases hr : r = A.flat
  · subst r
    simp only [if_pos rfl]
    unfold FlatAlloc.readFlat
    cases hc : A.decode off with
    | none => rfl
    | some p =>
      apply congrArg A.trCell
      apply hf p.1 p.2
      by_cases hp : p.1 = OUT
      · right
        intro i hi he
        rcases hn with hn | hn
        · exact hn rfl
        · apply hn i hi
          have ha := (A.decode_sound hc).2.1
          simpa only [hp, he] using ha
      · exact Or.inl hp
  · simp [hr]

/-- Successful finite-flat execution of either complete non-FP8 variant,
with its independent padded-input result and every unwritten raw cell. -/
theorem half (input : TypedSource.Storage) (outputFP8 variant : Bool) (A : FlatAlloc)
    (X W RMS OUT : RegionName) (s : BlockState)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (start : Int) (rbe : Bool) (theta EPS : ℝ)
    (hd : A.Disjoint) (hc : ∀ r, r ∉ A.regions → A.extent r = 0)
    (hu : s.undef = fun _ _ => 0)
    (hi : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN sob som son i = outOffset s N BM BN sob som son j → i = j)
    (hb : AccessSafety.Reads A.extent s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms)
    (ho : ∀ i : TileIndex [BM, BN], active s M N BM BN i →
      outOffset s N BM BN sob som son i < A.extent OUT) :
    ∃ t, exec (A.flattenKernel (TypedSource.kernel input Bool.false outputFP8 variant X W RMS OUT M N K
        sxb sxm sxk swk swn srms sob som son BM BN BK start rbe theta EPS).eraseDType.toAlgKernel)
        (A.flattenState s) = some t ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem A.flat (A.addr OUT (outOffset s N BM BN sob som son i)) =
          rmsSpec s X W RMS M N BM BN sxb sxm sxk swk swn srms BK (cdiv K BK) K EPS i.1 i.2.1) ∧
      (∀ r off, (r ≠ A.flat ∨ ∀ i : TileIndex [BM, BN], active s M N BM BN i →
        off ≠ A.addr OUT (outOffset s N BM BN sob som son i)) →
        t.mem r off = (A.flattenState s).mem r off) := by
  have hu' : ∀ r o, s.undef r o = 0 := by intro r o; rw [hu]
  obtain ⟨t, he, _hp, _hu, hv, hf⟩ := RealExecution.run input outputFP8 variant X W RMS OUT s M N K
    sxb sxm sxk swk swn srms sob som son BM BN BK start rbe theta EPS hu' hi
  have hs := AccessSafety.complete_half input outputFP8 variant A.extent X W RMS OUT s M N K
    sxb sxm sxk swk swn srms sob som son BM BN BK start rbe theta EPS hu' hb ho
  refine ⟨A.flattenState t, ?_, ?_, frame A s t OUT M N BM BN sob som son hf⟩
  · rw [A.exec_flatten hd hc _ s hs
      (flattenOk input Bool.false outputFP8 variant X W RMS OUT M N K
        sxb sxm sxk swk swn srms sob som son BM BN BK start rbe theta EPS) hu,
      he, Option.map_some]
  · intro i hact
    have hr : OUT ∈ A.regions := by
      by_contra hn
      have hz := hc OUT hn
      have hb := ho i hact
      omega
    have hh := A.flattenState_readMemAs hd t hr (ho i hact) .real
    simp only [BlockState.readMemAs_real] at hh
    exact (Option.some.inj hh).trans (hv i hact)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.FlatIO
