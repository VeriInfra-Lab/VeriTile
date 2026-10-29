import bench.tritonbench_g.rms_rbe_matmul.Proofs.FP8Arrays
import bench.tritonbench_g.rms_rbe_matmul.Proofs.FP8Safety
import bench.tritonbench_g.rms_rbe_matmul.Proofs.Flat

namespace VeriTile.Bench.TritonBenchG.RmsRbeMatmul.FlatIO
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000

/-- Successful finite-flat byte execution with the independent decoded-array
formula. Actual raw byte loads, all padded lanes and every unwritten typed
cell are retained. The real output models an E5M2 carrier, not packed bytes. -/
theorem bytes (input : TypedSource.Storage) (variant : Bool) (A : FlatAlloc)
    (X W RMS OUT : RegionName) (s : BlockState)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (start : Int) (rbe : Bool) (theta EPS : ℝ)
    (xs : Fin (cdiv K BK) → Fin BM → Fin BK → ℝ)
    (rs : Fin (cdiv K BK) → Fin BK → ByteExecution.Byte)
    (ws : Fin (cdiv K BK) → Fin BK → Fin BN → ByteExecution.Byte)
    (hd : A.Disjoint) (hc : ∀ r, r ∉ A.regions → A.extent r = 0)
    (hu : s.undef = fun _ _ => 0) (hcol : 0 < son) (hrow : N * son ≤ som)
    (hXW : X ≠ W) (hXR : X ≠ RMS)
    (hinput : ByteExecution.Inputs s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms xs rs ws)
    (hb : AccessSafety.Reads A.extent s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms)
    (ho : ∀ i : TileIndex [BM, BN], active s M N BM BN i →
      outOffset s N BM BN sob som son i < A.extent OUT) :
    ∃ t, exec (A.flattenKernel (TypedSource.kernel input Bool.true variant X W RMS OUT M N K
        sxb sxm sxk swk swn srms sob som son BM BN BK start rbe theta EPS).eraseDType.toAlgKernel)
        (A.flattenState s) = some t ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem A.flat (A.addr OUT (outOffset s N BM BN sob som son i)) =
          PaddedRMSGemm.output K (cdiv K BK) BM BN BK EPS xs
            (fun t e => (rs t e).value) (fun t e j => (ws t e j).value) i.1 i.2.1) ∧
      (∀ r off, (r ≠ A.flat ∨ ∀ i : TileIndex [BM, BN], active s M N BM BN i →
        off ≠ A.addr OUT (outOffset s N BM BN sob som son i)) →
        t.mem r off = (A.flattenState s).mem r off) := by
  have hu' : ∀ r o, s.undef r o = 0 := by intro r o; rw [hu]
  obtain ⟨t, he, hv, hf⟩ := ByteExecution.arrays input variant X W RMS OUT s M N K
    sxb sxm sxk swk swn srms sob som son BM BN BK start rbe theta EPS
    xs rs ws hu' hcol hrow hXW hXR hinput
  have hs := ByteExecution.complete input variant A.extent X W RMS OUT s M N K
    sxb sxm sxk swk swn srms sob som son BM BN BK start rbe theta EPS hu' hXW hXR
    (ByteExecution.inputs_bytes s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms xs rs ws hinput) hb ho
  refine ⟨A.flattenState t, ?_, ?_, frame A s t OUT M N BM BN sob som son hf⟩
  · rw [A.exec_flatten hd hc _ s hs
      (flattenOk input Bool.true variant X W RMS OUT M N K
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
end VeriTile.Bench.TritonBenchG.RmsRbeMatmul.FlatIO
