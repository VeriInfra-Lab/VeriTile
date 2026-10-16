import bench.tritonbench_g.rms_matmul_rbe.Proofs.FP8Execution
import bench.tritonbench_g.rms_matmul_rbe.Proofs.Safety

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.ByteExecution
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- Source read safety follows the actual raw state's three pointer tiles. -/
theorem iteration_safe (bounds : RegionBounds) (X W RMS : RegionName) (initial s : BlockState)
    (M N K BM BN BK sxb sxm sxk swk swn srms t : Nat)
    (ht : t < cdiv K BK)
    (h : Invariant X W RMS initial M N BM BN BK sxb sxm sxk swk swn srms (cdiv K BK) t s)
    (hb : AccessSafety.Reads bounds initial X W RMS M N K BM BN BK sxb sxm sxk swk swn srms) :
    Stmt.TraceSafeList bounds (RealSource.loop Bool.true BM BN BK sxk swk srms)
      (s.setReg "_i" .nat [] (Tile.scalar t)) := by
  obtain ⟨_, _, _, _, _, _, _, hx, hw, hr, _, _⟩ := h.1
  apply AccessSafety.iteration Bool.true bounds _ BM BN BK sxk swk srms _ _ _
    (by simpa [ByteLoads.view] using hx) (by simpa [ByteLoads.view] using hw)
    (by simpa [ByteLoads.view] using hr)
  · intro i
    simpa [ByteLoads.view, rowIndex, rowGlobal, Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hb.1 t ht i.1 i.2.1
  · intro i
    simpa [ByteLoads.view, colIndex, colGlobal, Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hb.2.1 t ht i.1 i.2.1
  · intro i
    simpa [Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hb.2.2 t ht i.2.1

/-- Full raw-byte source trace safety on the finite-input formula domain,
including the batch offset, wrapped M/N lanes and unmasked K padding. -/
theorem complete (input : TypedSource.Storage) (outputFP8 variant : Bool) (bounds : RegionBounds)
    (X W RMS OUT : RegionName) (s : BlockState)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (start : Int) (rbe : Bool) (theta EPS : ℝ)
    (hu : ∀ rg o, s.undef rg o = 0) (hXW : X ≠ W) (hXR : X ≠ RMS)
    (hbytes : Bytes s W RMS N BN BK swk swn srms (cdiv K BK))
    (hb : AccessSafety.Reads bounds s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms)
    (ho : ∀ i : TileIndex [BM, BN], active s M N BM BN i →
      outOffset s N BM BN sob som son i < bounds OUT) :
    (TypedSource.kernel input Bool.true outputFP8 variant X W RMS OUT M N K sxb sxm sxk swk swn srms
      sob som son BM BN BK start rbe theta EPS).eraseDType.toAlgKernel.TraceSafe bounds s := by
  obtain ⟨s0, hpre, hP0⟩ := initialize_run X W RMS OUT s M N K
    sxb sxm sxk swk swn srms sob som son BM BN BK EPS hu
  obtain ⟨final, sLoop, hLoop, hfinal, hPLoop⟩ :=
    forRange_inv (idx := "_i") (start := 0) (stop := cdiv K BK) (step := 1)
      (by omega) hP0
      (fun i st hi hinv => by
        have := step X W RMS s M N BM BN BK sxb sxm sxk swk swn srms (cdiv K BK)
          hXW hXR hbytes i st hi hinv
        simpa using this)
  have hf : final = cdiv K BK := le_antisymm hPLoop.1.2.1 hfinal
  rw [hf] at hPLoop
  have hloop : stepStmt (.forRangeDyn "_i" (.constNat 0) (RealSource.ceilDivOp K BK)
      (.constNat 1) (RealSource.loop Bool.true BM BN BK sxk swk srms)) s0 = some sLoop := by
    simp only [stepForRangeAux.forRangeDyn_unfold, evalOp_constNat,
      RealTransport.ceil_eval, Option.bind_some, Tile.scalar_data]
    simpa only [stepStmt] using hLoop
  rw [ComputeKernel.toAlgKernel, RealSource.projection]
  change Stmt.TraceSafeList bounds (RealSource.body Bool.true variant X W RMS OUT M N K
    sxb sxm sxk swk swn srms sob som son BM BN BK EPS) s
  rw [RealSource.body, RealTransport.prologue]
  apply TraceProofs.append_safe _ s (AccessSafety.prologue bounds s X W RMS M N BM BN BK sxb sxm sxk swk swn srms)
  intro t ht
  have he : t = s0 := Option.some.inj (ht.symm.trans hpre)
  subst t
  rw [TraceProofs.cons_of_run_iff hloop]
  refine ⟨?_, AccessSafety.tail variant bounds s sLoop OUT M N BM BN BK K sob som son EPS
    hPLoop.1.2.2.1 hPLoop.1.2.2.2.1 hPLoop.1.2.2.2.2.1 ho⟩
  rw [Stmt.TraceSafe]
  refine ⟨by simp [Op.SafeAt.eq_def], by simp [RealSource.ceilDivOp, Op.SafeAt.eq_def],
    by simp [Op.SafeAt.eq_def], ?_⟩
  simp only [evalOp_constNat, RealTransport.ceil_eval, Tile.scalar_data]
  apply TraceProofs.range_safe (P := Invariant X W RMS s M N BM BN BK sxb sxm sxk swk swn srms (cdiv K BK)) ?_ ?_ 0 s0 hP0
  · intro i st hi hinv
    exact iteration_safe bounds X W RMS s st M N K BM BN BK sxb sxm sxk swk swn srms i hi hinv hb
  · intro i st st' hi hinv hr
    obtain ⟨u, he, hu⟩ := step X W RMS s M N BM BN BK sxb sxm sxk swk swn srms (cdiv K BK)
      hXW hXR hbytes i st hi hinv
    have ht : st' = u := Option.some.inj (hr.symm.trans he)
    subst st'
    exact hu

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.ByteExecution
