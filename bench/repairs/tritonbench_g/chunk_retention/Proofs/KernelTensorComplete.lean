import bench.repairs.tritonbench_g.chunk_retention.Proofs.KernelTensor

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelTensor
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel KernelPacket KernelStaging
noncomputable section
set_option maxHeartbeats 5000000

def nativeInitial (views : Binding c) (read : Nat→BitVec 32) (pid grid : Nat→Nat) : State (Term (BitVec 32)) :=
  KernelNative.QKV.reify (initial views read pid grid)

theorem native_input_read (ops : Ops (BitVec 32)) (literal : ℝ→BitVec 32)
    (views : Binding c) (read : Nat→BitVec 32) (pid grid : Nat→Nat)
    (r : RegionName) (off : Nat) (hi : off<(views r).tensor.metadata.numel) :
    lower ops literal (((nativeInitial views read pid grid).mem r off).read .real)=
      (VeriTile.Torch.FP32.logicalValue (fun addr=>⟨read addr⟩) (views r).tensor (views r).laws off).bits := by
  rw [native_logical]
  exact input_read views read pid grid r off hi

/-- Real selected-quotation correctness at genuine Tensor coordinates, with a
single shared post-backing read for all output regions and simultaneous exact
exterior, full input-storage, and region-cell frames. -/
theorem real_complete (a : Call) (views : Binding a.shape) (read : Nat→ℝ) (pid grid : Nat→Nat)
    (legal : Legal a (initial views read pid grid))
    (outputs : OutputSeparated a (initial views read pid grid) views)
    (inputs : InputSeparated a (initial views read pid grid) views) :
    ∃ t,FP.Structural.exec (engine (withDot KernelReal.algebra KernelReal.dot)) a.kernel
        (initial views read pid grid)=some t ∧
      (∀ o,Active a (initial views read pid grid) o →
        logical (project id read a (initial views read pid grid) views t)
          (views o.region) (o.offset (initial views read pid grid))=
          realTarget a (initial views read pid grid) o) ∧
      (∀ addr,¬Written a (initial views read pid grid) views addr →
        project id read a (initial views read pid grid) views t addr=read addr) ∧
      (∀ r,ReadRegion a r → ∀ addr,Inside (views r).tensor addr →
        project id read a (initial views read pid grid) views t addr=read addr) ∧
      Frame a (initial views read pid grid) t := by
  obtain ⟨t,ht,hr,hf⟩:=KernelPacket.real_complete a (initial views read pid grid) legal
  refine ⟨t,ht,?_,?_,?_,hf⟩
  · intro o ho
    exact (projected id read a _ legal views outputs t o ho).trans (hr o ho)
  · exact fun addr h=>exterior id read a _ views t addr h
  · exact fun r hr addr hi=>input_frame id read a _ legal views inputs t r hr addr hi

/-- Source execution and all actual Tensor FP32 logical observations equal the
independent software IEEE/FMA targets. Only local scalar/literal laws and
ordinary Tensor/layout/launch/separation guards are premises. -/
theorem fp_complete (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (literal : ℝ→BitVec 32)
    (literals : LiteralLaws (wordOps (Numeric.nativeISA ports)) literal)
    (a : Call) (views : Binding a.shape) (read : Nat→BitVec 32) (pid grid : Nat→Nat)
    (legal : Legal a (nativeInitial views read pid grid))
    (outputs : OutputSeparated a (nativeInitial views read pid grid) views)
    (inputs : InputSeparated a (nativeInitial views read pid grid) views) :
    ∃ t,FP.Structural.exec (engine (withDot (algebra (0 : BitVec 32)) (nativeDot a))) a.kernel
        (nativeInitial views read pid grid)=some t ∧
      (∀ o,Active a (nativeInitial views read pid grid) o →
        (VeriTile.Torch.FP32.logicalValue
          (fun addr=>⟨project (lower (wordOps (Numeric.nativeISA ports)) literal) read a
            (nativeInitial views read pid grid) views t addr⟩)
          (views o.region).tensor (views o.region).laws (o.offset (nativeInitial views read pid grid))).bits=
          nativeTarget (wordOps (Numeric.referenceISA u)) literal a (nativeInitial views read pid grid) o) ∧
      (∀ addr,¬Written a (nativeInitial views read pid grid) views addr →
        project (lower (wordOps (Numeric.nativeISA ports)) literal) read a
          (nativeInitial views read pid grid) views t addr=read addr) ∧
      (∀ r,ReadRegion a r → ∀ addr,Inside (views r).tensor addr →
        project (lower (wordOps (Numeric.nativeISA ports)) literal) read a
          (nativeInitial views read pid grid) views t addr=read addr) ∧
      Frame a (nativeInitial views read pid grid) t := by
  obtain ⟨t,ht,hr,hf⟩:=KernelPacket.fp_complete u ports laws literal literals a
    (nativeInitial views read pid grid) (KernelNative.QKV.reify_tagged _) legal
  refine ⟨t,ht,?_,?_,?_,hf⟩
  · intro o ho
    rw [native_logical]
    exact (projected (lower (wordOps (Numeric.nativeISA ports)) literal) read a _ legal views outputs t o ho).trans (hr o ho)
  · exact fun addr h=>exterior _ read a _ views t addr h
  · exact fun r hr addr hi=>input_frame _ read a _ legal views inputs t r hr addr hi

#axiomsClean native_input_read
#axiomsClean real_complete
#axiomsClean fp_complete
#specNonCircular nativeInitial avoiding [run,FP.Structural.exec,KernelPacket.nativeTarget,project]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelTensor
