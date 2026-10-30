import bench.repairs.tritonbench_g.matrix_vector_multip.CandidateIO
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.NativeResult
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourcePairSafety
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.ExplicitFMA
open VeriTile.Triton FP.Structural KernelSource
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeSource WholeNative
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 24000

theorem execution_equal (native : Scalars) (g : Graph BN BM) (N M : Nat)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM)
    (s : State Payload) :
    exec (algebra native g) (Kernels.explicitFMA N M BN BM rowShape columnShape) s =
      exec (WholeNative.algebra native g) (Kernels.source N M BN BM rowShape columnShape) s :=
  SourcePair.execution_equal native N M BN BM g rowShape columnShape s

theorem source_safe (native : Scalars) (g : Graph BN BM) (N M : Nat) (width : 0<BM)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM)
    (bounds : RegionName → Nat) (s : State Payload)
    (domain : SourceDomain N M BN BM (s.pids 0))
    (matrix : ∀ i j, i<N → j<M → i*M+j<bounds "A")
    (vector : ∀ j, j<M → j<bounds "B")
    (output : ∀ i : Fin BN, row BN (s.pids 0) i.val<N → row BN (s.pids 0) i.val<bounds "C") :
    TraceSafe (algebra native g) bounds (Kernels.explicitFMA N M BN BM rowShape columnShape) s :=
  (SourcePairSafety.source_equal native N M BN BM g rowShape columnShape bounds s).mpr
    (SourceSafety.source_safe (WholeNative.algebra native g) N M BN BM rowShape columnShape s
      domain bounds matrix vector output)

theorem equivalent (native : Scalars) (N M BN BM pid : Nat) (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM) (g : Graph BN BM)
    (width : 0<BM) (domain : SourceDomain N M BN BM pid) :
    (originalIO native N M BN BM pid rowShape columnShape g).Equivalent (variant native N M BN BM pid rowShape columnShape g) := by
  intro vals _ bounds covered initial pins
  obtain ⟨safe,final,ran,values,frame⟩ :=
    KernelNativeResult.io_correct native N M BN BM pid g width rowShape columnShape domain vals trivial bounds covered initial pins
  refine ⟨safe,?_,final,final,ran,?_,?_,frame,frame⟩
  · change (invocation native N M BN BM pid rowShape columnShape g).Safe bounds initial ∧ _
    refine ⟨?_,fun _ _ => trivial⟩
    apply source_safe native g N M width rowShape columnShape bounds _
      (by simpa [invocation,IO.invocation,Invocation.initial,bindRegisters] using domain)
    · intro i j hi hj
      exact covered.2.1 (0 : Fin 2) (⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial
    · intro j hj
      exact covered.2.1 (1 : Fin 2) (⟨j,hj⟩,PUnit.unit) trivial
    · intro i active
      have act : pid*BN+i.val<N := by
        simpa [WholeSource.row,parameters,invocation,IO.invocation,Invocation.initial,
          bindRegisters] using active
      simpa [WholeSource.row,parameters,invocation,IO.invocation,Invocation.initial,
        bindRegisters] using covered.2.2.1 (0 : Fin 1) (i,PUnit.unit) act
  · change ((invocation native N M BN BM pid rowShape columnShape g).run initial).bind (fun s => some s)=some final
    change (exec (algebra native g) (Kernels.explicitFMA N M BN BM rowShape columnShape)
      ((invocation native N M BN BM pid rowShape columnShape g).initial initial)).bind _=some final
    rw [execution_equal]
    exact ran
  · intro o j active
    rfl

#axiomsClean execution_equal
#axiomsClean source_safe
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.ExplicitFMA
