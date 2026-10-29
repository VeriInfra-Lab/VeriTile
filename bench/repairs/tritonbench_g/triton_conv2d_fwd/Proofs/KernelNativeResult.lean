import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelExecution
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelStores
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelDefaultNative

namespace VeriTile.Bench.Repairs.Conv2d.KernelNativeResult
open VeriTile.Triton FP.Structural KernelNumeric
noncomputable section

/-- Raw complete grid execution agrees with the independent selected native
instruction recurrence under the unchanged scalar/conversion/fixed-MMA laws. -/
theorem fp_equivalent (s : Shape) (valid : Valid s) (flags : Flags)
    (initial : State KernelStorage.Value) (actual numerics : KernelNative.Instructions)
    (laws : KernelNative.LocalLaws actual (KernelNative.software numerics)) :
    let o:=KernelNative.operations actual flags
    ∃ final,KernelGrid.launch (algebra o) (halfDot o) (defaultDot o) s flags initial=some final ∧
      KernelGrid.SameInputs final initial ∧
      (∀ x,s.Active x → final.mem "output" (outputPointer s x)=
        .mk .fp32 (.scalar (reference (KernelNative.operations (KernelNative.software numerics) flags)
          s flags (memory o initial) x))) ∧
      (∀ (r : RegionName) a,r≠"output" ∨ s.size≤a → final.mem r a=initial.mem r a) := by
  rw [KernelNative.instructions_eq actual _ laws]
  exact launch_reference (KernelNative.operations (KernelNative.software numerics) flags) s valid flags initial

/-- The default raw kernel closes its target to the IEEE RN-even FP32 FMA
circuit, retaining the complete execution and exterior frame. -/
theorem fp_default (s : Shape) (valid : Valid s) (flags : Flags)
    (defaults : flags=⟨Bool.false,Bool.false⟩) (initial : State KernelStorage.Value)
    (actual : KernelNative.Instructions)
    (fmaLaw : ∀ a b c,actual.fmaRN a b c=Software.fmaRN a b c) :
    let o:=KernelNative.operations actual flags
    ∃ final,KernelGrid.launch (algebra o) (halfDot o) (defaultDot o) s flags initial=some final ∧
      KernelGrid.SameInputs final initial ∧
      (∀ x,s.Active x → final.mem "output" (outputPointer s x)=
        .mk .fp32 (.scalar (KernelNative.defaultReference s (memory o initial) x))) ∧
      (∀ (r : RegionName) a,r≠"output" ∨ s.size≤a → final.mem r a=initial.mem r a) := by
  obtain ⟨final,run,inputs,values,frame⟩ := launch_reference (KernelNative.operations actual flags) s valid flags initial
  refine ⟨final,run,inputs,?_,frame⟩
  intro x active
  rw [values x active,defaults,KernelNative.default_reference actual fmaLaw]

/-- Every successful source program, including every grid prefix, commits exactly its own concrete masked byte stores. -/
theorem physical_execution (instructions : KernelNative.Instructions) (s : Shape) (v : Valid s)
    (f : Flags) (pid : Program) (hg : pid.2.2<s.groups) (base : Nat) (before : KernelStorage.Heap)
    (t : State KernelStorage.Value) :
    let o:=KernelNative.operations instructions f
    ∃ u,KernelGrid.program (KernelNumeric.algebra o) (KernelNumeric.halfDot o)
      (KernelNumeric.defaultDot o) s f pid t=some u ∧
      KernelStorage.image o base s.size before u.mem=
        (TileShape.allIndices [128,32]).foldl
          (fun heap i=>if KernelHalf.active (KernelBinding.parameters s f) (KernelBinding.atProgram pid t) i then
            Physical.write heap (base+4*KernelHalf.outputOffset (KernelBinding.parameters s f) (KernelBinding.atProgram pid t) i)
              (KernelNumeric.scalar o (KernelGrid.value (KernelNumeric.algebra o) (KernelNumeric.halfDot o)
                (KernelNumeric.defaultDot o) s f pid t i)) else heap)
          (KernelStorage.image o base s.size before t.mem) :=
  KernelStores.program_bytes instructions s v f pid hg base before t

#axiomsClean fp_equivalent
#axiomsClean fp_default
#axiomsClean physical_execution
#specNonCircular KernelNumeric.reference avoiding [KernelSource.kernel,KernelGrid.launch]
end
end VeriTile.Bench.Repairs.Conv2d.KernelNativeResult
