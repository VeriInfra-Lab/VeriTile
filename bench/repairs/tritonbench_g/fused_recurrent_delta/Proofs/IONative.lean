import bench.repairs.tritonbench_g.fused_recurrent_delta.NativeIO
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.IOResult

namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.IONative
open VeriTile.Triton FP.Structural KernelProgram KernelNativeCarrier Interface
noncomputable section

theorem forward_io (platform : KernelCompiler.Platform) (s : Shape) (f : Flags)
    (binding : KernelCompiler.Binding platform .forward s f) (host : UInt64) (p : Program .forward s) :
    nativeForwardIO platform s f binding host p=
      forwardIO (KernelCompiler.algebra platform binding host) s f scaleSlot p.head p.value := by
  simp only [nativeForwardIO,KernelCompiler.decoded_source]
  rfl

theorem backward_io (platform : KernelCompiler.Platform) (s : Shape) (f : Flags)
    (binding : KernelCompiler.Binding platform .backward s f) (host : UInt64) (p : Program .backward s) :
    nativeBackwardIO platform s f binding host p=
      backwardIO (KernelCompiler.algebra platform binding host) s f scaleSlot p.head p.value := by
  simp only [nativeBackwardIO,KernelCompiler.decoded_source]
  rfl

theorem forward (platform : KernelCompiler.Platform) (s : Shape) (f : Flags)
    (binding : KernelCompiler.Binding platform .forward s f) (host : UInt64) (p : Program .forward s) :
    (nativeForwardIO platform s f binding host p).Observes (fun _ _ => observeNative platform binding)
      (nativeForwardFormula platform s f binding host p.value) := by
  have observed := ProgramIO.Observes.of_implements
    (IOResult.forward (KernelCompiler.algebra platform binding host) s f scaleSlot p.head p.value)
    (fun _ _ => observeNative platform binding)
  intro vals domain bounds hb initial hin
  obtain ⟨safe,final,ran,output,frame⟩ := observed vals domain bounds hb initial hin
  have schedule : (nativeForwardIO platform s f binding host p).schedule vals=
      [job (KernelCompiler.algebra platform binding host) .forward s f scaleSlot p.head p.value] := by
    dsimp only [nativeForwardIO]
    rw [KernelCompiler.decoded_source]
    rfl
  refine ⟨?_,final,?_,?_,frame⟩
  · simpa only [schedule,forwardIO] using safe
  · simpa only [schedule,forwardIO] using ran
  · rintro ⟨port,hp⟩
    match port with
    | 0 => exact output ⟨0,hp⟩
    | 1 => exact output ⟨1,hp⟩
    | 2 => exact output ⟨2,hp⟩
    | n+3 => simp [nativeForwardIO,forwardIO,forwardOutputs] at hp

theorem backward (platform : KernelCompiler.Platform) (s : Shape) (f : Flags)
    (binding : KernelCompiler.Binding platform .backward s f) (host : UInt64) (p : Program .backward s) :
    (nativeBackwardIO platform s f binding host p).Observes (fun _ _ => observeNative platform binding)
      (nativeBackwardFormula platform s f binding host p.value) := by
  have observed := ProgramIO.Observes.of_implements
    (IOResult.backward (KernelCompiler.algebra platform binding host) s f scaleSlot p.head p.value)
    (fun _ _ => observeNative platform binding)
  intro vals domain bounds hb initial hin
  obtain ⟨safe,final,ran,output,frame⟩ := observed vals domain bounds hb initial hin
  have schedule : (nativeBackwardIO platform s f binding host p).schedule vals=
      [job (KernelCompiler.algebra platform binding host) .backward s f scaleSlot p.head p.value] := by
    dsimp only [nativeBackwardIO]
    rw [KernelCompiler.decoded_source]
    rfl
  refine ⟨?_,final,?_,?_,frame⟩
  · simpa only [schedule,backwardIO] using safe
  · simpa only [schedule,backwardIO] using ran
  · rintro ⟨port,hp⟩
    match port with
    | 0 => exact output ⟨0,hp⟩
    | 1 => exact output ⟨1,hp⟩
    | 2 => exact output ⟨2,hp⟩
    | 3 => exact output ⟨3,hp⟩
    | 4 => exact output ⟨4,hp⟩
    | 5 => exact output ⟨5,hp⟩
    | n+6 => simp [nativeBackwardIO,backwardIO,backwardOutputs] at hp

#axiomsClean forward_io
#axiomsClean backward_io
#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.IONative
