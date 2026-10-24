import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.PublicIO
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Misc.NativeReference
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Misc.NativeIO

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.NativePublicIO
open VeriTile.Triton FP.Structural KernelProgram KernelCarrier
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000

theorem target (isa : Instructions) (plan : Reduction.ProductPlan 1024) (kind : Kind)
    (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c) :
    (fun a => (IO.program (model isa plan kind c) kind c s seed p).mapOutputs
      (fun _ _ => Node.value) (PublicIO.target (model isa plan kind c) kind c s seed p a))=
        Reference.native isa plan kind c s seed p := by
  funext a o
  cases kind with
  | forward =>
    fin_cases o
    funext ⟨t,e,j,⟨⟩⟩
    change (KernelReference.forward (algebra isa plan (swapped .forward c))
      (Reference.forwardInputs a) seed t.val (e,j,PUnit.unit)).value=_
    rw [KernelWords.forward_lowering]
    rfl
  | backward =>
    fin_cases o
    · funext ⟨t,e,j,⟨⟩⟩
      change (KernelReference.reverse (algebra isa plan (swapped .backward c))
        (Reference.backwardInputs a) c.N (c.N-1-t.val) (e,j,PUnit.unit)).value=_
      rw [KernelWords.reverse_lowering]
      rfl
    · funext ⟨t,⟨⟩⟩
      change (KernelReference.contraction (algebra isa plan (swapped .backward c))
        (KernelReference.reverse (algebra isa plan (swapped .backward c))
          (Reference.backwardInputs a) c.N (c.N-1-t.val))
        ((Reference.backwardInputs a).saved t.val)).value=_
      rw [KernelWords.contraction_lowering,KernelWords.reverse_lowering]
      rfl
    · funext ⟨e,j,⟨⟩⟩
      change (KernelReference.reverse (algebra isa plan (swapped .backward c))
        (Reference.backwardInputs a) c.N c.N (e,j,PUnit.unit)).value=_
      rw [KernelWords.reverse_lowering]
      rfl

theorem native (isa : Instructions) (plan : Reduction.ProductPlan 1024) (kind : Kind)
    (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c) (hn : 0<c.N) :
    (IO.program (model isa plan kind c) kind c s seed p).Observes (fun _ _ => Node.value)
      (Reference.native isa plan kind c s seed p) := by
  rw [←target]
  exact ProgramIO.Observes.of_implements (PublicIO.implements _ kind c s seed p hn) _

theorem compiled_eq (platform : KernelCompiler.Platform)
    (binding : KernelCompiler.Binding platform kind c s seed) (p : Program c) :
    NativeIO.compiled platform binding p=
      IO.program (model (platform.instructions binding.object.primitives)
        (KernelCompiler.plan platform binding) kind c) kind c s seed p := by
  unfold NativeIO.compiled
  rw [KernelCompiler.decoded_source]
  have same : KernelCompiler.algebra platform binding=
      model (platform.instructions binding.object.primitives) (KernelCompiler.plan platform binding) kind c := by
    simp only [KernelCompiler.algebra,model,binding.operands]
  rw [same]
  cases kind <;> rfl

theorem compiled (platform : KernelCompiler.Platform)
    (binding : KernelCompiler.Binding platform kind c s seed) (p : Program c) (hn : 0<c.N) :
    (NativeIO.compiled platform binding p).Observes (fun _ _ => Node.value)
      (Reference.native (platform.instructions binding.object.primitives)
        (KernelCompiler.plan platform binding) kind c s seed p) := by
  have result := native (platform.instructions binding.object.primitives)
    (KernelCompiler.plan platform binding) kind c s seed p hn
  have same : KernelCompiler.algebra platform binding=
      model (platform.instructions binding.object.primitives) (KernelCompiler.plan platform binding) kind c := by
    simp only [KernelCompiler.algebra,model,binding.operands]
  cases kind <;> simpa only [ProgramIO.Observes,ProgramIO.ObservesOn,NativeIO.compiled,
    KernelCompiler.decoded_source,same,IO.program,IO.forward,IO.backward,IO.launch] using result

end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.NativePublicIO
