import bench.repairs.tritonbench_g.mul_exponent_compensator.Proofs.FP

namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.FPEquiv
open VeriTile.Triton Host FPProof
open VeriTile.Torch.FP32 (Tensor TensorLaws)
open TritonMul2 (Memory)
open TritonMul2.Host (logical logical_fp32 observeByte byte_frame)
noncomputable section

/-- All-word actual finite-source execution, exact returned allocated object,
entry logical values, preserved input and exterior bytes, and actual pointers. -/
specification wrapper_fp32 (multiply : FPWord → FPWord → FPWord)
    (x : Tensor) (lx : TensorLaws x) (current B : Nat) (h : InputContract x.metadata current B)
    (before : Memory FPWord) (p : Prepared x lx current B h before)
    (order : List Nat) (schedule : order.Perm (List.range (grid x.metadata.numel B))) :
    ∃ final,
      runWrapper (operations multiply) p order=.ok (some (p.output,final)) ∧
      (∀ i,i<x.metadata.numel →
        VeriTile.Torch.FP32.logicalValue (fun a => final (a/4)) p.output p.laws i=
        nativeReference multiply (VeriTile.Torch.FP32.logicalValue (fun a => before (a/4)) x lx i)) ∧
      (∀ address,address<4*(layout p).output ∨ 4*((layout p).output+x.metadata.numel)≤address →
        observeByte final address=observeByte before address) ∧
      (∀ i,i<x.metadata.numel → logical final x i=logical before x i) ∧
      p.output.metadata.shape=x.metadata.shape ∧ p.output.metadata.device=⟨"cuda",current⟩ ∧
      p.output.object≠x.object ∧
      (0<x.metadata.numel →
        BitVec.ofNat 64 (4*(layout p).input)=x.dataPtr ∧
        BitVec.ofNat 64 (4*(layout p).output)=p.output.dataPtr ∧ x.metadata.device=⟨"cuda",current⟩) := by
  obtain ⟨final,he,hv,hf⟩ := wrapper_correct (operations multiply) x lx current B h before p order schedule
  refine ⟨final,he,?_,fun address outside => byte_frame before final _ _ hf address outside,
    input_frame p final hf,p.shape,p.device,p.freshObject,fun positive =>
      ⟨input_pointer x lx current B h positive,output_pointer p positive,h.device positive⟩⟩
  intro i hi
  rw [logical_fp32 final p.output p.laws p.noNeg i,logical_fp32 before x lx h.noNeg i]
  exact hv i hi

/-- Conditional numerical refinement retains the same checked host contract. -/
specification wrapper_ieee_if_local (multiply : FPWord → FPWord → FPWord)
    (localLaw : LocalMultiplyExact multiply) (x : Tensor) (lx : TensorLaws x) (current B : Nat)
    (h : InputContract x.metadata current B) (before : Memory FPWord)
    (p : Prepared x lx current B h before) (order : List Nat)
    (schedule : order.Perm (List.range (grid x.metadata.numel B))) :
    ∃ final,
      runWrapper (operations multiply) p order=.ok (some (p.output,final)) ∧
      (∀ i,i<x.metadata.numel → finite (logical before x i) →
        logical final p.output i=softwareReference (logical before x i)) ∧
      (∀ address,address<4*(layout p).output ∨ 4*((layout p).output+x.metadata.numel)≤address →
        observeByte final address=observeByte before address) := by
  obtain ⟨final,he,hv,hf⟩ := wrapper_correct (operations multiply) x lx current B h before p order schedule
  exact ⟨final,he,fun i hi finite => (hv i hi).trans (localLaw _ finite),
    fun address outside => byte_frame before final _ _ hf address outside⟩

#axiomsClean wrapper_fp32
#axiomsClean wrapper_ieee_if_local
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.FPEquiv
