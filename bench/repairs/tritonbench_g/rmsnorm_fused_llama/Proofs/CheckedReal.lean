import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.Physical

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.CheckedReal
open VeriTile Triton Contract
noncomputable section

structure Result where
  tensor : TensorInfo
  source : Host.Result BlockState

def invoke (x w : TensorInfo) (eps : Epsilon) (ok : Admitted x w eps)
    (call : Host.Preparation BlockState MemCell BlockState.mem (request x w eps ok))
    (binding : Physical.Binding x w eps ok call) : Option Result :=
  (Host.invokeReal (request x w eps ok) call).map fun source => ⟨binding.output,source⟩

theorem whole_real (x w : TensorInfo) (eps : Epsilon) (ok : Admitted x w eps)
    (call : Host.Preparation BlockState MemCell BlockState.mem (request x w eps ok))
    (binding : Physical.Binding x w eps ok call) :
    ∃ result,invoke x w eps ok call binding=some result ∧ result.tensor=binding.output ∧
      result.tensor.shape=x.shape ∧ result.tensor.dtype=.floating .fp32 ∧ result.tensor.device=x.device ∧
      (∀ row,row<(request x w eps ok).rows → ∀ col,col<(request x w eps ok).columns →
        result.source.state.readMem call.output.region (row*(request x w eps ok).columns+col)=
          RealGrid.value (call.before.readMem call.input.region) (call.before.readMem call.weight.region)
            row (request x w eps ok).columns eps.value col) ∧
      (∀ r o,r≠call.output.region ∨ (request x w eps ok).size≤o → result.source.state.mem r o=call.before.mem r o) := by
  obtain ⟨source,run,tensor,shape,dtype,device,values,frame⟩:=Host.whole_real (request x w eps ok) call
  refine ⟨⟨binding.output,source⟩,by simp [invoke,run],rfl,binding.outputShape,binding.outputFP32,binding.outputDevice,?_,?_⟩
  · intro row hr col hc
    rw [←tensor]
    simpa [(request_facts x w eps ok).2.2.1] using values row hr col hc
  · intro r o outside
    rw [←tensor] at outside
    exact frame r o outside

#axiomsClean whole_real
#specNonCircular RealGrid.value avoiding [invoke,Host.invokeReal,RealGrid.run,Triton.exec]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.CheckedReal
