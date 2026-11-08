import bench.tritonbench_g.int8_matmul_quantization.Proofs.ProducerPassConnection
namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer
open VeriTile.Triton FP.Structural ControlArithmetic
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000

theorem execution_exists {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (s : State α) (width : 0<p.width) :
    ∃ t, exec (backend M integer8) (KernelSource.quantize (parameters p)) s=some t := by
  let start:=prefixExecution M integer8 p s
  obtain ⟨maxed,maxRan,maxInv⟩:=MaxPass.loop M integer8 p s start.val 0 (tripCount p)
    (by omega) (MaxPass.initial M integer8 p s) width
  let reset:=resetExecution M integer8 p s maxed maxInv
  have resetInv:=resetInvariant M integer8 p s maxed maxInv
  obtain ⟨quantized,quantRan,quantInv⟩:=QuantizePass.loop M integer8 p s reset.val
    (finalScale M p s) 0 (tripCount p) (by omega) resetInv
  let finish:=tailExecution M integer8 p s quantized (finalScale M p s) (tripCount p) quantInv
  refine ⟨finish.val,?_⟩
  change run (backend M integer8) (code p) s=some finish.val
  rw [code_decomposition,run_append,run_append,run_append,run_append,start.property]
  simp only [Option.bind_some]
  rw [maxLoop_run]
  have maxDone : rangeInt (backend M integer8) "k" 0 (tripCount p : Int) 1 (maxBody p) start.val=some maxed := maxRan
  rw [maxDone]
  simp only [Option.bind_some]
  rw [reset.property]
  simp only [Option.bind_some]
  rw [quantizeLoop_run]
  have quantDone : rangeInt (backend M integer8) "k" 0 (tripCount p : Int) 1 (quantizeBody p) reset.val=some quantized := quantRan
  rw [quantDone]
  simp only [Option.bind_some]
  exact finish.property

/-- Successful execution of the complete repaired quotation, with its actual
ceil bounds, both loops and both stores. No numerical-domain premise is used. -/
def execution {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (s : State α) (width : 0<p.width) :
    {t : State α // exec (backend M integer8) (KernelSource.quantize (parameters p)) s=some t} :=
  ⟨Classical.choose (execution_exists M integer8 p s width),
    Classical.choose_spec (execution_exists M integer8 p s width)⟩

#axiomsClean execution_exists
#axiomsClean execution
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer
