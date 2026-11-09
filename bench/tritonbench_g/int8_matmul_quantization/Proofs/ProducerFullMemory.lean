import bench.tritonbench_g.int8_matmul_quantization.Proofs.KernelProducer
import bench.tritonbench_g.int8_matmul_quantization.Proofs.ProducerQuantizeMemory
namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000

/-- Independent ordered FP data graph for both source passes. Each write
block reads the incoming memory, and the FP16 scale is stored last. -/
def fullMemory {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (s : State α) : QuantizePass.Memory α :=
  fun region offset => if region=p.scales ∧ offset=s.pids 0*p.scaleStride then
    .mk .fp16 (M.cast none .real .fp16 (finalScale M p s))
  else QuantizePass.writeIterations M integer8 p s (finalScale M p s) 0 (tripCount p) s.mem region offset

theorem full_memory_exists {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (s : State α) (width : 0<p.width) :
    ∃ t, exec (backend M integer8) (KernelSource.quantize (parameters p)) s=some t ∧
      t.mem=fullMemory M integer8 p s := by
  let start:=prefixExecution M integer8 p s
  obtain ⟨maxed,maxRan,maxInv⟩:=MaxPass.loop M integer8 p s start.val 0 (tripCount p)
    (by omega) (MaxPass.initial M integer8 p s) width
  let reset:=resetExecution M integer8 p s maxed maxInv
  have resetInv:=resetInvariant M integer8 p s maxed maxInv
  obtain ⟨quantized,quantRan,quantInv⟩:=QuantizePass.loop M integer8 p s reset.val
    (finalScale M p s) 0 (tripCount p) (by omega) resetInv
  let finish:=tailExecution M integer8 p s quantized (finalScale M p s) (tripCount p) quantInv
  refine ⟨finish.val,?_,?_⟩
  ·
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
  · have written:=QuantizePass.loop_memory M integer8 p s reset.val (finalScale M p s)
      0 (tripCount p) (by omega) resetInv quantized quantRan
    rw [reset_memory, maxInv.memory] at written
    have tail:=tail_memory M integer8 p s quantized (finalScale M p s) (tripCount p) quantInv
    rw [written] at tail
    simpa only [fullMemory,Nat.sub_zero] using tail

theorem execution_full_memory {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (s : State α) (width : 0<p.width) :
    (execution M integer8 p s width).val.mem=fullMemory M integer8 p s := by
  obtain ⟨final,ran,memory⟩:=full_memory_exists M integer8 p s width
  have same : (execution M integer8 p s width).val=final :=
    Option.some.inj ((execution M integer8 p s width).property.symm.trans ran)
  rw [same]
  exact memory

#axiomsClean full_memory_exists
#axiomsClean execution_full_memory
#specNonCircular fullMemory avoiding [FP.Structural.exec,FP.Structural.run]

end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer
