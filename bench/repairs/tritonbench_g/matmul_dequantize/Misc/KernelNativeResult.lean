import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.GPUFPEquiv

namespace VeriTile.Bench.Repairs.MatmulDequantize.KernelNativeResult
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 5000000

/-- Full raw source grid agrees with the independent ordered initial-memory
instruction recurrence under the unchanged native routing and scalar ISA. -/
theorem gptq_kernel_equiv (p : VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.Parameters) (valid : Geometry.Valid (FirstGrid.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output)
    (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (initial : State Native.Expr) (plan : Native.Plan) (isa : Native.WordISA)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (FirstGrid.geometry p)))) :
    ∃ final,Grid.run (FirstGrid.program Native.scalar Native.firstDot p valid ha hb hs hz) jobs initial=some final ∧
      (∀ pid,pid<Geometry.count (FirstGrid.geometry p) → ∀ i,Geometry.Active (FirstGrid.geometry p) pid i →
        (Native.lower plan ((final.mem p.output (Geometry.offset (FirstGrid.geometry p) pid i)).read .fp16) pid).eval isa=
          (Native.lower plan (VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.outputValue Native.scalar Native.firstDot p (GridExecution.atProgram initial pid) i) pid).eval isa) ∧
      (∀ r o,(r≠p.output ∨ ∀ pid,pid<Geometry.count (FirstGrid.geometry p) → ∀ i,
        Geometry.Active (FirstGrid.geometry p) pid i → o≠Geometry.offset (FirstGrid.geometry p) pid i) →
        final.mem r o=initial.mem r o) := by
  obtain ⟨final,ran,values,frame⟩ := FirstGrid.whole Native.scalar Native.firstDot p valid ha hb hs hz initial jobs schedule
  refine ⟨final,ran,?_,frame⟩
  intro pid hp i hi
  rw [values pid hp i hi, Cell.read_same]
#axiomsClean gptq_kernel_equiv
#specNonCircular VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.outputValue avoiding [VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.source,FP.Structural.exec,Grid.run]

/-- Full raw source grid agrees with the independent ordered initial-memory
instruction recurrence under the unchanged native routing and scalar ISA. -/
theorem s2_kernel_equiv (p : FPTypedSecond.Parameters) (valid : Geometry.Valid (SecondGrid.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output)
    (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (initial : State Native.Expr) (plan : Native.Plan) (isa : Native.WordISA)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (SecondGrid.geometry p)))) :
    ∃ final,Grid.run (SecondGrid.program Native.scalar Native.secondDot p valid ha hb hs hz) jobs initial=some final ∧
      (∀ pid,pid<Geometry.count (SecondGrid.geometry p) → ∀ i,Geometry.Active (SecondGrid.geometry p) pid i →
        (Native.lower plan ((final.mem p.output (Geometry.offset (SecondGrid.geometry p) pid i)).read .fp16) pid).eval isa=
          (Native.lower plan (FPTypedSecond.outputValue Native.scalar Native.secondDot p (GridExecution.atProgram initial pid) i) pid).eval isa) ∧
      (∀ r o,(r≠p.output ∨ ∀ pid,pid<Geometry.count (SecondGrid.geometry p) → ∀ i,
        Geometry.Active (SecondGrid.geometry p) pid i → o≠Geometry.offset (SecondGrid.geometry p) pid i) →
        final.mem r o=initial.mem r o) := by
  obtain ⟨final,ran,values,frame⟩ := SecondGrid.whole Native.scalar Native.secondDot p valid ha hb hs hz initial jobs schedule
  refine ⟨final,ran,?_,frame⟩
  intro pid hp i hi
  rw [values pid hp i hi, Cell.read_same]
#axiomsClean s2_kernel_equiv
#specNonCircular FPTypedSecond.outputValue avoiding [FPTypedSecond.source,FP.Structural.exec,Grid.run]

/-- Full raw source grid agrees with the independent ordered initial-memory
instruction recurrence under the unchanged native routing and scalar ISA. -/
theorem dequant32_kernel_equiv (p : Dequant32.Parameters) (valid : Geometry.Valid (DequantGrid32.geometry p))
    (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (initial : State Native.Expr) (plan : Native.Plan) (isa : Native.WordISA)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid32.geometry p)))) :
    ∃ final,Grid.run (DequantGrid32.program Native.scalar p valid hb hs hz) jobs initial=some final ∧
      (∀ pid,pid<Geometry.count (DequantGrid32.geometry p) → ∀ i,Geometry.Active (DequantGrid32.geometry p) pid i →
        (Native.lower plan ((final.mem p.output (Geometry.offset (DequantGrid32.geometry p) pid i)).read .fp32) pid).eval isa=
          (Native.lower plan (Dequant32.value Native.scalar p (DequantGrid32.atProgram p initial pid) i) pid).eval isa) ∧
      (∀ r o,(r≠p.output ∨ ∀ pid,pid<Geometry.count (DequantGrid32.geometry p) → ∀ i,
        Geometry.Active (DequantGrid32.geometry p) pid i → o≠Geometry.offset (DequantGrid32.geometry p) pid i) →
        final.mem r o=initial.mem r o) := by
  obtain ⟨final,ran,values,frame⟩ := DequantGrid32.whole Native.scalar p valid hb hs hz initial jobs schedule
  refine ⟨final,ran,?_,frame⟩
  intro pid hp i hi
  rw [values pid hp i hi, Cell.read_same]
#axiomsClean dequant32_kernel_equiv
#specNonCircular Dequant32.value avoiding [Dequant32.source,FP.Structural.exec,Grid.run]

/-- Full raw source grid agrees with the independent ordered initial-memory
instruction recurrence under the unchanged native routing and scalar ISA. -/
theorem dequant16_kernel_equiv (p : VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.Parameters) (valid : Geometry.Valid (DequantGrid16.geometry p))
    (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (initial : State Native.Expr) (plan : Native.Plan) (isa : Native.WordISA)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid16.geometry p)))) :
    ∃ final,Grid.run (DequantGrid16.program Native.scalar p valid hb hs hz) jobs initial=some final ∧
      (∀ pid,pid<Geometry.count (DequantGrid16.geometry p) → ∀ i,Geometry.Active (DequantGrid16.geometry p) pid i →
        (Native.lower plan ((final.mem p.output (Geometry.offset (DequantGrid16.geometry p) pid i)).read .fp16) pid).eval isa=
          (Native.lower plan (VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.value Native.scalar p (DequantGrid16.atProgram p initial pid) i) pid).eval isa) ∧
      (∀ r o,(r≠p.output ∨ ∀ pid,pid<Geometry.count (DequantGrid16.geometry p) → ∀ i,
        Geometry.Active (DequantGrid16.geometry p) pid i → o≠Geometry.offset (DequantGrid16.geometry p) pid i) →
        final.mem r o=initial.mem r o) := by
  obtain ⟨final,ran,values,frame⟩ := DequantGrid16.whole Native.scalar p valid hb hs hz initial jobs schedule
  refine ⟨final,ran,?_,frame⟩
  intro pid hp i hi
  rw [values pid hp i hi, Cell.read_same]
#axiomsClean dequant16_kernel_equiv
#specNonCircular VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.value avoiding [VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.source,FP.Structural.exec,Grid.run]
end
end VeriTile.Bench.Repairs.MatmulDequantize.KernelNativeResult
