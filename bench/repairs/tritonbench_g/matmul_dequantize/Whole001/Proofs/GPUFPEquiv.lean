import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.ABI
import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.NativeBindings
namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open Host
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Complete source graph execution; only scalar/fixed-instruction numerical
ports and compiler operand routing remain local. This is not FP=Real. -/
theorem gptq (a : Arguments) (cfg : FirstConfig)  (h : Admissible a .gptq)
    (abi : ABI.IndexABI a .gptq cfg) (initial : State Native.Expr) (packed : Host.Packed a initial) (words : Native.InputWords a initial)
    (plan : Native.Plan) (isa : Native.WordISA)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (FirstGrid.geometry (first a cfg))))) :
    ABI.SourceExact a .gptq cfg ∧ ∃ final,
      Grid.run (FirstGrid.program Native.scalar Native.firstDot (first a cfg) (first_valid a cfg h) (distinct a h).1 (distinct a h).2.1 (distinct a h).2.2.1 (distinct a h).2.2.2.1) jobs initial=some final ∧
      (∀ pid, pid<Geometry.count (FirstGrid.geometry (first a cfg)) → ∀ i, Geometry.Active (FirstGrid.geometry (first a cfg)) pid i →
        final.mem a.c.region (Geometry.offset (FirstGrid.geometry (first a cfg)) pid i)=.mk .fp16 (VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.outputValue Native.scalar Native.firstDot (first a cfg) (GridExecution.atProgram initial pid) i)) ∧
      (∀ pid, pid<Geometry.count (FirstGrid.geometry (first a cfg)) → ∀ i, Geometry.Active (FirstGrid.geometry (first a cfg)) pid i →
        (Native.lower plan ((final.mem a.c.region (Geometry.offset (FirstGrid.geometry (first a cfg)) pid i)).read .fp16) pid).eval isa=
          (Native.lower plan (VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.outputValue Native.scalar Native.firstDot (first a cfg) (GridExecution.atProgram initial pid) i) pid).eval isa) ∧
      (∀ r o, (r≠a.c.region ∨ ∀ pid, pid<Geometry.count (FirstGrid.geometry (first a cfg)) → ∀ i,
        Geometry.Active (FirstGrid.geometry (first a cfg)) pid i → o≠Geometry.offset (FirstGrid.geometry (first a cfg)) pid i) →
        final.mem r o=initial.mem r o) ∧
      (∀ x : Geometry.Cell (FirstGrid.geometry (first a cfg)), ∃ pid, pid<Geometry.count (FirstGrid.geometry (first a cfg)) ∧ Geometry.Owns (FirstGrid.geometry (first a cfg)) pid x) := by
  refine ⟨ABI.source_exact a .gptq cfg abi,?_⟩
  obtain ⟨final,he,hr,hf⟩ := FirstGrid.whole Native.scalar Native.firstDot (first a cfg) (first_valid a cfg h)
    (distinct a h).1 (distinct a h).2.1 (distinct a h).2.2.1 (distinct a h).2.2.2.1 initial jobs schedule
  refine ⟨final,he,hr,?_,hf,Geometry.covers _ (first_valid a cfg h)⟩
  intro pid hp i hi
  exact congrArg (fun c : Cell Native.Expr => (Native.lower plan (c.read .fp16) pid).eval isa) (hr pid hp i hi)
#axiomsClean gptq

/-- Complete source graph execution; only scalar/fixed-instruction numerical
ports and compiler operand routing remain local. This is not FP=Real. -/
theorem s2 (a : Arguments)   (h : Admissible a .s2)
    (abi : ABI.IndexABI a .s2 .m64n256) (initial : State Native.Expr) (packed : Host.Packed a initial) (words : Native.InputWords a initial)
    (plan : Native.Plan) (isa : Native.WordISA)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (SecondGrid.geometry (second a))))) :
    ABI.SourceExact a .s2 .m64n256 ∧ ∃ final,
      Grid.run (SecondGrid.program Native.scalar Native.secondDot (second a) (second_valid a h) (distinct a h).1 (distinct a h).2.1 (distinct a h).2.2.1 (distinct a h).2.2.2.1) jobs initial=some final ∧
      (∀ pid, pid<Geometry.count (SecondGrid.geometry (second a)) → ∀ i, Geometry.Active (SecondGrid.geometry (second a)) pid i →
        final.mem a.c.region (Geometry.offset (SecondGrid.geometry (second a)) pid i)=.mk .fp16 (FPTypedSecond.outputValue Native.scalar Native.secondDot (second a) (GridExecution.atProgram initial pid) i)) ∧
      (∀ pid, pid<Geometry.count (SecondGrid.geometry (second a)) → ∀ i, Geometry.Active (SecondGrid.geometry (second a)) pid i →
        (Native.lower plan ((final.mem a.c.region (Geometry.offset (SecondGrid.geometry (second a)) pid i)).read .fp16) pid).eval isa=
          (Native.lower plan (FPTypedSecond.outputValue Native.scalar Native.secondDot (second a) (GridExecution.atProgram initial pid) i) pid).eval isa) ∧
      (∀ r o, (r≠a.c.region ∨ ∀ pid, pid<Geometry.count (SecondGrid.geometry (second a)) → ∀ i,
        Geometry.Active (SecondGrid.geometry (second a)) pid i → o≠Geometry.offset (SecondGrid.geometry (second a)) pid i) →
        final.mem r o=initial.mem r o) ∧
      (∀ x : Geometry.Cell (SecondGrid.geometry (second a)), ∃ pid, pid<Geometry.count (SecondGrid.geometry (second a)) ∧ Geometry.Owns (SecondGrid.geometry (second a)) pid x) := by
  refine ⟨ABI.source_exact a .s2 .m64n256 abi,?_⟩
  obtain ⟨final,he,hr,hf⟩ := SecondGrid.whole Native.scalar Native.secondDot (second a) (second_valid a h)
    (distinct a h).1 (distinct a h).2.1 (distinct a h).2.2.1 (distinct a h).2.2.2.1 initial jobs schedule
  refine ⟨final,he,hr,?_,hf,Geometry.covers _ (second_valid a h)⟩
  intro pid hp i hi
  exact congrArg (fun c : Cell Native.Expr => (Native.lower plan (c.read .fp16) pid).eval isa) (hr pid hp i hi)
#axiomsClean s2

/-- Complete source graph execution; only scalar/fixed-instruction numerical
ports and compiler operand routing remain local. This is not FP=Real. -/
theorem dequant32 (a : Arguments)   (h : Admissible a .dequant32)
    (abi : ABI.IndexABI a .dequant32 .m64n256) (initial : State Native.Expr) (packed : Host.Packed a initial) (words : Native.InputWords a initial)
    (plan : Native.Plan) (isa : Native.WordISA)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid32.geometry (Host.dequant32 a))))) :
    ABI.SourceExact a .dequant32 .m64n256 ∧ ∃ final,
      Grid.run (DequantGrid32.program Native.scalar  (Host.dequant32 a) (d32_valid a h) (distinct a h).2.2.2.2.1 (distinct a h).2.2.2.2.2.1 (distinct a h).2.2.2.2.2.2) jobs initial=some final ∧
      (∀ pid, pid<Geometry.count (DequantGrid32.geometry (Host.dequant32 a)) → ∀ i, Geometry.Active (DequantGrid32.geometry (Host.dequant32 a)) pid i →
        final.mem a.d.region (Geometry.offset (DequantGrid32.geometry (Host.dequant32 a)) pid i)=.mk .fp32 (Dequant32.value Native.scalar (Host.dequant32 a) (DequantGrid32.atProgram (Host.dequant32 a) initial pid) i)) ∧
      (∀ pid, pid<Geometry.count (DequantGrid32.geometry (Host.dequant32 a)) → ∀ i, Geometry.Active (DequantGrid32.geometry (Host.dequant32 a)) pid i →
        (Native.lower plan ((final.mem a.d.region (Geometry.offset (DequantGrid32.geometry (Host.dequant32 a)) pid i)).read .fp32) pid).eval isa=
          (Native.lower plan (Dequant32.value Native.scalar (Host.dequant32 a) (DequantGrid32.atProgram (Host.dequant32 a) initial pid) i) pid).eval isa) ∧
      (∀ r o, (r≠a.d.region ∨ ∀ pid, pid<Geometry.count (DequantGrid32.geometry (Host.dequant32 a)) → ∀ i,
        Geometry.Active (DequantGrid32.geometry (Host.dequant32 a)) pid i → o≠Geometry.offset (DequantGrid32.geometry (Host.dequant32 a)) pid i) →
        final.mem r o=initial.mem r o) ∧
      (∀ x : Geometry.Cell (DequantGrid32.geometry (Host.dequant32 a)), ∃ pid, pid<Geometry.count (DequantGrid32.geometry (Host.dequant32 a)) ∧ Geometry.Owns (DequantGrid32.geometry (Host.dequant32 a)) pid x) := by
  refine ⟨ABI.source_exact a .dequant32 .m64n256 abi,?_⟩
  obtain ⟨final,he,hr,hf⟩ := DequantGrid32.whole Native.scalar  (Host.dequant32 a) (d32_valid a h)
    (distinct a h).2.2.2.2.1 (distinct a h).2.2.2.2.2.1 (distinct a h).2.2.2.2.2.2 initial jobs schedule
  refine ⟨final,he,hr,?_,hf,Geometry.covers _ (d32_valid a h)⟩
  intro pid hp i hi
  exact congrArg (fun c : Cell Native.Expr => (Native.lower plan (c.read .fp32) pid).eval isa) (hr pid hp i hi)
#axiomsClean dequant32

/-- Complete source graph execution; only scalar/fixed-instruction numerical
ports and compiler operand routing remain local. This is not FP=Real. -/
theorem dequant16 (a : Arguments)  (route : Host.Route) (h : Admissible a route)
    (route16 : route=.dequant16 ∨ route=.s1)
    (abi : ABI.IndexABI a route .m64n256) (initial : State Native.Expr) (packed : Host.Packed a initial) (words : Native.InputWords a initial)
    (plan : Native.Plan) (isa : Native.WordISA)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid16.geometry (Host.dequant16 a))))) :
    ABI.SourceExact a route .m64n256 ∧ ∃ final,
      Grid.run (DequantGrid16.program Native.scalar  (Host.dequant16 a) (d16_valid a h) (distinct a h).2.2.2.2.1 (distinct a h).2.2.2.2.2.1 (distinct a h).2.2.2.2.2.2) jobs initial=some final ∧
      (∀ pid, pid<Geometry.count (DequantGrid16.geometry (Host.dequant16 a)) → ∀ i, Geometry.Active (DequantGrid16.geometry (Host.dequant16 a)) pid i →
        final.mem a.d.region (Geometry.offset (DequantGrid16.geometry (Host.dequant16 a)) pid i)=.mk .fp16 (VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.value Native.scalar (Host.dequant16 a) (DequantGrid16.atProgram (Host.dequant16 a) initial pid) i)) ∧
      (∀ pid, pid<Geometry.count (DequantGrid16.geometry (Host.dequant16 a)) → ∀ i, Geometry.Active (DequantGrid16.geometry (Host.dequant16 a)) pid i →
        (Native.lower plan ((final.mem a.d.region (Geometry.offset (DequantGrid16.geometry (Host.dequant16 a)) pid i)).read .fp16) pid).eval isa=
          (Native.lower plan (VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.value Native.scalar (Host.dequant16 a) (DequantGrid16.atProgram (Host.dequant16 a) initial pid) i) pid).eval isa) ∧
      (∀ r o, (r≠a.d.region ∨ ∀ pid, pid<Geometry.count (DequantGrid16.geometry (Host.dequant16 a)) → ∀ i,
        Geometry.Active (DequantGrid16.geometry (Host.dequant16 a)) pid i → o≠Geometry.offset (DequantGrid16.geometry (Host.dequant16 a)) pid i) →
        final.mem r o=initial.mem r o) ∧
      (∀ x : Geometry.Cell (DequantGrid16.geometry (Host.dequant16 a)), ∃ pid, pid<Geometry.count (DequantGrid16.geometry (Host.dequant16 a)) ∧ Geometry.Owns (DequantGrid16.geometry (Host.dequant16 a)) pid x) := by
  refine ⟨ABI.source_exact a route .m64n256 abi,?_⟩
  obtain ⟨final,he,hr,hf⟩ := DequantGrid16.whole Native.scalar  (Host.dequant16 a) (d16_valid a h)
    (distinct a h).2.2.2.2.1 (distinct a h).2.2.2.2.2.1 (distinct a h).2.2.2.2.2.2 initial jobs schedule
  refine ⟨final,he,hr,?_,hf,Geometry.covers _ (d16_valid a h)⟩
  intro pid hp i hi
  exact congrArg (fun c : Cell Native.Expr => (Native.lower plan (c.read .fp16) pid).eval isa) (hr pid hp i hi)
#axiomsClean dequant16
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.FPEquiv
