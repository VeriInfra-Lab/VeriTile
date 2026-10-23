import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.ABI
import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.Post
import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.RealFirst
import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.RealSecond
import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.RealDequant
namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open Host Real
noncomputable section
set_option maxHeartbeats 100000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
abbrev First := VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.Parameters

def firstProgram (a : Arguments) (cfg : FirstConfig) (h : Admissible a .gptq) :=
  FirstGrid.program scalar firstDot (first a cfg) (first_valid a cfg h)
    (distinct a h).1 (distinct a h).2.1 (distinct a h).2.2.1 (distinct a h).2.2.2.1

theorem gptq (a : Arguments) (cfg : FirstConfig) (h : Admissible a .gptq)
    (abi : ABI.IndexABI a .gptq cfg) (initial : State ℝ) (packed : Host.Packed a initial)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (FirstGrid.geometry (first a cfg))))) :
    ABI.SourceExact a .gptq cfg ∧ ∃ final,
      Grid.run (firstProgram a cfg h) jobs initial=some final ∧
      Grid.Post (FirstGrid.geometry (first a cfg)) a.c.region initial final
        (fun x => .mk .fp16 (rawMatmul (RealFirst.inputs (first a cfg) initial) a.K a.group x.1.val x.2.val)) := by
  refine ⟨ABI.source_exact a .gptq cfg abi,?_⟩
  obtain ⟨final,he,hr,hf⟩ := FirstGrid.whole scalar firstDot (first a cfg) (first_valid a cfg h)
    (distinct a h).1 (distinct a h).2.1 (distinct a h).2.2.1 (distinct a h).2.2.2.1 initial jobs schedule
  refine ⟨final,he,Grid.post_of_lanes _ (first_valid a cfg h) _ initial final _ ?_ hf⟩
  intro pid hp i hi
  refine (hr pid hp i hi).trans ?_
  have hG : (first a cfg).groupSize=32*(a.group/32) := by
    have hh := (h.gptqK rfl).2
    change a.group=32*(a.group/32)
    omega
  have hK : (first a cfg).inner=VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.steps (first a cfg)*32 := by
    have hh := (h.gptqK rfl).1
    change a.K=((a.K+32-1)/32)*32
    omega
  have formula := RealFirst.output_formula (first a cfg) (GridExecution.atProgram initial pid)
    rfl (a.group/32) hG (by intro hm; simpa [first] using hm) hK i hi
  exact congrArg (fun v : ℝ => (Cell.mk .fp16 v : Cell ℝ)) formula

def secondProgram (a : Arguments) (h : Admissible a .s2) :=
  SecondGrid.program scalar secondDot (second a) (second_valid a h)
    (distinct a h).1 (distinct a h).2.1 (distinct a h).2.2.1 (distinct a h).2.2.2.1

theorem s2 (a : Arguments) (h : Admissible a .s2)
    (abi : ABI.IndexABI a .s2 .m64n256) (initial : State ℝ) (packed : Host.Packed a initial)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (SecondGrid.geometry (second a))))) :
    ABI.SourceExact a .s2 .m64n256 ∧ ∃ final,
      Grid.run (secondProgram a h) jobs initial=some final ∧
      Grid.Post (SecondGrid.geometry (second a)) a.c.region initial final
        (fun x => .mk .fp16 (rawMatmul (RealSecond.inputs (second a) initial) a.K a.group x.1.val x.2.val)) := by
  refine ⟨ABI.source_exact a .s2 .m64n256 abi,?_⟩
  obtain ⟨final,he,hr,hf⟩ := SecondGrid.whole scalar secondDot (second a) (second_valid a h)
    (distinct a h).1 (distinct a h).2.1 (distinct a h).2.2.1 (distinct a h).2.2.2.1 initial jobs schedule
  refine ⟨final,he,Grid.post_of_lanes _ (second_valid a h) _ initial final _ ?_ hf⟩
  intro pid hp i hi
  refine (hr pid hp i hi).trans ?_
  have formula := RealSecond.output_formula (second a) (GridExecution.atProgram initial pid)
    rfl (by simp [GridExecution.atProgram]) rfl i hi
  exact congrArg (fun v : ℝ => (Cell.mk .fp16 v : Cell ℝ)) formula

def dequant32Program (a : Arguments) (h : Admissible a route) :=
  DequantGrid32.program scalar (dequant32 a) (d32_valid a h)
    (distinct a h).2.2.2.2.1 (distinct a h).2.2.2.2.2.1 (distinct a h).2.2.2.2.2.2

theorem dequant32 (a : Arguments) (h : Admissible a route)
    (abi : ABI.IndexABI a route .m64n256) (hr : route=Host.Route.dequant32)
    (initial : State ℝ) (packed : Host.Packed a initial)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid32.geometry (Host.dequant32 a))))) :
    ABI.SourceExact a route .m64n256 ∧ ∃ final,
      Grid.run (dequant32Program a h) jobs initial=some final ∧
      Grid.Post (DequantGrid32.geometry (Host.dequant32 a)) a.d.region initial final
        (fun x => .mk .fp32 (rawDequant (RealDequant.D32.inputs (Host.dequant32 a) initial) a.group x.1.val x.2.val)) := by
  refine ⟨ABI.source_exact a route .m64n256 abi,?_⟩
  obtain ⟨final,he,readback,frame⟩ := DequantGrid32.whole scalar (Host.dequant32 a) (d32_valid a h)
    (distinct a h).2.2.2.2.1 (distinct a h).2.2.2.2.2.1 (distinct a h).2.2.2.2.2.2 initial jobs schedule
  refine ⟨final,he,Grid.post_of_lanes _ (d32_valid a h) _ initial final _ ?_ frame⟩
  intro pid hp i hi
  refine (readback pid hp i hi).trans ?_
  rw [RealDequant.D32.formula]
  rfl

def dequant16Program (a : Arguments) (h : Admissible a route) :=
  DequantGrid16.program scalar (dequant16 a) (d16_valid a h)
    (distinct a h).2.2.2.2.1 (distinct a h).2.2.2.2.2.1 (distinct a h).2.2.2.2.2.2

theorem dequant16 (a : Arguments) (h : Admissible a route)
    (abi : ABI.IndexABI a route .m64n256) (hr : route=Host.Route.dequant16 ∨ route=Host.Route.s1)
    (initial : State ℝ) (packed : Host.Packed a initial)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid16.geometry (Host.dequant16 a))))) :
    ABI.SourceExact a route .m64n256 ∧ ∃ final,
      Grid.run (dequant16Program a h) jobs initial=some final ∧
      Grid.Post (DequantGrid16.geometry (Host.dequant16 a)) a.d.region initial final
        (fun x => .mk .fp16 (rawDequant (RealDequant.D16.inputs (Host.dequant16 a) initial) a.group x.1.val x.2.val)) := by
  refine ⟨ABI.source_exact a route .m64n256 abi,?_⟩
  obtain ⟨final,he,readback,frame⟩ := DequantGrid16.whole scalar (Host.dequant16 a) (d16_valid a h)
    (distinct a h).2.2.2.2.1 (distinct a h).2.2.2.2.2.1 (distinct a h).2.2.2.2.2.2 initial jobs schedule
  refine ⟨final,he,Grid.post_of_lanes _ (d16_valid a h) _ initial final _ ?_ frame⟩
  intro pid hp i hi
  refine (readback pid hp i hi).trans ?_
  rw [RealDequant.D16.formula]
  rfl

#axiomsClean gptq
#axiomsClean s2
#axiomsClean dequant32
#axiomsClean dequant16
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Correct
