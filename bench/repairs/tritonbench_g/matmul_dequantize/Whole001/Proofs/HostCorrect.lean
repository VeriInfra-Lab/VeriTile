import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.HostOutputs
namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open Host TorchMM Real
noncomputable section
theorem s1 (api : RealAPI) (a : Arguments) (h : Admissible a .s1)
    (abi : ABI.IndexABI a .s1 .m64n256) (initial : State ℝ) (packed : Packed a initial)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid16.geometry (Host.dequant16 a))))) :
    ABI.SourceExact a .s1 .m64n256 ∧ ∃ middle, ∃ result : Host.Observation (ℝ),
      Grid.run (Correct.dequant16Program a h) jobs initial=some middle ∧
      api.run a middle=some result.state ∧ s1RealRun api a h jobs initial=some result.state ∧
      Host.complete a .s1 (s1RealRun api a h jobs initial)=some result ∧
      (∀ x : Geometry.Cell (SecondGrid.geometry (second a)),
        result.state.mem a.c.region (Geometry.address (SecondGrid.geometry (second a)) x)=
          .mk .fp16 (Real.rawMatmul (RealFirst.inputs (first a .m64n256) initial)
            a.K a.group x.1.val x.2.val)) ∧
      (∀ r o, r≠a.d.region →
        (r≠a.c.region ∨ ∀ x : Geometry.Cell (SecondGrid.geometry (second a)),
          o≠Geometry.address (SecondGrid.geometry (second a)) x) → result.state.mem r o=initial.mem r o) ∧
      result.returnedIdentity=some a.c.identity ∧ result.outputDescriptor=a.c ∧ result.localScratchReleased=Bool.true := by
  obtain ⟨hx,middle,final,hd,hm,hrun,hread,hframe,hreturn⟩ := TorchMM.s1_real api a h abi initial packed jobs schedule
  refine ⟨hx,middle,Host.observe a .s1 final,hd,hm,hrun,?_,hread,hframe,hreturn,rfl,rfl⟩
  simp [Host.complete,hrun]

#axiomsClean s1
theorem host_gptq (a : Arguments) (cfg : FirstConfig) (h : Admissible a .gptq)
    (abi : ABI.IndexABI a .gptq cfg) (initial : State ℝ) (packed : Host.Packed a initial)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (FirstGrid.geometry (first a cfg))))) :
    ABI.SourceExact a .gptq cfg ∧ ∃ result : Host.Observation (ℝ),
      Host.complete a .gptq (Grid.run (firstProgram a cfg h) jobs initial)=some result ∧ (
      Grid.Post (FirstGrid.geometry (first a cfg)) a.c.region initial result.state
        (fun x => .mk .fp16 (rawMatmul (RealFirst.inputs (first a cfg) initial) a.K a.group x.1.val x.2.val))) ∧
      result.returnedIdentity=Host.returned .gptq a ∧ result.outputDescriptor=a.c := by
  obtain ⟨hx,final,he,hpost⟩ := Correct.gptq a cfg h abi initial packed jobs schedule
  refine ⟨hx,Host.observe a .gptq final,?_,hpost,rfl,?_⟩
  · simp [Host.complete,he]
  · simp [Host.observe]
#axiomsClean host_gptq

theorem host_s2 (a : Arguments) (h : Admissible a .s2)
    (abi : ABI.IndexABI a .s2 .m64n256) (initial : State ℝ) (packed : Host.Packed a initial)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (SecondGrid.geometry (second a))))) :
    ABI.SourceExact a .s2 .m64n256 ∧ ∃ result : Host.Observation (ℝ),
      Host.complete a .s2 (Grid.run (secondProgram a h) jobs initial)=some result ∧ (
      Grid.Post (SecondGrid.geometry (second a)) a.c.region initial result.state
        (fun x => .mk .fp16 (rawMatmul (RealSecond.inputs (second a) initial) a.K a.group x.1.val x.2.val))) ∧
      result.returnedIdentity=Host.returned .s2 a ∧ result.outputDescriptor=a.c := by
  obtain ⟨hx,final,he,hpost⟩ := Correct.s2 a h abi initial packed jobs schedule
  refine ⟨hx,Host.observe a .s2 final,?_,hpost,rfl,?_⟩
  · simp [Host.complete,he]
  · simp [Host.observe]
#axiomsClean host_s2

theorem host_dequant32 (a : Arguments) (h : Admissible a route)
    (abi : ABI.IndexABI a route .m64n256) (hr : route=Host.Route.dequant32)
    (initial : State ℝ) (packed : Host.Packed a initial)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid32.geometry (Host.dequant32 a))))) :
    ABI.SourceExact a route .m64n256 ∧ ∃ result : Host.Observation (ℝ),
      Host.complete a route (Grid.run (dequant32Program a h) jobs initial)=some result ∧ (
      Grid.Post (DequantGrid32.geometry (Host.dequant32 a)) a.d.region initial result.state
        (fun x => .mk .fp32 (rawDequant (RealDequant.D32.inputs (Host.dequant32 a) initial) a.group x.1.val x.2.val))) ∧
      result.returnedIdentity=Host.returned route a ∧ result.outputDescriptor=a.d := by
  obtain ⟨hx,final,he,hpost⟩ := Correct.dequant32 a h abi hr initial packed jobs schedule
  refine ⟨hx,Host.observe a route final,?_,hpost,rfl,?_⟩
  · simp [Host.complete,he]
  · rcases hr with rfl <;> simp [Host.observe]
#axiomsClean host_dequant32

theorem host_dequant16 (a : Arguments) (h : Admissible a route)
    (abi : ABI.IndexABI a route .m64n256) (hr : route=Host.Route.dequant16 ∨ route=Host.Route.s1)
    (initial : State ℝ) (packed : Host.Packed a initial)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid16.geometry (Host.dequant16 a))))) :
    ABI.SourceExact a route .m64n256 ∧ ∃ result : Host.Observation (ℝ),
      Host.complete a .dequant16 (Grid.run (dequant16Program a h) jobs initial)=some result ∧ (
      Grid.Post (DequantGrid16.geometry (Host.dequant16 a)) a.d.region initial result.state
        (fun x => .mk .fp16 (rawDequant (RealDequant.D16.inputs (Host.dequant16 a) initial) a.group x.1.val x.2.val))) ∧
      result.returnedIdentity=Host.returned .dequant16 a ∧ result.outputDescriptor=a.d := by
  obtain ⟨hx,final,he,hpost⟩ := Correct.dequant16 a h abi hr initial packed jobs schedule
  refine ⟨hx,Host.observe a .dequant16 final,?_,hpost,rfl,?_⟩
  · simp [Host.complete,he]
  · simp [Host.observe]
#axiomsClean host_dequant16

end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Correct
