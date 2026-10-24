import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.HostOutputs
namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open Host TorchMM
noncomputable section
theorem s1 (api : NativeAPI) (a : Arguments) (h : Admissible a .s1)
    (abi : ABI.IndexABI a .s1 .m64n256) (initial : State Native.Expr) (packed : Packed a initial) (words : Native.InputWords a initial)
    (plan : Native.Plan) (isa : Native.WordISA)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid16.geometry (Host.dequant16 a))))) :
    ABI.SourceExact a .s1 .m64n256 ∧ ∃ middle, ∃ result : Host.Observation (Native.Expr),
      Grid.run (DequantGrid16.program Native.scalar (Host.dequant16 a) (d16_valid a h)
        (distinct a h).2.2.2.2.1 (distinct a h).2.2.2.2.2.1 (distinct a h).2.2.2.2.2.2) jobs initial=some middle ∧
      (∀ pid, pid<Geometry.count (DequantGrid16.geometry (Host.dequant16 a)) → ∀ i,
        Geometry.Active (DequantGrid16.geometry (Host.dequant16 a)) pid i →
        (Native.lower plan ((middle.mem a.d.region (Geometry.offset (DequantGrid16.geometry (Host.dequant16 a)) pid i)).read .fp16) pid).eval isa=
          (Native.lower plan (VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.value Native.scalar
            (Host.dequant16 a) (DequantGrid16.atProgram (Host.dequant16 a) initial pid) i) pid).eval isa) ∧
      api.run a middle=some result.state ∧ api.call a middle result.state ∧
      s1NativeRun api a h jobs initial=some result.state ∧
      Host.complete a .s1 (s1NativeRun api a h jobs initial)=some result ∧
      (∀ r o, r≠a.d.region →
        (r≠a.c.region ∨ ∀ x : Geometry.Cell (SecondGrid.geometry (second a)),
          o≠Geometry.address (SecondGrid.geometry (second a)) x) → result.state.mem r o=initial.mem r o) ∧
      result.returnedIdentity=some a.c.identity ∧ result.outputDescriptor=a.c ∧ result.localScratchReleased=Bool.true := by
  obtain ⟨hx,middle,final,hd,dgraph,hm,hcall,hrun,hframe,hreturn⟩ := TorchMM.s1_native api a h abi initial packed words plan isa jobs schedule
  refine ⟨hx,middle,Host.observe a .s1 final,hd,dgraph,hm,hcall,hrun,?_,hframe,hreturn,rfl,rfl⟩
  simp [Host.complete,hrun]

#axiomsClean s1
theorem host_gptq (a : Arguments) (cfg : FirstConfig)  (h : Admissible a .gptq)
    (abi : ABI.IndexABI a .gptq cfg) (initial : State Native.Expr) (packed : Host.Packed a initial) (words : Native.InputWords a initial)
    (plan : Native.Plan) (isa : Native.WordISA)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (FirstGrid.geometry (first a cfg))))) :
    ABI.SourceExact a .gptq cfg ∧ ∃ result : Host.Observation (Native.Expr),
      Host.complete a .gptq (Grid.run (FirstGrid.program Native.scalar Native.firstDot (first a cfg) (first_valid a cfg h) (distinct a h).1 (distinct a h).2.1 (distinct a h).2.2.1 (distinct a h).2.2.2.1) jobs initial)=some result ∧ (
      (∀ pid, pid<Geometry.count (FirstGrid.geometry (first a cfg)) → ∀ i, Geometry.Active (FirstGrid.geometry (first a cfg)) pid i →
        result.state.mem a.c.region (Geometry.offset (FirstGrid.geometry (first a cfg)) pid i)=.mk .fp16 (VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.outputValue Native.scalar Native.firstDot (first a cfg) (GridExecution.atProgram initial pid) i)) ∧
      (∀ pid, pid<Geometry.count (FirstGrid.geometry (first a cfg)) → ∀ i, Geometry.Active (FirstGrid.geometry (first a cfg)) pid i →
        (Native.lower plan ((result.state.mem a.c.region (Geometry.offset (FirstGrid.geometry (first a cfg)) pid i)).read .fp16) pid).eval isa=
          (Native.lower plan (VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.outputValue Native.scalar Native.firstDot (first a cfg) (GridExecution.atProgram initial pid) i) pid).eval isa) ∧
      (∀ r o, (r≠a.c.region ∨ ∀ pid, pid<Geometry.count (FirstGrid.geometry (first a cfg)) → ∀ i,
        Geometry.Active (FirstGrid.geometry (first a cfg)) pid i → o≠Geometry.offset (FirstGrid.geometry (first a cfg)) pid i) →
        result.state.mem r o=initial.mem r o) ∧
      (∀ x : Geometry.Cell (FirstGrid.geometry (first a cfg)), ∃ pid, pid<Geometry.count (FirstGrid.geometry (first a cfg)) ∧ Geometry.Owns (FirstGrid.geometry (first a cfg)) pid x)) ∧
      result.returnedIdentity=Host.returned .gptq a ∧ result.outputDescriptor=a.c := by
  obtain ⟨hx,final,he,hpost⟩ := FPEquiv.gptq a cfg h abi initial packed words plan isa jobs schedule
  refine ⟨hx,Host.observe a .gptq final,?_,hpost,rfl,?_⟩
  · simp [Host.complete,he]
  · simp [Host.observe]
#axiomsClean host_gptq

theorem host_s2 (a : Arguments)   (h : Admissible a .s2)
    (abi : ABI.IndexABI a .s2 .m64n256) (initial : State Native.Expr) (packed : Host.Packed a initial) (words : Native.InputWords a initial)
    (plan : Native.Plan) (isa : Native.WordISA)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (SecondGrid.geometry (second a))))) :
    ABI.SourceExact a .s2 .m64n256 ∧ ∃ result : Host.Observation (Native.Expr),
      Host.complete a .s2 (Grid.run (SecondGrid.program Native.scalar Native.secondDot (second a) (second_valid a h) (distinct a h).1 (distinct a h).2.1 (distinct a h).2.2.1 (distinct a h).2.2.2.1) jobs initial)=some result ∧ (
      (∀ pid, pid<Geometry.count (SecondGrid.geometry (second a)) → ∀ i, Geometry.Active (SecondGrid.geometry (second a)) pid i →
        result.state.mem a.c.region (Geometry.offset (SecondGrid.geometry (second a)) pid i)=.mk .fp16 (FPTypedSecond.outputValue Native.scalar Native.secondDot (second a) (GridExecution.atProgram initial pid) i)) ∧
      (∀ pid, pid<Geometry.count (SecondGrid.geometry (second a)) → ∀ i, Geometry.Active (SecondGrid.geometry (second a)) pid i →
        (Native.lower plan ((result.state.mem a.c.region (Geometry.offset (SecondGrid.geometry (second a)) pid i)).read .fp16) pid).eval isa=
          (Native.lower plan (FPTypedSecond.outputValue Native.scalar Native.secondDot (second a) (GridExecution.atProgram initial pid) i) pid).eval isa) ∧
      (∀ r o, (r≠a.c.region ∨ ∀ pid, pid<Geometry.count (SecondGrid.geometry (second a)) → ∀ i,
        Geometry.Active (SecondGrid.geometry (second a)) pid i → o≠Geometry.offset (SecondGrid.geometry (second a)) pid i) →
        result.state.mem r o=initial.mem r o) ∧
      (∀ x : Geometry.Cell (SecondGrid.geometry (second a)), ∃ pid, pid<Geometry.count (SecondGrid.geometry (second a)) ∧ Geometry.Owns (SecondGrid.geometry (second a)) pid x)) ∧
      result.returnedIdentity=Host.returned .s2 a ∧ result.outputDescriptor=a.c := by
  obtain ⟨hx,final,he,hpost⟩ := FPEquiv.s2 a h abi initial packed words plan isa jobs schedule
  refine ⟨hx,Host.observe a .s2 final,?_,hpost,rfl,?_⟩
  · simp [Host.complete,he]
  · simp [Host.observe]
#axiomsClean host_s2

theorem host_dequant32 (a : Arguments)   (h : Admissible a .dequant32)
    (abi : ABI.IndexABI a .dequant32 .m64n256) (initial : State Native.Expr) (packed : Host.Packed a initial) (words : Native.InputWords a initial)
    (plan : Native.Plan) (isa : Native.WordISA)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid32.geometry (Host.dequant32 a))))) :
    ABI.SourceExact a .dequant32 .m64n256 ∧ ∃ result : Host.Observation (Native.Expr),
      Host.complete a .dequant32 (Grid.run (DequantGrid32.program Native.scalar  (Host.dequant32 a) (d32_valid a h) (distinct a h).2.2.2.2.1 (distinct a h).2.2.2.2.2.1 (distinct a h).2.2.2.2.2.2) jobs initial)=some result ∧ (
      (∀ pid, pid<Geometry.count (DequantGrid32.geometry (Host.dequant32 a)) → ∀ i, Geometry.Active (DequantGrid32.geometry (Host.dequant32 a)) pid i →
        result.state.mem a.d.region (Geometry.offset (DequantGrid32.geometry (Host.dequant32 a)) pid i)=.mk .fp32 (Dequant32.value Native.scalar (Host.dequant32 a) (DequantGrid32.atProgram (Host.dequant32 a) initial pid) i)) ∧
      (∀ pid, pid<Geometry.count (DequantGrid32.geometry (Host.dequant32 a)) → ∀ i, Geometry.Active (DequantGrid32.geometry (Host.dequant32 a)) pid i →
        (Native.lower plan ((result.state.mem a.d.region (Geometry.offset (DequantGrid32.geometry (Host.dequant32 a)) pid i)).read .fp32) pid).eval isa=
          (Native.lower plan (Dequant32.value Native.scalar (Host.dequant32 a) (DequantGrid32.atProgram (Host.dequant32 a) initial pid) i) pid).eval isa) ∧
      (∀ r o, (r≠a.d.region ∨ ∀ pid, pid<Geometry.count (DequantGrid32.geometry (Host.dequant32 a)) → ∀ i,
        Geometry.Active (DequantGrid32.geometry (Host.dequant32 a)) pid i → o≠Geometry.offset (DequantGrid32.geometry (Host.dequant32 a)) pid i) →
        result.state.mem r o=initial.mem r o) ∧
      (∀ x : Geometry.Cell (DequantGrid32.geometry (Host.dequant32 a)), ∃ pid, pid<Geometry.count (DequantGrid32.geometry (Host.dequant32 a)) ∧ Geometry.Owns (DequantGrid32.geometry (Host.dequant32 a)) pid x)) ∧
      result.returnedIdentity=Host.returned .dequant32 a ∧ result.outputDescriptor=a.d := by
  obtain ⟨hx,final,he,hpost⟩ := FPEquiv.dequant32 a h abi initial packed words plan isa jobs schedule
  refine ⟨hx,Host.observe a .dequant32 final,?_,hpost,rfl,?_⟩
  · simp [Host.complete,he]
  · simp [Host.observe]
#axiomsClean host_dequant32

theorem host_dequant16 (a : Arguments)  (route : Host.Route) (h : Admissible a route)
    (route16 : route=.dequant16 ∨ route=.s1)
    (abi : ABI.IndexABI a route .m64n256) (initial : State Native.Expr) (packed : Host.Packed a initial) (words : Native.InputWords a initial)
    (plan : Native.Plan) (isa : Native.WordISA)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid16.geometry (Host.dequant16 a))))) :
    ABI.SourceExact a route .m64n256 ∧ ∃ result : Host.Observation (Native.Expr),
      Host.complete a .dequant16 (Grid.run (DequantGrid16.program Native.scalar  (Host.dequant16 a) (d16_valid a h) (distinct a h).2.2.2.2.1 (distinct a h).2.2.2.2.2.1 (distinct a h).2.2.2.2.2.2) jobs initial)=some result ∧ (
      (∀ pid, pid<Geometry.count (DequantGrid16.geometry (Host.dequant16 a)) → ∀ i, Geometry.Active (DequantGrid16.geometry (Host.dequant16 a)) pid i →
        result.state.mem a.d.region (Geometry.offset (DequantGrid16.geometry (Host.dequant16 a)) pid i)=.mk .fp16 (VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.value Native.scalar (Host.dequant16 a) (DequantGrid16.atProgram (Host.dequant16 a) initial pid) i)) ∧
      (∀ pid, pid<Geometry.count (DequantGrid16.geometry (Host.dequant16 a)) → ∀ i, Geometry.Active (DequantGrid16.geometry (Host.dequant16 a)) pid i →
        (Native.lower plan ((result.state.mem a.d.region (Geometry.offset (DequantGrid16.geometry (Host.dequant16 a)) pid i)).read .fp16) pid).eval isa=
          (Native.lower plan (VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.value Native.scalar (Host.dequant16 a) (DequantGrid16.atProgram (Host.dequant16 a) initial pid) i) pid).eval isa) ∧
      (∀ r o, (r≠a.d.region ∨ ∀ pid, pid<Geometry.count (DequantGrid16.geometry (Host.dequant16 a)) → ∀ i,
        Geometry.Active (DequantGrid16.geometry (Host.dequant16 a)) pid i → o≠Geometry.offset (DequantGrid16.geometry (Host.dequant16 a)) pid i) →
        result.state.mem r o=initial.mem r o) ∧
      (∀ x : Geometry.Cell (DequantGrid16.geometry (Host.dequant16 a)), ∃ pid, pid<Geometry.count (DequantGrid16.geometry (Host.dequant16 a)) ∧ Geometry.Owns (DequantGrid16.geometry (Host.dequant16 a)) pid x)) ∧
      result.returnedIdentity=Host.returned .dequant16 a ∧ result.outputDescriptor=a.d := by
  obtain ⟨hx,final,he,hpost⟩ := FPEquiv.dequant16 a route h route16 abi initial packed words plan isa jobs schedule
  refine ⟨hx,Host.observe a .dequant16 final,?_,hpost,rfl,?_⟩
  · simp [Host.complete,he]
  · simp [Host.observe]
#axiomsClean host_dequant16

end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.FPEquiv
