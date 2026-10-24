import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.RealBackwardBlock
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.FPEquiv

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealTensor
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
set_option maxHeartbeats 6000000

/-- A thin binding of the executed kernel memory to the actual allocator descriptor. -/
structure Result where
  tensor : VeriTile.Torch.TensorMetadata
  state : State (WithBot ℝ)

def forward (x y : VeriTile.Torch.TensorMetadata) (ax : Fin x.shape.length)
    (X Y : RegionName) (c : LaunchConfig) (s : State (WithBot ℝ)) : Option Result :=
  if x.numel=0 then some ⟨y,s⟩ else
    (FPGrid.forwardRun Ideal.algebra X Y (dimensions x.shape ax) c s).map (fun t => ⟨y,t⟩)
def backward (x y : VeriTile.Torch.TensorMetadata) (ax : Fin x.shape.length)
    (P G Y : RegionName) (c : LaunchConfig) (s : State (WithBot ℝ)) : Option Result :=
  if x.numel=0 then some ⟨y,s⟩ else
    (FPGrid.backwardRun Ideal.algebra P G Y (dimensions x.shape ax) c s).map (fun t => ⟨y,t⟩)

/-- Independent target values plus exact frame, including every input cell. -/
def Output (Y : RegionName) (count : Nat) (v : Nat → ℝ) (s t : State (WithBot ℝ)) : Prop :=
  ∀ r o, t.mem r o=if r=Y ∧ o<count then .mk .real (v o : WithBot ℝ) else s.mem r o

def MathematicalInput (x : VeriTile.Torch.TensorMetadata) (ax : Fin x.shape.length)
    (X : RegionName) (words : Nat → BitVec 32) (s : State (WithBot ℝ)) : Prop :=
  Target.MathematicalWords x.numel words (FPReference.read s X) ∧
    (x.numel≠0 → RealForward.Defined (FPReference.read s X) (dimensions x.shape ax))

theorem output_values (Y : RegionName) (count : Nat) (v : Nat → ℝ) (s t : State (WithBot ℝ))
    (h : Output Y count v s t) : RealBackward.Read Y v count t := by
  intro o ho
  change (t.mem Y o).read .real=(v o : WithBot ℝ)
  rw [h Y o,if_pos ⟨rfl,ho⟩]
  rfl

theorem output_frame (Y : RegionName) (count : Nat) (v : Nat → ℝ) (s t : State (WithBot ℝ))
    (h : Output Y count v s t) : FPTrace.Frame Y count s t := by
  intro r o outside
  rw [h,if_neg outside]

theorem forward_complete (x y : VeriTile.Torch.TensorMetadata) (ax : Fin x.shape.length)
    (X Y : RegionName) (separate : X≠Y) (c : LaunchConfig) (s : State (WithBot ℝ))
    (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile (dimensions x.shape ax).N c.B)
    (defined : x.numel≠0 → RealForward.Defined (FPReference.read s X) (dimensions x.shape ax)) :
    ∃ result, forward x y ax X Y c s=some result ∧ result.tensor=y ∧
      Output Y x.numel (Target.flatForward (FPReference.read s X) (dimensions x.shape ax).N
        (dimensions x.shape ax).K) s result.state := by
  by_cases hz : x.numel=0
  · refine ⟨⟨y,s⟩,by simp [forward,hz],rfl,?_⟩
    intro r o
    simp [hz]
  · obtain ⟨hm,hn,hk,count⟩ := FPHost.dimensions_positive x.shape ax hz
    obtain ⟨t,he,hmem⟩ := FPGrid.forward_whole Ideal.algebra X Y (dimensions x.shape ax) c hB s s rfl
    have target := (RealForward.whole_graph X Y (FPReference.read s X) separate
      (dimensions x.shape ax) c (defined hz) hn hk hB hC hone s (fun _ => rfl)).2
    refine ⟨⟨y,t⟩,by simp [forward,hz,he],rfl,?_⟩
    intro r o
    rw [hmem,target]
    simp only [VeriTile.Torch.TensorMetadata.numel,count,RealForward.cells]

theorem backward_complete (x y : VeriTile.Torch.TensorMetadata) (ax : Fin x.shape.length)
    (P G Y : RegionName) (hPY : P≠Y) (hGY : G≠Y) (p g : Nat → ℝ)
    (c : LaunchConfig) (s : State (WithBot ℝ)) (hB : 0<c.B) (hC : 0<c.C)
    (hone : c.one=oneTile (dimensions x.shape ax).N c.B) (input : RealBackward.Input P G p g x.numel s) :
    ∃ result, backward x y ax P G Y c s=some result ∧ result.tensor=y ∧
      Output Y x.numel (Target.flatBackward p g (dimensions x.shape ax).N (dimensions x.shape ax).K) s result.state := by
  by_cases hz : x.numel=0
  · refine ⟨⟨y,s⟩,by simp [backward,hz],rfl,?_⟩
    intro r o
    simp [hz]
  · obtain ⟨hm,hn,hk,count⟩ := FPHost.dimensions_positive x.shape ax hz
    obtain ⟨t,he,hmem⟩ := FPGrid.backward_whole Ideal.algebra P G Y (dimensions x.shape ax) c s s rfl
    have hi : RealBackward.Input P G p g
        ((dimensions x.shape ax).M*(dimensions x.shape ax).N*(dimensions x.shape ax).K) s := by
      simpa only [←count] using input
    have target := (RealBackward.whole_graph P G Y p g hPY hGY
      (dimensions x.shape ax) c hn hk hB hC hone s hi).2
    refine ⟨⟨y,t⟩,by simp [backward,hz,he],rfl,?_⟩
    intro r o
    rw [hmem,target]
    simp only [VeriTile.Torch.TensorMetadata.numel,count,RealBackward.cells]

#axiomsClean forward_complete
#axiomsClean backward_complete
#axiomsClean output_values
#axiomsClean output_frame
#specNonCircular Output avoiding [forward,backward,FPGrid.forwardRun,FPGrid.backwardRun,FP.Structural.exec]
#specNonCircular MathematicalInput avoiding [forward,backward,FPGrid.forwardRun,FPGrid.backwardRun,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealTensor
