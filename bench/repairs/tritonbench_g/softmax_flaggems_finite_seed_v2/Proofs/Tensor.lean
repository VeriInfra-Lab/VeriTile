import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.Proofs.Kernel
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.Contract
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.FPHostCore

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Tensor
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
open Native
noncomputable section
set_option maxHeartbeats 6000000

/-- Caller-visible raw words. A later kernel receives this memory through
Native.entry and therefore cannot reinterpret a producer expression. -/
structure Result where
  tensor : VeriTile.Torch.TensorMetadata
  words : RegionName → Nat → Word

def pack (ops : NativeProfile.Operations) (scalar : ℝ → Word) (S : Schedule)
    (y : VeriTile.Torch.TensorMetadata) (state : State Term) : Result :=
  ⟨y,freeze (Scalar.instructions ops scalar) S state⟩

def forward (ops : NativeProfile.Operations) (scalar : ℝ → Word) (R : Reductions) (S : Schedule)
    (x y : VeriTile.Torch.TensorMetadata) (ax : Fin x.shape.length) (X Y : RegionName) (c : LaunchConfig)
    (memory : RegionName → Nat → Word) : Option Result :=
  if x.numel=0 then some ⟨y,memory⟩ else
    (SoftmaxFlaggemsFiniteSeed.FPGrid.forwardRun (algebra R) X Y (dimensions x.shape ax) c (entry memory)).map (pack ops scalar S y)
def backward (ops : NativeProfile.Operations) (scalar : ℝ → Word) (R : Reductions) (S : Schedule)
    (x y : VeriTile.Torch.TensorMetadata) (ax : Fin x.shape.length) (P G Y : RegionName) (c : LaunchConfig)
    (memory : RegionName → Nat → Word) : Option Result :=
  if x.numel=0 then some ⟨y,memory⟩ else
    (SoftmaxFlaggemsFiniteSeed.FPGrid.backwardRun (algebra R) P G Y (dimensions x.shape ax) c (entry memory)).map (pack ops scalar S y)

def Output (Y : RegionName) (count : Nat) (expected : Nat → Word) (before : RegionName → Nat → Word)
    (result : Result) : Prop :=
  ∀ r o, result.words r o=if r=Y ∧ o<count then expected o else before r o

def forwardWords (ops : NativeProfile.Operations) (scalar : ℝ → Word) (R : Reductions) (S : Schedule)
    (X Y : RegionName) (d : Dimensions) (c : LaunchConfig) (memory : RegionName → Nat → Word) (o : Nat) : Word :=
  Ksoftmax.Graph.eval ops (NativeProfile.lower scalar S (Y,o) (((Kernel.forwardTarget R X Y d c memory).mem Y o).read .real))
def backwardWords (ops : NativeProfile.Operations) (scalar : ℝ → Word) (R : Reductions) (S : Schedule)
    (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig) (memory : RegionName → Nat → Word) (o : Nat) : Word :=
  Ksoftmax.Graph.eval ops (NativeProfile.lower scalar S (Y,o) (((Kernel.backwardTarget R P G Y d c memory).mem Y o).read .real))

theorem forward_complete (ops : NativeProfile.Operations) (scalar : ℝ → Word) (R : Reductions) (S : Schedule)
    (x y : VeriTile.Torch.TensorMetadata) (ax : Fin x.shape.length) (X Y : RegionName) (c : LaunchConfig)
    (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile (dimensions x.shape ax).N c.B)
    (memory : RegionName → Nat → Word) :
    ∃ result, forward ops scalar R S x y ax X Y c memory=some result ∧ result.tensor=y ∧
      Output Y x.numel (forwardWords ops scalar R S X Y (dimensions x.shape ax) c memory) memory result := by
  by_cases hz : x.numel=0
  · refine ⟨⟨y,memory⟩,by simp [forward,hz],rfl,?_⟩
    intro r o; simp [hz]
  · obtain ⟨hm,hn,hk,count⟩ := SoftmaxFlaggemsFiniteSeed.FPHost.dimensions_positive x.shape ax hz
    change x.numel=(dimensions x.shape ax).M*(dimensions x.shape ax).N*(dimensions x.shape ax).K at count
    obtain ⟨actual,he,hmem,values⟩ := Kernel.forward_complete ops scalar R S X Y (dimensions x.shape ax) c hB memory
    refine ⟨pack ops scalar S y actual,by simp [forward,hz,he],rfl,?_⟩
    intro r o
    by_cases hit : r=Y ∧ o<x.numel
    · rw [if_pos hit]
      rw [hit.1]
      exact values o
    · rw [if_neg hit]
      exact Kernel.forward_frame (Scalar.instructions ops scalar) R S X Y (dimensions x.shape ax) c hn hk hB hC hone
        memory actual hmem r o (by simpa only [←count] using hit)

theorem backward_complete (ops : NativeProfile.Operations) (scalar : ℝ → Word) (R : Reductions) (S : Schedule)
    (x y : VeriTile.Torch.TensorMetadata) (ax : Fin x.shape.length) (P G Y : RegionName) (c : LaunchConfig)
    (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile (dimensions x.shape ax).N c.B)
    (memory : RegionName → Nat → Word) :
    ∃ result, backward ops scalar R S x y ax P G Y c memory=some result ∧ result.tensor=y ∧
      Output Y x.numel (backwardWords ops scalar R S P G Y (dimensions x.shape ax) c memory) memory result := by
  by_cases hz : x.numel=0
  · refine ⟨⟨y,memory⟩,by simp [backward,hz],rfl,?_⟩
    intro r o; simp [hz]
  · obtain ⟨hm,hn,hk,count⟩ := SoftmaxFlaggemsFiniteSeed.FPHost.dimensions_positive x.shape ax hz
    change x.numel=(dimensions x.shape ax).M*(dimensions x.shape ax).N*(dimensions x.shape ax).K at count
    obtain ⟨actual,he,hmem,values⟩ := Kernel.backward_complete ops scalar R S P G Y (dimensions x.shape ax) c memory
    refine ⟨pack ops scalar S y actual,by simp [backward,hz,he],rfl,?_⟩
    intro r o
    by_cases hit : r=Y ∧ o<x.numel
    · rw [if_pos hit]
      rw [hit.1]
      exact values o
    · rw [if_neg hit]
      exact Kernel.backward_frame (Scalar.instructions ops scalar) R S P G Y (dimensions x.shape ax) c hn hk hB hC hone
        memory actual hmem r o (by simpa only [←count] using hit)

#axiomsClean forward_complete
#axiomsClean backward_complete
#specNonCircular forwardWords avoiding [forward,SoftmaxFlaggemsFiniteSeed.FPGrid.forwardRun,FP.Structural.exec]
#specNonCircular backwardWords avoiding [backward,SoftmaxFlaggemsFiniteSeed.FPGrid.backwardRun,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Tensor
