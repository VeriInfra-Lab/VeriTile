import bench.tritonbench_g.lightning_attention.NativeIO

/-! Independent fixed-K8 native observation from named logical input windows.
The reference uses the existing source graph recurrences, never execution.
Overlapping padded windows are sampled at a deterministic representative;
the simultaneous IO bindings enforce their physical alias consistency. -/
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.NativeReference
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.LightningAttention.Native (Word Instructions Plan)
noncomputable section
open Classical

abbrev Lane (ports : List IOChannel) :=
  (i : Fin ports.length) × TileIndex ports[i].shape

def Sites (ports : List IOChannel)
    (index : (i : Fin ports.length) → TileIndex ports[i].shape → Nat)
    (r : RegionName) (o : Nat) : Prop :=
  ∃ lane : Lane ports, ports[lane.1].buf=r ∧ index lane.1 lane.2=o

def samples (ports : List IOChannel)
    (index : (i : Fin ports.length) → TileIndex ports[i].shape → Nat)
    (words : Lane ports → Word) (r : RegionName) (o : Nat) : Word :=
  if h : Sites ports index r o then words (Classical.choose h) else 0

def forwardSample (d : Host.Dimensions) (vals : ChannelValues Word (NativeIO.forwardInputs d)) :
    Lane (NativeIO.forwardInputs d) → Word
  | ⟨⟨0,_⟩,j⟩ | ⟨⟨1,_⟩,j⟩ | ⟨⟨2,_⟩,j⟩ | ⟨⟨3,_⟩,j⟩ => vals _ j

def backwardSample (d : Host.Dimensions) (vals : ChannelValues Word (NativeIO.backwardInputs d)) :
    Lane (NativeIO.backwardInputs d) → Word
  | ⟨⟨0,_⟩,j⟩ | ⟨⟨1,_⟩,j⟩ | ⟨⟨2,_⟩,j⟩ | ⟨⟨3,_⟩,j⟩
  | ⟨⟨4,_⟩,j⟩ | ⟨⟨5,_⟩,j⟩ | ⟨⟨6,_⟩,j⟩ => vals _ j

def forwardScope (d : Host.Dimensions) : RegionName → Nat → Prop :=
  Sites (NativeIO.forwardInputs d) (NativeIO.forwardInputIndex d)

def backwardScope (d : Host.Dimensions) : RegionName → Nat → Prop :=
  Sites (NativeIO.backwardInputs d) (NativeIO.backwardInputIndex d)

def forwardWords (d : Host.Dimensions) (vals : ChannelValues Word (NativeIO.forwardInputs d)) :
    Native.WordMemory :=
  samples (NativeIO.forwardInputs d) (NativeIO.forwardInputIndex d) (forwardSample d vals)

def backwardWords (d : Host.Dimensions) (vals : ChannelValues Word (NativeIO.backwardInputs d)) :
    Native.WordMemory :=
  samples (NativeIO.backwardInputs d) (NativeIO.backwardInputIndex d) (backwardSample d vals)

def forward (ops : Instructions) (plan : Plan) (d : Host.Dimensions)
    (vals : ChannelValues Word (NativeIO.forwardInputs d)) :
    ChannelValues Word (NativeIO.forwardOutputs d) :=
  fun ⟨0,_⟩ j => Native.forwardReference ops plan d (forwardWords d vals)
    "output" (NativeIO.paddedIndex d j)

def backward (ops : Instructions) (plan : Plan) (d : Host.Dimensions)
    (vals : ChannelValues Word (NativeIO.backwardInputs d)) :
    ChannelValues Word (backwardChannels d) :=
  fun o j => match o with
  | ⟨0,_⟩ => Native.backwardReference ops plan d (backwardWords d vals) "queryGradient" (denseIndex j)
  | ⟨1,_⟩ => Native.backwardReference ops plan d (backwardWords d vals) "keyGradient" (denseIndex j)
  | ⟨2,_⟩ => Native.backwardReference ops plan d (backwardWords d vals) "valueGradient" (denseIndex j)

#specNonCircular forward avoiding [runMaterializedSchedule,MaterializedInvocation.run,
  FP.Structural.exec,FP.Structural.run,Native.forwardExecute,Native.stageExecute,Native.gridExecute]
#specNonCircular backward avoiding [runMaterializedSchedule,MaterializedInvocation.run,
  FP.Structural.exec,FP.Structural.run,Native.backwardExecute,Native.stageExecute,Native.gridExecute]
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.NativeReference

