import bench.tritonbench_g.chunk_cumsum_vector.Proofs.FPValues
import VeriTile.Triton.Float.ProgramEquivalence
namespace VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIO
open VeriTile.Triton FP.Structural FPValues
noncomputable section
variable {α : Type} [Inhabited α]

abbrev inputChannels (h : Call) : List IOChannel := [⟨h.inputDType.toTileDType,[h.T,BS],h.input⟩]
abbrev outputChannels (h : Call) : List IOChannel := [⟨h.outputDType.toTileDType,[h.T,BS],h.output⟩]



def inputRead (h : Call) (p : Geometry.Program h) (i : Fin (inputChannels h).length) :
    ChannelValues α (inputChannels h) → TileIndex (inputChannels h)[i].shape → Nat :=
  match i with
  | ⟨0,_⟩ => fun _ j => h.inputStrides.address h.H p.row j.1.val (p.feature*BS+j.2.1.val)
  | ⟨_+1,impossible⟩ => False.elim (by simpa using impossible)
def inputMask (h : Call) (p : Geometry.Program h) (i : Fin (inputChannels h).length) :
    ChannelValues α (inputChannels h) → TileIndex (inputChannels h)[i].shape → Prop :=
  match i with
  | ⟨0,_⟩ => fun _ j => p.feature*BS+j.2.1.val < h.S
  | ⟨_+1,impossible⟩ => False.elim (by simpa using impossible)
def outputWrite (h : Call) (p : Geometry.Program h) (i : Fin (outputChannels h).length) :
    ChannelValues α (inputChannels h) → TileIndex (outputChannels h)[i].shape → Nat :=
  match i with
  | ⟨0,_⟩ => fun _ j => h.outputStrides.address h.H p.row j.1.val (p.feature*BS+j.2.1.val)
  | ⟨_+1,impossible⟩ => False.elim (by simpa using impossible)
def outputMask (h : Call) (p : Geometry.Program h) (i : Fin (outputChannels h).length) :
    ChannelValues α (inputChannels h) → TileIndex (outputChannels h)[i].shape → Prop :=
  match i with
  | ⟨0,_⟩ => fun _ j => p.feature*BS+j.2.1.val < h.S
  | ⟨_+1,impossible⟩ => False.elim (by simpa using impossible)










def invocation (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config)
    (p : Geometry.Program h) (commuted : Bool) : Invocation α where
  kernel := if commuted then Kernels.sumLeft h c else Kernels.source h c
  backend := machine M dot
  pids := fun ax => if ax=0 then p.feature else if ax=1 then p.row else 0
  setPids := fun ax => decide (ax<2)

abbrev io (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config)
    (p : Geometry.Program h) : ProgramIO α α where
  inputs := inputChannels h
  outputs := outputChannels h
  scratch := []
  buffers := [h.input,h.output]
  read := inputRead h p
  inputMask := inputMask h p
  write := outputWrite h p
  outputMask := outputMask h p
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [invocation M dot h c p Bool.false]
  encode := id
  observe := fun _ _ => id

def candidate (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config)
    (p : Geometry.Program h) : (io M dot h c p).Variant where
  schedule := fun _ => [invocation M dot h c p Bool.true]
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIO
