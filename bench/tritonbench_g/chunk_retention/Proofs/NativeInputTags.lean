import bench.tritonbench_g.chunk_retention.IO
import bench.tritonbench_g.chunk_retention.Sources.BackwardQKVReadTags

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
open VeriTile Triton FP.Structural KernelPacket KernelStaging BackwardQKVSource
noncomputable section
variable {α : Type} [Inhabited α]

/-- Finite addressed logical inputs establish exactly the native provenance
facts used by the actual QKV loads. Unrelated initial cells stay arbitrary. -/
theorem native_input_tags (ops : Ops α) (literal : ℝ→α) (a : Call)
    (hs : a.stage=.backwardQKV) (s : State (Term α)) (legal : Legal a s.pids s.numPids)
    (xs : Inputs α a.shape) (binding : (nativeIO ops literal a s.pids s.numPids).InputBinding xs s) :
    KernelNative.QKV.ReadTags a.shape s := by
  have head : s.pids 2<a.shape.B*a.shape.H := legal.1
  constructor
  · intro i hi
    let j : TileIndex [a.shape.B*a.shape.H,a.shape.T,a.shape.K] :=
      (⟨s.pids 2,head⟩,⟨row s i.1.val,hi.1⟩,⟨feature a.shape s i.2.1.val,hi.2⟩,())
    refine ⟨xs 1 j,?_⟩
    exact binding ⟨1,by simp [nativeIO,layout,inputs]⟩ j (Or.inr (Or.inr hs))
  · intro r hr f i hi
    let j : TileIndex [a.shape.B*a.shape.H,a.shape.T,a.shape.V] :=
      (⟨s.pids 2,head⟩,⟨row s i.1.val,hi.1⟩,⟨column a.shape f i.2.1.val,hi.2⟩,())
    rcases hr with rfl|rfl
    · exact ⟨xs 3 j,binding ⟨3,by simp [nativeIO,layout,inputs]⟩ j (Or.inr hs)⟩
    · exact ⟨xs 2 j,binding ⟨2,by simp [nativeIO,layout,inputs]⟩ j (Or.inr (Or.inr hs))⟩
  · intro e t he ht
    let j : TileIndex [a.shape.B*a.shape.H,a.shape.T,a.shape.K] :=
      (⟨s.pids 2,head⟩,⟨row s t.val,ht⟩,⟨feature a.shape s e.val,he⟩,())
    refine ⟨xs 0 j,?_⟩
    have h := binding ⟨0,by simp [nativeIO,layout,inputs]⟩ j (Or.inr (Or.inr hs))
    simpa only [nativeIO,layout,inputs,inputOffset,AddressedProgramIO.InputBinding,mapValue,j,
      Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using h

#axiomsClean native_input_tags
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
