import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.IOBindings
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.Complete

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.SourceSafety
open VeriTile Triton FP.Structural Source Full Whole FPValue Interface
noncomputable section
set_option linter.unusedSimpArgs false

structure Accesses {α : Type} (p : Parameters) (bounds : RegionName → Nat) (s : State α) : Prop where
  query : ∀ i : QLane p,row p s i.1<p.length → qOffset p s i<bounds p.q
  bias1 : ∀ i : ScoreLane p,row p s i.1<p.length → b1Offset p s i<bounds p.bias
  key : ∀ c,c<p.blocks → ∀ i : KLane p,c*p.blockN+i.2.1.val<p.length → kOffset p s c i<bounds p.k
  value : ∀ c,c<p.blocks → ∀ i : VLane p,c*p.blockN+i.1.val<p.length → vOffset p s c i<bounds p.v
  bias0 : ∀ c,c<p.blocks → ∀ i : Fin p.blockM,row p s i<p.length → b0Offset p s c i<bounds p.bias
  output : ∀ i : QLane p,row p s i.1<p.length → outputOffset p s i<bounds p.output

theorem accesses {α : Type} (p : Parameters) (g : Guard p) (bounds : RegionName → Nat)
    (s : State α) (head : s.pids 1<p.batches*p.heads)
    (inputs : ∀ i : Point p,pointAddress p i<bounds (inputRegion p i.role))
    (outputs : ∀ b : Fin p.batches,∀ h : Fin p.heads,∀ n : Fin p.length,∀ d : Fin p.head,
      offset (oStrides p) b.val h.val n.val d.val<bounds p.output) : Accesses p bounds s := by
  obtain ⟨positive,hb,hh⟩ := head_decode p (s.pids 1) head
  refine ⟨?_,?_,?_,?_,?_,?_⟩
  · intro i active
    have backed := inputs ⟨0,⟨_,hb⟩,⟨_,hh⟩,⟨_,active⟩,i.2.1⟩
    simpa [pointAddress,inputStrides,inputRegion,offset,qStrides,qOffset,headBase,row] using backed
  · intro i active
    have width : i.2.1.val%p.biasLast+p.biasLast<128 := by rw [g.selected.biasLast];omega
    have backed := inputs ⟨3,⟨_,hb⟩,⟨_,hh⟩,⟨_,active⟩,⟨_,width⟩⟩
    simpa [pointAddress,inputStrides,inputRegion,offset,bStrides,b1Offset,biasBase,row] using backed
  · intro c hc i active
    have backed := inputs ⟨1,⟨_,hb⟩,⟨_,hh⟩,⟨_,active⟩,i.1⟩
    simpa [pointAddress,inputStrides,inputRegion,offset,kStrides,kOffset,kvBase,
      Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using backed
  · intro c hc i active
    have backed := inputs ⟨2,⟨_,hb⟩,⟨_,hh⟩,⟨_,active⟩,i.2.1⟩
    simpa [pointAddress,inputStrides,inputRegion,offset,vStrides,vOffset,valueBase] using backed
  · intro c hc i active
    have capacity := blocks_capacity p g
    have backed := inputs ⟨3,⟨_,hb⟩,⟨_,hh⟩,⟨_,active⟩,⟨c,by change c<128;omega⟩⟩
    simpa [pointAddress,inputStrides,inputRegion,offset,bStrides,b0Offset,biasBase,row] using backed
  · intro i active
    simpa [offset,oStrides,outputOffset,outputBase,row] using outputs ⟨_,hb⟩ ⟨_,hh⟩ ⟨_,active⟩ i.2.1

#axiomsClean accesses
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.SourceSafety
