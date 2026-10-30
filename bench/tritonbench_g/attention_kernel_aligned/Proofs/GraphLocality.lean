import bench.tritonbench_g.attention_kernel_aligned.Proofs.IOBindings
import bench.tritonbench_g.attention_kernel_aligned.Proofs.Specification

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Locality
open VeriTile Triton FP.Structural Full Whole FPValue Interface KernelOnly.Specification
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem loads (P : Primitives α) (p : Parameters) (g : Guard p)
    (r t : RegionName → Nat → α) (agree : Agree p r t)
    (x b h c : Nat) (hb : b<p.batches) (hh : h<p.heads) (hc : c<p.blocks) :
    qValue P p r x b h=qValue P p t x b h ∧
    kValue P p r x b h c=kValue P p t x b h c ∧
    vValue P p r x b h c=vValue P p t x b h c ∧
    b0Value P p r x b h c=b0Value P p t x b h c ∧
    b1Value P p r x b h=b1Value P p t x b h := by
  refine ⟨?_,?_,?_,?_,?_⟩
  · funext i
    by_cases hr : x*p.blockM+i.1.val<p.length
    · have same := agree ⟨0,⟨b,hb⟩,⟨h,hh⟩,⟨_,hr⟩,i.2.1⟩
      simpa [qValue,qOffset,headBase,row,inputRegion,pointAddress,inputStrides,offset,qStrides,hr] using same
    · simp [qValue,row,hr]
  · funext i
    by_cases hn : c*p.blockN+i.2.1.val<p.length
    · have same := agree ⟨1,⟨b,hb⟩,⟨h,hh⟩,⟨_,hn⟩,i.1⟩
      simp only [kValue,if_pos hn]
      simpa [kOffset,kvBase,inputRegion,pointAddress,inputStrides,offset,kStrides,
        Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using same
    · simp [kValue,hn]
  · funext i
    by_cases hn : c*p.blockN+i.1.val<p.length
    · have same := agree ⟨2,⟨b,hb⟩,⟨h,hh⟩,⟨_,hn⟩,i.2.1⟩
      simpa [vValue,vOffset,valueBase,inputRegion,pointAddress,inputStrides,offset,vStrides,hn] using same
    · simp [vValue,hn]
  · funext i
    by_cases hr : x*p.blockM+i.val<p.length
    · have cap := blocks_capacity p g
      have same := agree ⟨3,⟨b,hb⟩,⟨h,hh⟩,⟨_,hr⟩,⟨c,by change c<128;omega⟩⟩
      simpa [b0Value,b0Offset,biasBase,row,inputRegion,pointAddress,inputStrides,offset,bStrides,hr] using same
    · simp [b0Value,row,hr]
  · funext i
    by_cases hr : x*p.blockM+i.1.val<p.length
    · have width : i.2.1.val%p.biasLast+p.biasLast<128 := by rw [g.selected.biasLast];omega
      have same := agree ⟨3,⟨b,hb⟩,⟨h,hh⟩,⟨_,hr⟩,⟨_,width⟩⟩
      simpa [b1Value,b1Offset,biasBase,row,inputRegion,pointAddress,inputStrides,offset,bStrides,hr] using same
    · simp [b1Value,row,hr]

theorem advance_agree (P : Primitives α) (p : Parameters) (g : Guard p)
    (r t : RegionName → Nat → α) (agree : Agree p r t)
    (x b h c : Nat) (hb : b<p.batches) (hh : h<p.heads) (hc : c<p.blocks)
    (st : Accumulator α p) : advance P p r x b h c st=advance P p t x b h c st := by
  obtain ⟨hq,hk,hv,hb0,hb1⟩ := loads P p g r t agree x b h c hb hh hc
  have qs : qScaled P p r x b h=qScaled P p t x b h := by
    funext i
    simp only [qScaled,hq]
  have sc : score32 P p r x b h c=score32 P p t x b h c := by
    funext i
    simp only [score32,score,rawScore,qkDot,qs,hk,hb0,hb1]
  have mx : newMax P p r x b h c st=newMax P p t x b h c st := by
    funext i
    simp only [newMax,sc]
  have al : alphaValue P p r x b h c st=alphaValue P p t x b h c st := by
    funext i
    simp only [alphaValue,mx]
  have wt : weights P p r x b h c st=weights P p t x b h c st := by
    funext i
    simp only [weights,sc,mx]
  have wd : weightedDot P p r x b h c st=weightedDot P p t x b h c st := by
    funext i
    simp only [weightedDot,wt,hv]
  simp only [advance,mx,al,wt,wd]

theorem rounds_agree (P : Primitives α) (p : Parameters) (g : Guard p)
    (r t : RegionName → Nat → α) (agree : Agree p r t)
    (x b h : Nat) (hb : b<p.batches) (hh : h<p.heads) :
    ∀ n,n≤p.blocks → rounds P p r x b h n=rounds P p t x b h n := by
  intro n
  induction n with
  | zero => intro _;rfl
  | succ n ih =>
    intro hn
    rw [rounds,rounds,ih (by omega)]
    exact advance_agree P p g r t agree x b h n hb hh (by omega) _

theorem output_agree (P : Primitives α) (p : Parameters) (g : Guard p)
    (r t : RegionName → Nat → α) (agree : Agree p r t)
    (x b h : Nat) (hb : b<p.batches) (hh : h<p.heads) :
    outputValue P p r x b h=outputValue P p t x b h := by
  funext i
  simp only [outputValue,rounds_agree P p g r t agree x b h hb hh p.blocks (by omega)]

#axiomsClean output_agree
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Locality
