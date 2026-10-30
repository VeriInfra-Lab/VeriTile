import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.DirectLayout
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.RealTrials
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.Cleanup

namespace VeriTile.Bench.Repairs.BmmChunkFwd.Direct
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
open Classical
set_option maxHeartbeats 4000000

def finish (h : Host.Call) (A : FlatAlloc) (s : BlockState) : BlockState :=
  if h.causal then
    { s with mem := fun r off =>
        if r=A.flat then Cleanup.runAt h (A.base h.Out) (MemCell.real 0) (s.mem r) off
        else s.mem r off }
  else s

def RealRuns (p : Metadata) (a : Allocation p) (d : RealData)
    (trials : List Host.Config) (chosen : Host.Config) (final : BlockState) : Prop :=
  let A := flat p a
  if (shape p).outputSize=0 then trials=[] ∧ final=A.flattenState (realState p d) else
  ∃ after native, RealTrials.Runs (call p) A trials (A.flattenState (realState p d)) after ∧
    Nonempty (Kernel.GridLaunchedOrdinary
      (A.flattenKernel (Kernels.source (call p) chosen).eraseDType.toAlgKernel)
      ((call p).grid chosen) after native) ∧ final=finish (call p) A native

def OutputCell (h : Host.Call) (A : FlatAlloc) (r : RegionName) (off : Nat) : Prop :=
  r=A.flat ∧ ∃ i,i<h.shape.outputSize ∧ off=A.addr h.Out i

theorem written_inside (h : Host.Call) (c : Host.Config) (A : FlatAlloc) (r : RegionName) (off : Nat)
    (written : Host.Written h c A r off) : OutputCell h A r off := by
  obtain ⟨p,hr,i,hi,eq⟩ := written
  exact ⟨hr,Host.offset h c p i,Host.output_bound h c p i hi,eq.symm⟩

theorem finish_frame (h : Host.Call) (A : FlatAlloc) (s : BlockState) (r : RegionName) (off : Nat)
    (outside : ¬OutputCell h A r off) : (finish h A s).mem r off=s.mem r off := by
  unfold finish
  split
  · dsimp only
    by_cases hr : r=A.flat
    · rw [if_pos hr]
      apply Cleanup.runAt_frame
      intro i hi eq
      obtain ⟨p,row,col,valid,he⟩ := (Cleanup.member h i).mp ((Cleanup.image h i).mpr hi)
      apply outside
      refine ⟨hr,i,?_,eq⟩
      rw [he]
      exact Cleanup.offset_bound h p row col valid
    · simp [hr]
  · rfl

theorem finish_value (h : Host.Call) (A : FlatAlloc) (s : BlockState)
    (b ch g r n : Nat) (hb : b<h.shape.batch) (hch : ch<h.shape.chunks)
    (hg : g<h.shape.groups) (hr : r<h.shape.chunk) (hn : n<h.shape.chunk) :
    (finish h A s).readMem A.flat (A.addr h.Out (Host.returnedOffset h b ch g r n))=
      if h.causal=Bool.true ∧ r<n then 0 else s.readMem A.flat (A.addr h.Out (Host.returnedOffset h b ch g r n)) := by
  have offset : Host.returnedOffset h b ch g r n=
      ((b*(h.shape.chunks*h.shape.groups)+(ch*h.shape.groups+g))*h.shape.chunk+r)*h.shape.chunk+n := by
    unfold Host.returnedOffset; ring
  have upper : Cleanup.Upper h (Host.returnedOffset h b ch g r n) ↔ r<n := by
    rw [offset]
    exact Cleanup.upper_at h b (ch*h.shape.groups+g) r n hb (Host.mul_add_bound hch hg) hr hn
  cases causal : h.causal
  · simp [finish,causal]
  · simp only [finish,causal,↓reduceIte,BlockState.readMem,FlatAlloc.addr]
    rw [Cleanup.runAt_value]
    by_cases triangle : r<n
    · simp only [if_pos (upper.mpr triangle),triangle,and_self,ite_true]
      rfl
    · simp only [if_neg (fun hu => triangle (upper.mp hu)),triangle,and_false,ite_false]

theorem mathematical_link (p : Metadata) (d : RealData) (b ch g r n : Nat) :
    Mathematical.product p d b ch g r n=
      if p.causal=Bool.true ∧ r<n then 0 else Host.expected (call p) (realInputs p d) b ch g r n := by
  simp only [Mathematical.product,Host.expected,call,realInputs,Mathematical.inputIndex,inputIndex,Nat.add_assoc]
  rfl

theorem empty_coordinate (p : Metadata) (empty : (shape p).outputSize=0)
    (b ch g r n : Nat) (hb : b<(shape p).batch) (hch : ch<(shape p).chunks)
    (hg : g<(shape p).groups) (hr : r<(shape p).chunk) (_hn : n<(shape p).chunk) : False := by
  have bound := Host.mul_add_bound (Host.mul_add_bound (Host.mul_add_bound hb
    (Host.mul_add_bound hch hg)) hr) hr
  change _<(shape p).outputSize at bound
  omega

/-- Full arbitrary-Real returned tensor for every finite tuning trace. Each
trial executes on its predecessor's actual flat memory; all lower coordinates
are written by the chosen configuration, and the final clearing launch makes
all upper coordinates zero. There is no result or execution premise. -/
theorem real_complete (p : Metadata) (h : InputContract p) (a : Allocation p) (d : RealData)
    (trials : List Host.Config) (legal : ∀ c∈trials,c∈Host.configurations)
    (emptyTrials : (shape p).outputSize=0 → trials=[])
    (chosen : Host.Config) (hc : chosen∈Host.configurations) :
    ∃ final, RealRuns p a d trials chosen final ∧
      (∀ b ch g r n,b<(shape p).batch → ch<(shape p).chunks → g<(shape p).groups →
        r<(shape p).chunk → n<(shape p).chunk →
        final.readMem (flat p a).flat ((flat p a).addr (call p).Out (Host.returnedOffset (call p) b ch g r n))=
          Mathematical.product p d b ch g r n) ∧
      (∀ r off,¬OutputCell (call p) (flat p a) r off →
        final.mem r off=((flat p a).flattenState (realState p d)).mem r off) := by
  by_cases empty : (shape p).outputSize=0
  · refine ⟨(flat p a).flattenState (realState p d),?_,?_,fun _ _ _ => rfl⟩
    · simp [RealRuns,empty,emptyTrials empty]
    · intro b ch g r n hb hch hg hr hn
      exact (empty_coordinate p empty b ch g r n hb hch hg hr hn).elim
  · have positive : 0<(shape p).outputSize := by omega
    have binding := real_binding p h d positive
    have disjoint := flat_disjoint p h a
    have regions : (flat p a).regions=[(call p).A,(call p).B,Region.cast (call p).Seq,(call p).Out] := rfl
    have closed : ∀ r,r∉(flat p a).regions → (flat p a).extent r=0 := extent_closed p
    have out : (flat p a).extent (call p).Out=(shape p).outputSize := rfl
    obtain ⟨after,runs,inputs,undef,trialFrame⟩ := RealTrials.execute (call p) trials legal (realInputs p d)
      (realState p d) (flat p a) disjoint regions closed out rfl binding
    obtain ⟨native,launch,values,frame⟩ := Host.returned_launch (call p) chosen hc Bool.false
      (realInputs p d) after (Host.tensor_pins _ _ _ _ _ inputs) (flat p a)
      ⟨disjoint,regions,closed,out,undef,Host.tensor_bounds _ _ _ _ _ inputs out⟩
    refine ⟨finish (call p) (flat p a) native,?_,?_,?_⟩
    · simp only [RealRuns,if_neg empty]
      exact ⟨(flat p a).flattenState after,native,runs,⟨launch⟩,rfl⟩
    · intro b ch g r n hb hch hg hr hn
      rw [finish_value _ _ _ b ch g r n hb hch hg hr hn,mathematical_link]
      change (if p.causal=Bool.true ∧ r<n then 0 else _)=_
      split_ifs with upper
      · rfl
      · apply values b ch g r n hb hch hg hr hn
        by_cases lower : n≤r
        · exact Cleanup.lower_active (call p) chosen hc r n lower
        · left
          have triangle : r<n := by omega
          change p.causal=Bool.false
          cases hs : p.causal <;> simp_all
    · intro r off outside
      rw [finish_frame _ _ _ r off outside,frame r off (fun hw => outside (written_inside _ _ _ _ _ hw))]
      apply trialFrame r off
      rintro ⟨c,_,written⟩
      exact outside (written_inside _ _ _ _ _ written)

#axiomsClean real_complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkFwd.Direct
