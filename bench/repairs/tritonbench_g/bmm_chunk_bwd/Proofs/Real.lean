import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.Layout
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.RealTrials
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Target

namespace VeriTile.Bench.Repairs.BmmChunkBwd.Direct
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
noncomputable section
open Classical
open scoped BigOperators
set_option maxHeartbeats 6000000

abbrev RealData := Mathematical.Inputs

def realState (_p : Metadata) (d : RealData) : BlockState where
  mem := fun r off => if r="A" then .real (d.a off) else if r="Dout" then .real (d.dout off)
    else if r="Residual" then .real (d.residual off) else .real (d.output off)
  regs := fun _ _ _ => none
  pids := fun _ => 0

theorem expected_link (p : Metadata) (d : RealData) (c : Tuning) (q : HostGrid.Pid)
    (i : TileIndex [c.BM,c.BN]) (hg : head p q<groups p) :
    HostGrid.expected c.BM c.BN (HostContract.parameters (dense p)) (realState p d) q i=
      Mathematical.chunkProduct p d q.batch (slice p q) (head p q) (row p c q i.1) (column p c q i.2.1) := by
  have hd (k : Fin (chunk p)) :
      (HostGrid.inputs c.BM c.BN (HostContract.parameters (dense p)) (realState p d) q).dout i.1 k=
        d.dout (doutIndex p q.batch (slice p q) (head p q) k.val (row p c q i.1)) := by
    change (realState p d).readMem "Dout"
      ((HostGrid.io Bool.false c.BM c.BN c.BCS (HostContract.parameters (dense p))).doutWindow q.tile q.batch q.chunkHead (i.1,k,PUnit.unit))=_
    rw [dout_offset p c q _ hg]
    rfl
  have ha (k : Fin (chunk p)) :
      (HostGrid.inputs c.BM c.BN (HostContract.parameters (dense p)) (realState p d) q).a k i.2.1=
        d.a (tensorIndex p q.batch (slice p q*chunk p+k.val) (head p q) (column p c q i.2.1)) := by
    change (realState p d).readMem "A"
      ((HostGrid.io Bool.false c.BM c.BN c.BCS (HostContract.parameters (dense p))).aWindow q.tile q.batch q.chunkHead (k,i.2.1,PUnit.unit))=_
    rw [a_offset p c q _ hg]
    rfl
  have hr (present : p.hasResidual=Bool.true) :
      (HostGrid.inputs c.BM c.BN (HostContract.parameters (dense p)) (realState p d) q).residual i.1 i.2.1=
        d.residual (tensorIndex p q.batch (slice p q*chunk p+row p c q i.1) (head p q) (column p c q i.2.1)) := by
    change (realState p d).readMem "Residual"
      ((HostGrid.io Bool.false c.BM c.BN c.BCS (HostContract.parameters (dense p))).residualWindow q.tile q.batch q.chunkHead (i.1,i.2.1,PUnit.unit))=_
    rw [residual_offset p c q _ hg present]
    rfl
  unfold HostGrid.expected Mathematical.chunkProduct
  simp only [source_group]
  change (∑ k : Fin (chunk p),if k.val<min (chunk p) (seqlen p-slice p q*chunk p) then _ else 0)+
    (if p.hasResidual then _ else 0)=_
  congr 1
  · apply Finset.sum_congr rfl
    intro k _
    split_ifs
    · rw [hd,ha]; rfl
    · rfl
  · cases present : p.hasResidual
    · simp [present]
    · simp only [present,↓reduceIte]
      exact hr present

theorem coordinate_covered (p : Metadata) (h : InputContract p) (d : RealData) (choice : Fin 9)
    (b s g n : Nat) (hb : b<batch p) (hs : s<seqlen p) (hg : g<groups p) (hn : n<width p) :
    ∃ q∈HostGrid.programs (HostContract.grid (dense p) (tuning choice)),
      ∃ i : TileIndex [(tuning choice).BM,(tuning choice).BN],
        HostGrid.active (tuning choice).BM (tuning choice).BN (HostContract.parameters (dense p)) q i ∧
        HostGrid.address (tuning choice).BM (tuning choice).BN (HostContract.parameters (dense p)) q i=tensorIndex p b s g n ∧
        HostGrid.expected (tuning choice).BM (tuning choice).BN (HostContract.parameters (dense p)) (realState p d) q i=
          Mathematical.product p d b s g n := by
  have dims := HostContract.tuning_positive choice
  obtain ⟨hc,hr,rebuild⟩ := HostCoverage.chunk_for_sequence (seqlen p) (chunk p) s h.chunk hs
  rw [←h.shapes.2] at hc
  have hp : 0<(HostContract.parameters (dense p)).groups := by simpa using Nat.zero_lt_of_lt hg
  obtain ⟨q,hq,i,hi,eb,ec,eg,er,en⟩ := HostCoverage.covers (tuning choice).BM (tuning choice).BN
    (batch p) (chunks p) (HostContract.parameters (dense p)) dims.1 dims.2.1 hp
    b (s/chunk p) g (s%chunk p) n hb hc (by simpa using hg) hr hn
  simp only [source_slice,source_head,source_row,source_column] at ec eg er en
  refine ⟨q,hq,i,hi,?_,?_⟩
  · rw [output_offset p _ q i (by rw [eg]; exact hg),eb,ec,eg,er,en,rebuild]
  · rw [expected_link p d _ q i (by rw [eg]; exact hg),eb,ec,eg,er,en]
    rfl

def OutputCell (p : Metadata) (a : Allocation p) (r : RegionName) (off : Nat) : Prop :=
  r=(flat p a).flat ∧ ∃ k,k<(p.tensor .a).numel ∧ off=(flat p a).addr (call p).Out k

def executeReal (p : Metadata) (a : Allocation p) (d : RealData)
    (trials : List (Fin 9)) (chosen : Fin 9) : Option BlockState :=
  if (p.tensor .a).numel=0 then some ((flat p a).flattenState (realState p d)) else do
    let after ← RealTrials.flatHistory (call p) (flat p a) trials ((flat p a).flattenState (realState p d))
    RealTrials.flatOne (call p) (flat p a) chosen after

theorem expected_frame (p : Metadata) (c : Tuning) (s t : BlockState)
    (frame : ∀ r off,r≠(call p).Out → t.mem r off=s.mem r off) (q : HostGrid.Pid)
    (i : TileIndex [c.BM,c.BN]) :
    HostGrid.expected c.BM c.BN (HostContract.parameters (call p)) t q i=
      HostGrid.expected c.BM c.BN (HostContract.parameters (call p)) s q i := by
  have ha (off : Nat) : t.readMem (call p).A off=s.readMem (call p).A off := by
    simp only [BlockState.readMem,frame _ off (by simp [call] : (call p).A≠(call p).Out)]
  have hd (off : Nat) : t.readMem (call p).Dout off=s.readMem (call p).Dout off := by
    simp only [BlockState.readMem,frame _ off (by simp [call] : (call p).Dout≠(call p).Out)]
  have hr (off : Nat) : t.readMem (call p).Residual off=s.readMem (call p).Residual off := by
    simp only [BlockState.readMem,frame _ off (by simp [call] : (call p).Residual≠(call p).Out)]
  simp only [HostGrid.expected,HostGrid.inputs,HostContract.parameters,ha,hd,hr]

/-- Complete ordinary flat-memory execution over arbitrary Reals. All actual
trial outputs feed the next launch, including supplied output storage. -/
theorem real_complete (p : Metadata) (h : InputContract p) (a : Allocation p) (d : RealData)
    (trials : List (Fin 9)) (chosen : Fin 9) :
    ∃ final,executeReal p a d trials chosen=some final ∧
      (∀ b s g n,b<batch p → s<seqlen p → g<groups p → n<width p →
        final.readMem (flat p a).flat ((flat p a).addr (call p).Out (tensorIndex p b s g n))=
          Mathematical.product p d b s g n) ∧
      (∀ r off,¬OutputCell p a r off →
        final.mem r off=((flat p a).flattenState (realState p d)).mem r off) := by
  by_cases empty : (p.tensor .a).numel=0
  · refine ⟨(flat p a).flattenState (realState p d),by simp [executeReal,empty],?_,fun _ _ _ => rfl⟩
    intro b s g n hb hs hg hn
    have bound := tensor_bound p h b s g n hb hs hg hn
    omega
  · have positive : 0<(p.tensor .a).numel := by omega
    have layouts (c : Fin 9) := layout p h a positive (tuning c)
    obtain ⟨after,history,trialFrame⟩ := RealTrials.history_frame (call p) (flat p a) trials
      (fun c _ => layouts c) (realState p d)
    have flatHistory : RealTrials.flatHistory (call p) (flat p a) trials ((flat p a).flattenState (realState p d))=
        some ((flat p a).flattenState after) := by
      rw [RealTrials.history_flatten (call p) (flat p a) trials (fun c _ => layouts c),history]
      rfl
    obtain ⟨final,launch,values,frame⟩ := HostContract.complete (call p) chosen Bool.false (flat p a) (layouts chosen) after
    have step : RealTrials.flatOne (call p) (flat p a) chosen ((flat p a).flattenState after)=some final := by
      unfold HostContract.run at launch
      change (RealTrials.flatOne (call p) (flat p a) chosen ((flat p a).flattenState after)).map
        (fun t => (HostContract.returned (call p),t))=some (HostContract.returned (call p),final) at launch
      cases hs : RealTrials.flatOne (call p) (flat p a) chosen ((flat p a).flattenState after)
      · simp [hs] at launch
      · simpa [hs] using launch
    refine ⟨final,by simp [executeReal,empty,flatHistory,step],?_,?_⟩
    · intro b s g n hb hs hg hn
      obtain ⟨q,hq,i,hi,address,target⟩ := coordinate_covered p h d chosen b s g n hb hs hg hn
      have jobs : q∈HostGrid.programs (HostContract.grid (call p) (tuning chosen)) := by
        rwa [grid_eq p h positive]
      have active : HostGrid.active (tuning chosen).BM (tuning chosen).BN (HostContract.parameters (call p)) q i := by
        rwa [parameters_eq p h positive]
      have got := values q jobs i active
      rw [expected_frame p _ (realState p d) after (fun r off hr => trialFrame r off (fun hh => hr hh.1))] at got
      rw [parameters_eq p h positive,address,target] at got
      exact got
    · intro r off outside
      rw [frame r off (by
        by_cases hr : r=(flat p a).flat
        · right
          intro q hq i hi equal
          apply outside
          exact ⟨hr,HostGrid.address _ _ _ q i,(layouts chosen).bounds q hq after |>.output i hi,equal⟩
        · exact Or.inl hr)]
      apply FlatAlloc.flattenState_frame (flat p a) (realState p d) after
        (RealTrials.scratch (call p) (flat p a)) trialFrame r off
      by_cases hr : r=(flat p a).flat
      · right
        intro region index written equal
        apply outside
        exact ⟨hr,index,written.2,by simpa only [written.1] using equal⟩
      · exact Or.inl hr

#axiomsClean expected_frame
#axiomsClean real_complete
#axiomsClean expected_link
#axiomsClean coordinate_covered
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.Direct
