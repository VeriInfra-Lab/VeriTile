import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.RealRevision.Frames
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.RealRevision.RealViews
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.RealRevision.ActiveGrid

namespace VeriTile.Bench.Repairs.LayerNormFwd.RealRevision
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper WrapperNative
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000
variable {α Host : Type} {caller : Address → Prop}

theorem prepared_observation
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs) (memory : RealMemory) (template : BlockState)
    (hfall : ∀ (i : Fin p.validated.M) (j : Fin p.validated.N),
      logicalRead store current memory p.inputs.originalX [i.val,j.val]=
        AliasMemory.read (PhysicalDomain.layout (preparedAllocations store current p)) memory
          (preparedRegions.x,i.val*(argumentStrides args).x+j.val)) :
    preparedObservation (readStore store memory) current p (HostReturn.seedCount (p.flags q) p.validated.M)=
      HostReturn.observe (p.flags q) preparedRegions (argumentStrides args) p.validated.M p.validated.N
        (AliasMemory.pull (PhysicalDomain.layout (preparedAllocations store current p)) memory template) := by
  have hread := role_read store hm tensor q initial current p args h hv hi ho memory
  have hy1 := optionAllocated_presence h.outputs.2.1
  have hmean := optionAllocated_presence h.outputs.2.2.2.1
  have hseeds := optionAllocated_presence h.outputs.2.2.2.2.2.1
  have hmask := optionAllocated_presence h.outputs.2.2.2.2.2.2
  have hdrop : p.outputs.mask.isSome=Bool.true → q.dropout=Bool.true := by
    intro he
    rw [he] at hmask
    cases hd : q.dropout
    · have hf : Bool.true=Bool.false := by simpa only [hd,Bool.and_false] using hmask
      cases hf
    · rfl
  apply HostReturn.Result.ext
  · funext i j
    exact hread i.val j.val i.isLt .y trivial j.isLt p.outputs.y rfl
  · cases he : p.outputs.y1 with
    | none =>
      have hf : (p.flags q).w1=Bool.false := by simpa [he,Prepared.flags] using hy1.symm
      simp [preparedObservation,HostReturn.observe,he,hf]
    | some value =>
      have hf : (p.flags q).w1=Bool.true := by simpa [he,Prepared.flags] using hy1.symm
      simp only [preparedObservation,HostReturn.observe,he,hf,ite_true,Option.map_some]
      apply congrArg some
      funext i j
      exact hread i.val j.val i.isLt .y1 hf j.isLt value he
  · cases he : p.outputs.mean with
    | none =>
      have hf : (p.flags q).rms=Bool.true := by
        simpa [he,Prepared.flags] using congrArg Bool.not hmean.symm
      simp [preparedObservation,HostReturn.observe,he,hf]
    | some value =>
      have hf : (p.flags q).rms=Bool.false := by
        simpa [he,Prepared.flags] using congrArg Bool.not hmean.symm
      simp only [preparedObservation,HostReturn.observe,he,hf,Bool.false_eq_true,ite_false,Option.map_some]
      apply congrArg some
      funext i
      exact hread i.val 0 i.isLt .mean hf trivial value he
  · funext i
    exact hread i.val 0 i.isLt .rstd trivial trivial p.outputs.rstd rfl
  · cases he : p.outputs.residualOut with
    | none =>
      have hf : (p.flags q).residualOut=Bool.false := by simp [Prepared.flags,he]
      simp only [preparedObservation,HostReturn.observe,he,hf,Bool.false_eq_true,ite_false,Option.getD_none]
      funext i j
      exact hfall i j
    | some value =>
      have hf : (p.flags q).residualOut=Bool.true := by simp [Prepared.flags,he]
      simp only [preparedObservation,HostReturn.observe,he,hf,ite_true,Option.getD_some]
      funext i j
      exact hread i.val j.val i.isLt .residualOut hf j.isLt value he
  · cases he : p.outputs.seeds with
    | none =>
      have hf : (p.flags q).dropout=Bool.false := by simpa [he,Prepared.flags] using hseeds.symm
      simp [preparedObservation,HostReturn.observe,he,hf]
    | some value =>
      have hf : (p.flags q).dropout=Bool.true := by simpa [he,Prepared.flags] using hseeds.symm
      simp only [preparedObservation,HostReturn.observe,he,hf,ite_true,Option.map_some]
      apply congrArg some
      funext i
      by_cases hir : i.val<p.validated.M
      · exact hread i.val 0 hir .seeds hf trivial value he
      · have hx : (p.flags q).x1=Bool.true := by
          cases hx : (p.flags q).x1
          · have hb := i.isLt
            simp [HostReturn.seedCount,hx] at hb
            exact False.elim (hir hb)
          · rfl
        have hb : i.val<2*p.validated.M := by simpa [HostReturn.seedCount,hx] using i.isLt
        have hrow : i.val-p.validated.M<p.validated.M := by omega
        have heq : p.validated.M+(i.val-p.validated.M)=i.val := by omega
        have hr := hread (i.val-p.validated.M) 0 hrow .seeds1 ⟨hx,hf⟩ trivial value he
        simpa only [roleIndex,LayerNormFwd.Address.region,LayerNormFwd.Address.offset,heq,readVector] using hr
  · cases he : p.outputs.mask with
    | none =>
      have hf : (p.flags q).dropoutMask=Bool.false := by simp [Prepared.flags,he]
      simp [preparedObservation,HostReturn.observe,he,hf]
    | some value =>
      have hm : (p.flags q).dropoutMask=Bool.true := by simp [Prepared.flags,he]
      have hd : (p.flags q).dropout=Bool.true := hdrop (by simp [he])
      simp only [preparedObservation,HostReturn.observe,he,hm,hd,Bool.true_and,ite_true,Option.map_some]
      apply congrArg some
      funext i j
      exact hread i.val j.val i.isLt .mask ⟨hd,hm⟩ j.isLt value he
  · cases hx : p.inputs.x1.isSome with
    | «false» =>
      have hf : (p.flags q).x1=Bool.false := hx
      simp [preparedObservation,HostReturn.observe,hx,hf]
    | «true» =>
      cases he : p.outputs.mask with
      | none =>
        have hf : (p.flags q).dropoutMask=Bool.false := by simp [Prepared.flags,he]
        simp [preparedObservation,HostReturn.observe,hx,he,hf]
      | some value =>
        have hm : (p.flags q).dropoutMask=Bool.true := by simp [Prepared.flags,he]
        have hd : (p.flags q).dropout=Bool.true := hdrop (by simp [he])
        have hf : (p.flags q).x1=Bool.true := hx
        simp only [preparedObservation,HostReturn.observe,hx,he,hm,hd,hf,Bool.true_and,ite_true,Option.map_some]
        apply congrArg some
        funext i j
        exact hread i.val j.val i.isLt .mask1 ⟨hf,hd,hm⟩ j.isLt value he


theorem width_arange {S Key Value : Type} {w : World S Key Value}
    (api : TensorAPI w) (v : Validated) (input : Wrapper.Inputs) (s : S) (block : Nat)
    (h : ((width api v input).run s).result=.ok block) (htype : input.x.dtype=DType.fp32)
    (hpos : 0<block) : Random.ArangeSizeDomain block := by
  have hlimit := width_fp32 api v input s block h htype
  unfold width at h
  dsimp only [Bind.bind] at h
  obtain ⟨bytes,hbytes,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨next,hnext,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨_,_,h,_⟩ := bind_success _ _ _ _ h
  have hb : bytes=input.x.dtype.bytes := by simpa [checked] using hbytes.symm
  have hblock : block=min (65536/bytes) next := by simpa [Pure.pure,Action.pure] using h.symm
  rw [hb,htype] at hblock
  norm_num [DType.fp32] at hblock
  have hn := api.nextPower_value v.N _ next (call_success _ _ _ _ hnext)
  refine ⟨hpos,by omega,?_⟩
  by_cases hmin : 16384≤next
  · refine ⟨14,?_⟩
    rw [hblock,Nat.min_eq_left hmin]
    norm_num
  · have he : block=next := hblock.trans (Nat.min_eq_right (by omega))
    rw [he,hn]
    split_ifs with hzero hone
    · rw [hn,if_pos hzero] at he
      omega
    · exact ⟨0,rfl⟩
    · exact ⟨Nat.log2 (v.N-1)+1,rfl⟩

theorem prepare_arange {S Key Value : Type} {w : World S Key Value}
    (api : TensorAPI w) (q : Request) (s : S) (p : Prepared)
    (h : ((prepare api q).run s).result=.ok p) (htype : p.inputs.x.dtype=DType.fp32)
    (hpos : 0<p.block) : Random.ArangeSizeDomain p.block := by
  obtain ⟨_,_,v,input,output,block,_,_,_,hw,he⟩ := prepare_success api q s p h
  rw [he] at htype hpos ⊢
  exact width_arange api v input _ block hw htype hpos

theorem prior_real_read
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs)
    (c : StorageMachine.Config α block) (hc : WrapperStorage.Placement store q p args current c)
    (before memory : RealMemory)
    (frame : ∀ i : Address, (∀ k, AliasMemory.Writes c.flags preparedRegions c.strides c.M c.N k →
      (PhysicalDomain.layout c.allocations).address k≠i.val) → memory i=before i)
    (old : Tensor) (hp : store.present initial old) (index : List Nat) (hindex : Coordinate old index) :
    logicalRead store current memory old index=logicalRead store current before old index := by
  have hd := (h.preserved old hp).2.1
  have hb := hm.bounds initial old hp index hindex
  have hbytes := (LayerNormFwd.Address.byte_address (store.descriptor initial old) hb).2.2.1
  let i : Address := Fin.ofNat (2^64) (tensorAddress store current old index)
  have haddress : i.val=(store.descriptor initial old).base+
      coordinateOffset old.strides index*(store.descriptor initial old).width := by
    change tensorAddress store current old index % 2^64=_
    unfold tensorAddress
    rw [hd,Nat.mod_eq_of_lt hbytes]
    rfl
  have hbyte := Nat.mul_lt_mul_of_pos_right hb (store.descriptor initial old).width_pos
  have hf := frame i (WrapperStorage.prior_outside store hm tensor q initial current p args h hv hi ho
    c hc old hp _ hbyte i haddress)
  simp only [logicalRead,if_pos hindex,rawRead]
  change (if old.negative then negateCell (memory i) else memory i)=_
  rw [hf]

theorem prepared_observation_extends
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (memory : RealMemory)
    (s t : HostState α Host) (he : (readStore store memory).Extends s t)
    (p : Prepared) (hp : PreparedPresent store p s) (K : Nat) :
    preparedObservation (readStore store memory) t p K=preparedObservation (readStore store memory) s p K := by
  let rs := readStore store memory
  apply HostReturn.Result.ext
  · exact matrix_extends rs s t he _ _ _ hp.y
  · exact optional_extends rs s t p.outputs.y1 (option_read_present store memory s _ hp.y1)
      (fun s t => readMatrix rs s p.validated.M p.validated.N t) (fun t hp => matrix_extends rs s _ he _ _ t hp)
  · exact optional_extends rs s t p.outputs.mean (option_read_present store memory s _ hp.mean)
      (fun s t => readVector rs s p.validated.M t) (fun t hp => vector_extends rs s _ he _ t hp)
  · exact vector_extends rs s t he _ _ hp.rstd
  · exact matrix_extends rs s t he _ _ _ hp.fallback
  · exact optional_extends rs s t p.outputs.seeds (option_read_present store memory s _ hp.seeds)
      (fun s t => readVector rs s K t) (fun t hp => vector_extends rs s _ he _ t hp)
  · exact optional_extends rs s t p.outputs.mask (option_read_present store memory s _ hp.mask)
      (fun s t => readMatrix rs s p.validated.M p.validated.N t) (fun t hp => matrix_extends rs s _ he _ _ t hp)
  · dsimp only [preparedObservation]
    split
    · exact optional_extends rs s t p.outputs.mask (option_read_present store memory s _ hp.mask)
        (fun s t => readSecondMatrix rs s p.validated.M p.validated.N t)
        (fun value hp => funext fun i => funext fun j => (he value hp).2.2 _)
    · rfl

namespace RealHistory

def eraseEvents {profile : StorageCompiler.Profile} {f : Flags} {N block : Nat}
    (events : List (StorageMachine.DispatchEvent profile f N block)) : List Bool :=
  events.map fun event => match event with | .launch _ => Bool.true | .setupFailure _ => Bool.false

structure Step (f : Flags) (stride : Strides) (M N : Nat) (eps p : ℝ) (block : Nat)
    (hBlock : Random.ArangeSizeDomain block) (a : AliasMemory.Layout) (template : BlockState)
    (before after : AliasMemory.Physical a) : Prop where
  witness : ∃ final,
    Nonempty (Kernel.GridLaunchedOrdinary
      ((repaired f preparedRegions stride M N eps p block hBlock).eraseDType.toAlgKernel)
      (WholeGrid.grid M) (AliasMemory.pull a before template) final) ∧
    after=AliasMemory.runPhysical a before (AliasGrid.events f preparedRegions stride M N final) ∧
    WholeGrid.Post f preparedRegions stride M N eps p (AliasMemory.pull a before template) final

inductive Replay (f : Flags) (stride : Strides) (M N : Nat) (eps p : ℝ) (block : Nat)
    (hBlock : Random.ArangeSizeDomain block) (a : AliasMemory.Layout) (template : BlockState) :
    List Bool → AliasMemory.Physical a → AliasMemory.Physical a → Prop
  | nil (memory) : Replay f stride M N eps p block hBlock a template [] memory memory
  | launch {events before middle after}
      (step : Step f stride M N eps p block hBlock a template before middle)
      (rest : Replay f stride M N eps p block hBlock a template events middle after) :
      Replay f stride M N eps p block hBlock a template (Bool.true::events) before after

def InputEqual (a : AliasMemory.Layout) (before after : AliasMemory.Physical a) : Prop :=
  ∀ region, StorageMachine.inputRegion region → ∀ offset,
    AliasMemory.read a after (region,offset)=AliasMemory.read a before (region,offset)

def Frame (f : Flags) (stride : Strides) (M N : Nat) (a : AliasMemory.Layout)
    (before after : AliasMemory.Physical a) : Prop :=
  ∀ i, (∀ k, AliasMemory.Writes f preparedRegions stride M N k → a.address k≠i.val) → after i=before i

theorem input_unwritten (f : Flags) (stride : Strides) (M N row : Nat) (region : RegionName)
    (hr : StorageMachine.inputRegion region) (offset : Nat) :
    ¬Contract.written f preparedRegions stride M N row region offset := by
  intro h
  have hw := HostReturn.written_region f preparedRegions stride M N row region offset h
  simp only [StorageMachine.inputRegion,List.mem_cons,List.not_mem_nil,or_false] at hr
  rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals simp [Domain.writeRegion,preparedRegions] at hw

theorem launch_exists (f : Flags) (stride : Strides) (M N : Nat) (eps p : ℝ)
    (block : Nat) (hBlock : Random.ArangeSizeDomain block) (a : AliasMemory.Layout) (template : BlockState)
    (hN : N≤block) (hg : NativeGeometry.Geometry f preparedRegions stride M N)
    (hf : AliasMemory.FreshWrites a (AliasMemory.Writes f preparedRegions stride M N))
    (before : AliasMemory.Physical a) :
    ∃ after, Step f stride M N eps p block hBlock a template before after ∧ InputEqual a before after ∧
      Frame f stride M N a before after := by
  obtain ⟨final,hrun,hpost⟩ := ActiveGrid.source_correct f preparedRegions stride M N eps p block hBlock
    hN (prepared_read_write_separated f) hg (AliasMemory.pull a before template)
  let after := AliasMemory.runPhysical a before (AliasGrid.events f preparedRegions stride M N final)
  have hpull := AliasGrid.source_trace_pull a before template final f preparedRegions stride M N eps p hf hpost
  refine ⟨after,⟨final,hrun,rfl,hpost⟩,?_,?_⟩
  · intro region hr offset
    change (AliasMemory.pull a after final).mem region offset=_
    rw [hpull]
    exact hpost.2 region offset (fun row _ => input_unwritten f stride M N row region hr offset)
  · intro i hi
    exact AliasMemory.trace_frame a before (AliasGrid.events f preparedRegions stride M N final) i
      (fun e he => Ne.symm (hi e.key (AliasGrid.events_written f preparedRegions stride M N final e he)))

theorem replay_exists [Inhabited α] (c : StorageMachine.Config α block)
    (events : List (StorageMachine.DispatchEvent c.profile c.flags c.N block))
    (native : StorageMachine.Memory c) (success : (StorageMachine.history c events native).error=none)
    (eps p : ℝ) (hBlock : Random.ArangeSizeDomain block) (template : BlockState)
    (before : AliasMemory.Physical (PhysicalDomain.layout c.allocations)) :
    ∃ after, Replay c.flags c.strides c.M c.N eps p block hBlock (PhysicalDomain.layout c.allocations)
      template (eraseEvents events) before after ∧
      InputEqual (PhysicalDomain.layout c.allocations) before after ∧
      Frame c.flags c.strides c.M c.N (PhysicalDomain.layout c.allocations) before after := by
  induction events generalizing native before with
  | nil => exact ⟨before,.nil before,fun _ _ _ => rfl,fun _ _ => rfl⟩
  | cons event events ih =>
    cases event with
    | setupFailure code => simp [StorageMachine.history] at success
    | launch artifact =>
      obtain ⟨middle,step,hinput,hframe⟩ := launch_exists c.flags c.strides c.M c.N eps p block hBlock
        (PhysicalDomain.layout c.allocations) template c.accepted.width c.geometry c.storage.fresh before
      obtain ⟨after,rest,hi,hf⟩ := ih (StorageMachine.execute c artifact native) success middle
      exact ⟨after,.launch step rest,fun r hr o => (hi r hr o).trans (hinput r hr o),
        fun i outside => (hf i outside).trans (hframe i outside)⟩

theorem snapshot_eq (a : AliasMemory.Layout) (before after : AliasMemory.Physical a)
    (h : InputEqual a before after) (stride : Strides) (template : BlockState) :
    snapshot preparedRegions stride (AliasMemory.pull a after template)=
      snapshot preparedRegions stride (AliasMemory.pull a before template) := by
  unfold snapshot
  congr 1
  all_goals first
    | (funext row col; simp only [BlockState.readMem,AliasMemory.pull]; rw [h _ (by simp [StorageMachine.inputRegion]) _])
    | (funext offset; simp only [BlockState.readMem,BlockState.readMemValue,BlockState.readMemTyped,AliasMemory.pull];
        rw [h _ (by simp [StorageMachine.inputRegion]) _])

theorem oracle_eq (a : AliasMemory.Layout) (before after : AliasMemory.Physical a)
    (h : InputEqual a before after) (f : Flags) (stride : Strides) (M N : Nat) (eps p : ℝ) (template : BlockState) :
    HostReturn.oracle f preparedRegions stride M N eps p (AliasMemory.pull a after template)=
      HostReturn.oracle f preparedRegions stride M N eps p (AliasMemory.pull a before template) := by
  simp only [HostReturn.oracle,snapshot_eq a before after h stride template]
  apply HostReturn.Result.ext
  all_goals try rfl
  · cases hf : f.residualOut <;> simp only [hf,Bool.false_eq_true,ite_false,ite_true]
    funext i j
    exact h preparedRegions.x (by simp [StorageMachine.inputRegion]) _
  · cases hf : f.dropout <;> simp only [hf,Bool.false_eq_true,ite_false,ite_true]
    apply congrArg some
    funext i
    exact h preparedRegions.seeds (by simp [StorageMachine.inputRegion]) _

end RealHistory

end
end VeriTile.Bench.Repairs.LayerNormFwd.RealRevision
