import bench.repairs.tritonbench_g.token_softmax_bloom.Misc.KernelRealResults
import bench.repairs.tritonbench_g.token_softmax_bloom.Misc.KernelNativeResults
import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.KernelReal
import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.KernelNative

/-! Optional input-check correspondence and Tensor binding helpers. The
complete accepted kernel specifications remain in the primary entries. -/
namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
open KernelTensor KernelGrid
noncomputable section
set_option maxHeartbeats 5000000

theorem python_checks_iff_preconditions (c : Call) (starts lengths : Nat → Int) :
    checkInput c starts lengths=.ok () ↔ InputContract c starts lengths :=
  KernelAdmission.check_input_iff c starts lengths

abbrev referenceReal := KernelReal.reference

def executeReal (c : TensorCall) (read : Nat → ℝ) (metadataRead : Nat → Word)
    (work : List Grid.PID) (before : NativeCells KernelReal.Scalar) : Option (NativeCells KernelReal.Scalar) :=
  if VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked.active c.observation then
    (Grid.run KernelReal.model (params c) work (view c (KernelReal.reads read) metadataRead before)).map
      (project c before)
  else some before

/-- All segments and heads, actual successful full-grid source execution,
original logical Tensor values, exact native addresses and complete cell frame.
The mathematical projection accepts arbitrary Real inputs; FP32 words are
handled independently in FPEquiv without a finite-word restriction. -/
theorem token_softmax_kernel_correct (c : TensorCall) (read : Nat → ℝ)
    (metadataRead : Nat → Word) (before : NativeCells KernelReal.Scalar)
    (binding : InputBinding c (KernelReal.reads read) before)
    (contract : InputContract c.observation (observedStart c metadataRead) (observedLength c metadataRead))
    (work : List Grid.PID)
    (order : (Grid.fullGrid (rows c.start.observation) (rows c.input.observation)).Perm work) :
    ∃ after, executeReal c read metadataRead work before=some after ∧
      (VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked.active c.observation → ∃ raw,
        Grid.run KernelReal.model (params c) work (view c (KernelReal.reads read) metadataRead before)=some raw ∧
        after=project c before raw ∧ RawFrame c (KernelReal.reads read) metadataRead before work raw) ∧
      (contentActive c.observation → ∀ b < rows c.start.observation, ∀ head < rows c.input.observation,
        ∀ n < length c metadataRead b,
        WithBot.unbotD 0 ((after (c.output.tensor.storage.address (c.output.tensor.physicalIndex
          (head*cols c.output.observation+start c metadataRead b+n)))).read .fp32)=
            referenceReal c metadataRead read b head n) ∧
      (∀ address,Exterior c metadataRead address → after address=before address) ∧
      (∀ t∈[c.input.observation,c.start.observation,c.length.observation], ∀ i < t.storage.elements,
        after (t.storage.address i)=before (t.storage.address i)) := by
  by_cases launch : VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked.active c.observation
  · obtain ⟨raw,run,_,values,frame⟩ := KernelGrid.complete KernelReal.model c
      (KernelReal.reads read) metadataRead before contract launch work order
    have sourceRun : Grid.run KernelReal.model (params c) work
        (view c (KernelReal.reads read) metadataRead before)=some raw := by simpa using run
    refine ⟨project c before raw,by simp [executeReal,launch,sourceRun],
      fun _ => ⟨raw,sourceRun,rfl,frame⟩,?_,?_,?_⟩
    · intro content b hb head hh n hn
      have cover := KernelAdmission.cover c metadataRead contract content b hb
      have positive : 0 < (params c).width := by omega
      let i : TileIndex [(params c).width] := (⟨n,lt_of_lt_of_le hn cover⟩,PUnit.unit)
      have live := KernelGrid.live c (KernelReal.reads read) metadataRead before work order b head i hb hh hn
      have value := values ((b,head),i) live
      rw [←KernelGeometry.output_physical c metadataRead contract content b head n hb hh hn]
      unfold outputAddress
      rw [project_read]
      calc
        WithBot.unbotD 0 ((raw "output" (head*c.output.rowStride+
            (start c metadataRead b+n)*c.output.colStride)).read .fp32)=
            WithBot.unbotD 0 (outputValues KernelReal.model .fp32 (params c)
              (Grid.state (view c (KernelReal.reads read) metadataRead before) (b,head)) i) := by
          simpa only [KernelReal.model_profile,Grid.values,offset_view,i] using
            congrArg (fun cell => WithBot.unbotD 0 (cell.read .fp32)) value
        _=VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Real.target (params c)
            (Grid.state (view c (KernelReal.reads read) metadataRead before) (b,head)) n :=
          VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Real.output_target (params c) _ positive
            (KernelReal.numeric c metadataRead read before b head) cover i.1 hn
        _=referenceReal c metadataRead read b head n :=
          KernelReal.target_logical c metadataRead read before binding contract launch b head n hb hh hn
    · exact KernelGrid.project_frame c (KernelReal.reads read) metadataRead before raw work order frame
    · intro t ht i hi
      exact KernelGrid.project_frame c (KernelReal.reads read) metadataRead before raw work order frame _
        (KernelGeometry.read_exterior c metadataRead contract launch t ht i hi)
  · refine ⟨before,by simp [executeReal,launch],fun h => False.elim (launch h),?_,fun _ _ => rfl,fun _ _ _ _ => rfl⟩
    intro content b hb head hh n hn
    exact False.elim (launch (KernelGeometry.coordinate_launch c metadataRead contract content b head n hb hh hn))

abbrev referenceNative := KernelNative.reference

def executeNative (I : KernelNative.Instructions) (V : Nat) (candidate : Bool) (c : TensorCall)
    (work : List Grid.PID) (before : NativeCells Word) : Option (NativeCells Word) :=
  if VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked.active c.observation then
    let initial := view c (KernelNative.nodeRead before) (KernelNative.wordRead before) (KernelNative.lift before)
    let run := if candidate then KernelRewrite.candidateRun (KernelNative.model I V) (params c) work initial
      else Grid.run (KernelNative.model I V) (params c) work initial
    run.map (KernelNative.observe V c before work)
  else some before

/-- Complete typed source/candidate grid success with one common raw state,
actual Tensor words at every output, raw typed and native physical Cell frames,
and preservation of every cell of all read allocations. No scalar numerical law
or whole-row success/result premise is assumed. -/
theorem token_softmax_kernel_fp32 (I : KernelNative.Instructions) (V : Nat)
    (c : TensorCall) (before : NativeCells Word)
    (contract : InputContract c.observation (observedStart c (KernelNative.wordRead before))
      (observedLength c (KernelNative.wordRead before)))
    (work : List Grid.PID)
    (order : (Grid.fullGrid (rows c.start.observation) (rows c.input.observation)).Perm work) :
    ∃ after, executeNative I V Bool.false c work before=some after ∧
      executeNative I V Bool.true c work before=some after ∧
      (VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked.active c.observation → ∃ raw,
        Grid.run (KernelNative.model I V) (params c) work
          (view c (KernelNative.nodeRead before) (KernelNative.wordRead before) (KernelNative.lift before))=some raw ∧
        KernelRewrite.candidateRun (KernelNative.model I V) (params c) work
          (view c (KernelNative.nodeRead before) (KernelNative.wordRead before) (KernelNative.lift before))=some raw ∧
        after=KernelNative.observe V c before work raw ∧
        RawFrame c (KernelNative.nodeRead before) (KernelNative.wordRead before) (KernelNative.lift before) work raw) ∧
      (contentActive c.observation → ∀ b < rows c.start.observation, ∀ head < rows c.input.observation,
        ∀ n < length c (KernelNative.wordRead before) b,
        (after (c.output.tensor.storage.address (c.output.tensor.physicalIndex
          (head*cols c.output.observation+start c (KernelNative.wordRead before) b+n)))).read .fp32=
            referenceNative I V c before b head n) ∧
      (∀ address,Exterior c (KernelNative.wordRead before) address → after address=before address) ∧
      (∀ t∈[c.input.observation,c.start.observation,c.length.observation], ∀ i < t.storage.elements,
        after (t.storage.address i)=before (t.storage.address i)) := by
  by_cases launch : VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked.active c.observation
  · obtain ⟨raw,run,candidateRun,values,frame⟩ := KernelGrid.complete (KernelNative.model I V) c
      (KernelNative.nodeRead before) (KernelNative.wordRead before) (KernelNative.lift before)
      contract launch work order
    have sourceRun : Grid.run (KernelNative.model I V) (params c) work
        (view c (KernelNative.nodeRead before) (KernelNative.wordRead before) (KernelNative.lift before))=some raw := by
      simpa using run
    have candidateRun' : KernelRewrite.candidateRun (KernelNative.model I V) (params c) work
        (view c (KernelNative.nodeRead before) (KernelNative.wordRead before) (KernelNative.lift before))=some raw := by
      simpa using candidateRun
    refine ⟨KernelNative.observe V c before work raw,by simp [executeNative,launch,sourceRun],
      by simp [executeNative,launch,candidateRun'],fun _ => ⟨raw,sourceRun,candidateRun',rfl,frame⟩,?_,?_,?_⟩
    · intro content b hb head hh n hn
      have cover := KernelAdmission.cover c (KernelNative.wordRead before) contract content b hb
      let i : TileIndex [(params c).width] := (⟨n,lt_of_lt_of_le hn cover⟩,PUnit.unit)
      let e : Grid.Event (params c) := ((b,head),i)
      have nodeLive := KernelGrid.live c (KernelNative.nodeRead before) (KernelNative.wordRead before)
        (KernelNative.lift before) work order b head i hb hh hn
      have wordLive := KernelGrid.live c (KernelNative.wordRead before) (KernelNative.wordRead before)
        before work order b head i hb hh hn
      have value := values e nodeLive
      have unique := KernelAdmission.unique c (KernelNative.wordRead before) (KernelNative.wordRead before)
        before contract launch work order
      have observed := KernelNative.observe_readback V c before work raw unique e wordLive
      rw [←KernelGeometry.output_physical c (KernelNative.wordRead before) contract content b head n hb hh hn]
      have readback : (KernelNative.observe V c before work raw
          (outputAddress c (KernelNative.wordRead before) b head n)).read .fp32=
          ((raw "output" (head*c.output.rowStride+(start c (KernelNative.wordRead before) b+n)*c.output.colStride)).read .fp32)
            (KernelNative.owner V (params c).width n) := by
        simpa only [offset_view,e,i,outputAddress] using congrArg (Cell.read .fp32) observed
      rw [readback]
      calc
        ((raw "output" (head*c.output.rowStride+(start c (KernelNative.wordRead before) b+n)*c.output.colStride)).read .fp32)
            (KernelNative.owner V (params c).width n)=
          outputValues (KernelNative.model I V) .fp32 (params c)
            (Grid.state (view c (KernelNative.nodeRead before) (KernelNative.wordRead before) (KernelNative.lift before)) (b,head)) i
            (KernelNative.owner V (params c).width n) := by
          simpa only [KernelNative.model_profile,Grid.values,offset_view,e,i] using
            congrArg (fun cell => (cell.read .fp32) (KernelNative.owner V (params c).width n)) value
        _=Native.target I V (params c) (view c (KernelNative.wordRead before) (KernelNative.wordRead before) before) (b,head) n := by
          rw [KernelNative.view_entry]
          exact Native.value_graph I V (params c) _ (b,head) i.1
        _=referenceNative I V c before b head n := KernelNative.target_logical I V c before contract launch b head n hb hh
    · exact KernelNative.observe_frame V c before work raw order
    · intro t ht i hi
      exact KernelNative.observe_frame V c before work raw order _
        (KernelGeometry.read_exterior c (KernelNative.wordRead before) contract launch t ht i hi)
  · refine ⟨before,by simp [executeNative,launch],by simp [executeNative,launch],
      fun h => False.elim (launch h),?_,fun _ _ => rfl,fun _ _ _ _ => rfl⟩
    intro content b hb head hh n hn
    exact False.elim (launch (KernelGeometry.coordinate_launch c (KernelNative.wordRead before)
      contract content b head n hb hh hn))

#axiomsClean token_softmax_kernel_correct
#axiomsClean token_softmax_kernel_fp32
#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.TensorBridge
