import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.RealFiniteInputs
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.RealFiniteGeometry
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.RealGridResult

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealFiniteRows
open VeriTile Triton Source RealFiniteContext RealFiniteGeometry
open VeriTile.Bench.Repairs.AttentionLlama LoopInvariant HostGrid HostResult MaskCoverage Correctness
open VeriTile.Bench.TritonBenchG.AttentionLlama
open VeriTile.Bench.TritonBenchG.AttentionLlama.RealKernel
open Backing
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem target_binding (B : Nat) (c : Config) (scale : ℝ)
    (vals : (RealLayout.memory B c).Ctx) (s : BlockState)
    (hin : (RealLayout.memory B c).InputBinding s vals)
    (sep : c.fp8=Bool.true → c.Q≠c.K ∧ c.Q≠c.V)
    (j : RealLayout.QueryIndex B c) :
    RealKernel.target (realView (contextFor (context c scale s) (index j)) c.fp8)
      (HostResult.lane (c:=context c scale s) (index j),(index j).channel,())=
      RealMathematical.attention B c scale vals 0 (RealLayout.querySlot B c j) := by
  have h:=HostResult.target_binding (realView (context c scale s) c.fp8)
    (RealMathematical.inputs B c vals) (RealFiniteInputs.binding B c scale vals s hin sep) (index j)
  have commute (fp8 : Bool) :
      realView (contextFor (context c scale s) (index j)) fp8=
        contextFor (realView (context c scale s) fp8) (index j) := by
    cases fp8 <;> rfl
  let left:=(realView (contextFor (context c scale s) (index j)) c.fp8).initial
  let right:=(contextFor (realView (context c scale s) c.fp8) (index j)).initial
  have hi : left=right := congrArg LoopInvariant.Context.initial (commute c.fp8)
  change RealKernel.target {context c scale s with initial:=left}
    (HostResult.lane (c:=context c scale s) (index j),(index j).channel,())=_
  rw [hi]
  change RealKernel.target (contextFor (realView (context c scale s) c.fp8) (index j))
    (HostResult.lane (c:=context c scale s) (index j),(index j).channel,())=_
  simpa only [HostResult.expected,realView,context,RealMathematical.attention,
    RealLayout.queryIndex_slot,index] using h

/-- Existing complete mathematical source executions, connected to finite
inputs and the original runtime scalar. No source execution is reproved. -/
theorem rows (B : Nat) (c : Config) (scale : ℝ)
    (hD : 0<c.D) (hN : 0<c.N) (hSP : c.causal=Bool.true → c.SP=0)
    (sep : c.fp8=Bool.true → c.Q≠c.K ∧ c.Q≠c.V ∧ c.Out≠c.K ∧ c.Out≠c.V)
    (hinj : Function.Injective (IOLayout.output (B:=B) c))
    (vals : (RealLayout.memory B c).Ctx) (s : BlockState)
    (hin : (RealLayout.memory B c).InputBinding s vals)
    (domain : RealLayout.domain B c vals)
    (bound : RealScalarInvariant.Bound scale s) (hu : ∀ r o,s.undef r o=0) :
    ∃ states : Program B c.NH c.H→BlockState,
      (∀ p,VeriTile.Triton.exec (Source.kernel c).eraseDType.toAlgKernel
        (contextFor (context c scale s) (anchor hD p)).initial=some (states p)) ∧
      (∀ i : RealLayout.QueryIndex B c,
        (states (owner (index i))).readMem c.Out (IOLayout.output c i)=
          RealMathematical.attention B c scale vals 0 (RealLayout.querySlot B c i)) ∧
      (∀ p r o,¬writes B c p (r,o) → (states p).mem r o=s.mem r o) := by
  let cc:=context c scale s
  let layout:=outputLayout B c scale s hinj
  obtain ⟨states,he,hv,hf⟩:=RealGridResult.attention_kernel_grid B cc c.Out layout c.fp8
    hD hN hSP hu (fun hb=>RealFiniteInputs.byte_domain B c scale vals s hin domain hb) sep
  refine ⟨states,?_,?_,?_⟩
  · intro p
    have eq:=RealSourceTransport.backing_execution_eq cc c.Out c.soz c.soh c.som c.son c.fp8
      (contextFor cc (anchor hD p)).initial bound
    simpa only [cc,config_context,layout,outputLayout] using eq.trans (he p)
  · intro j
    let i:=index j
    have active : alActive (contextFor cc i).initial c.H 64
        (HostResult.lane (c:=cc) i,i.channel,()) := by
      change i.query.val/64*64+i.query.val%64<c.H
      have h:=i.query.isLt
      omega
    have vi:=hv (owner i) (HostResult.lane (c:=cc) i,i.channel,())
    let sa:=(contextFor cc (anchor hD (owner i))).initial
    let si:=(contextFor cc i).initial
    have hs : sa=si := congrArg LoopInvariant.Context.initial (context_owner cc hD i)
    change alActive sa c.H 64 (HostResult.lane (c:=cc) i,i.channel,()) →
      (states (owner i)).readMem c.Out
        (alOutOffset sa c.NH c.soz c.soh c.som c.son 64 (HostResult.lane (c:=cc) i,i.channel,()))=
      RealKernel.target (realView {cc with initial:=sa} c.fp8)
        (HostResult.lane (c:=cc) i,i.channel,()) at vi
    rw [hs] at vi
    have address:=real_address cc c.Out c.soz c.soh c.som c.son c.fp8 i
    have address' : alOutOffset (contextFor cc i).initial c.NH c.soz c.soh c.som c.son 64
        (HostResult.lane (c:=cc) i,i.channel,())=IOLayout.output c j := by
      simpa only [cc,config_context,Tensor.offset,IOLayout.output,IOLayout.offset,i,index] using address
    change (states (owner i)).readMem c.Out (IOLayout.output c j)=_
    rw [←address']
    exact (vi active).trans (target_binding B c scale vals s hin (fun hb=>⟨(sep hb).1,(sep hb).2.1⟩) j)
  · intro p r o miss
    apply hf p r o
    by_cases hr:r=c.Out
    · right
      intro j active heq
      apply miss
      rw [hr,heq]
      exact active_writes B c scale s hD p j active
    · exact Or.inl hr

#axiomsClean rows
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealFiniteRows
