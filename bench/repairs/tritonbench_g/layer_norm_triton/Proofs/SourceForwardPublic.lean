import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceForwardStorage
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceForwardReal

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceForwardPublic
open VeriTile.Triton FP.Structural SourceModel
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm.Storage
noncomputable section

theorem values_real (model : Model ℝ) (laws : SourceForwardReal.Laws model) (B : Nat) (eps : ℝ)
    (c : KernelContract.Forward M N) (memory : World (Nat → ℝ)) (hn : 0<N) (hb : N≤B) :
    SourceForwardStorage.graphValues model B eps Memory.realAPI Neg.neg c memory=
      KernelForward.realValues c eps memory := by
  funext k i
  fin_cases k
  · exact SourceForwardReal.inverse_real model laws N B hb eps _
  · exact SourceForwardReal.mean_real model laws N B hb _
  · exact SourceForwardReal.output_real model laws N B hb eps _ _ _ (i.val%N) (Nat.mod_lt _ hn)

theorem real_executes (model : Model ℝ) (laws : SourceForwardReal.Laws model) (B : Nat) (eps : ℝ)
    (c : KernelContract.Forward M N) (memory : World (Nat → ℝ)) (hb : N≤B) :
    ∃ result,SourceForwardStorage.exec model B eps Memory.realAPI Neg.neg c memory=some result ∧
      result=SourceForwardStorage.final model B eps Memory.realAPI Neg.neg c memory ∧
      (KernelContract.launched M N=Bool.true → ∀ k i,
        logical Memory.realAPI Neg.neg result (c.buffers.output k).tensor i.val=KernelForward.realValues c eps memory k i) ∧
      (KernelContract.launched M N=Bool.false → result=memory) ∧
      (∀ key,KernelMemory.Outside c.buffers key → observe Memory.realAPI result key=observe Memory.realAPI memory key) ∧
      (∀ key,caller [c.x.dense.tensor,c.w.dense.tensor,c.bias.dense.tensor] key →
        observe Memory.realAPI result key=observe Memory.realAPI memory key) := by
  obtain ⟨result,he,hm,hv,hz,ho,hf⟩ := SourceForwardStorage.executes model B eps Memory.realAPI Neg.neg c memory hb
  refine ⟨result,he,hm,?_,hz,ho,hf⟩
  intro launch k i
  rw [hv launch,values_real model laws B eps c memory ((KernelContract.launched_iff M N).mp launch).2 hb]

/-- The source scalar epsilon denotes the exact host binary64 payload after
the selected ABI conversion. Positive binary64 values that underflow in this
conversion remain admitted. This local condition has no execution premise. -/
def EpsilonABI (model : Model UInt32) (eps : ℝ) (convert : UInt64 → UInt32) (host : Float) : Prop :=
  SourceForwardGraph.lit model eps=convert host.toBits

theorem source_and_candidate [Inhabited α] (model : Model α) (B : Nat) (eps : ℝ)
    (api : Memory.API α Store Byte) (neg : α → α) (c : KernelContract.Forward M N)
    (memory : World Store) (launch : KernelContract.launched M N=Bool.true) (hb : N≤B) (r : Nat) :
    ∃ t,SourceForwardStorage.runRow model B eps api neg c memory r=some t ∧
      SourceForwardStorage.candidateRunRow model B eps api neg c memory r=some t := by
  have hn := ((KernelContract.launched_iff M N).mp launch).2
  obtain ⟨t,ht⟩ := SourceForwardStorage.row_success model B eps api neg c memory hn hb r
  exact ⟨t,ht,(SourceForwardStorage.candidate_row_eq model B eps api neg c memory hn hb r).trans ht⟩

#axiomsClean real_executes
#axiomsClean source_and_candidate
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceForwardPublic
