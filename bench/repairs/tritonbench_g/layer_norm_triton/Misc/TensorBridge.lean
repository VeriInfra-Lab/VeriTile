import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.KernelSaved
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceForwardPublic
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Compiler
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceSaved
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceBackwardPublic
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceObserved
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceIdealInput
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceBackwardResult
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceNativeComposition
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceNativeBinding
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceNativeCandidate
import bench.repairs.tritonbench_g.layer_norm_triton.Correct
import bench.repairs.tritonbench_g.layer_norm_triton.Misc.NativeCorrect
import bench.repairs.tritonbench_g.layer_norm_triton.Misc.ForwardNativeObservation
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.NativeEndpoints

/-! Complete actual-Tensor source, graph, native and composed applications. -/

/-! Constructive mathematical results for the independent kernel graphs.
Every output and the complete memory/frame are included. Actual typed-source execution is connected to independent Real formulas,
physical outputs and saved-statistic composition. Every public Real epsilon is
projected from the exact host bits stored in its actual forward-call contract.
Historical whole-host results are preserved under Proofs/HostCorrect. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.TensorBridge.Correctness
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm.Storage
open KernelMemory
noncomputable section


/-- Actual typed forward execution, actual Tensor loads and stores, and the
independent Real specification. Launch settings retain the selected bound. -/
theorem layer_norm_forward_source_real (model : SourceModel.Model ℝ)
    (laws : SourceForwardReal.Laws model) (c : KernelContract.Forward M N)
    (B W : Nat) (settings : Compiler.settings N=.ok (B,W)) (projection : SourceIdealInput.ScalarProjection) (memory : World (Nat → ℝ)) (view : SourceStrideRefinement.MatrixView c.x M N) :
    ∃ result,SourceObserved.forward model B (SourceIdealInput.epsilon projection c) Memory.realAPI Neg.neg c memory view=some result ∧
      result=SourceForwardStorage.final model B (SourceIdealInput.epsilon projection c) Memory.realAPI Neg.neg c memory ∧
      (KernelContract.launched M N=Bool.true → ∀ k i,
        logical Memory.realAPI Neg.neg result (c.buffers.output k).tensor i.val=KernelForward.realValues c (SourceIdealInput.epsilon projection c) memory k i) ∧
      (KernelContract.launched M N=Bool.false → result=memory) ∧
      (∀ key,Outside c.buffers key → observe Memory.realAPI result key=observe Memory.realAPI memory key) ∧
      (∀ key,caller [c.x.dense.tensor,c.w.dense.tensor,c.bias.dense.tensor] key →
        observe Memory.realAPI result key=observe Memory.realAPI memory key) := by
  have hb := (Compiler.settings_success N B W settings).2.1
  simpa only [SourceObserved.forward_eq model B (SourceIdealInput.epsilon projection c) Memory.realAPI Neg.neg c memory view hb] using
    SourceForwardPublic.real_executes model laws B (SourceIdealInput.epsilon projection c) c memory hb

theorem layer_norm_forward_graph_real (p : KernelGraph.ForwardPlan N B)
    (c : KernelContract.Forward M N) (projection : SourceIdealInput.ScalarProjection) (m : World (Nat → ℝ)) :
    ∃ result,KernelForward.exec RealProof.ops p Memory.realAPI Neg.neg c (SourceIdealInput.epsilon projection c) m=.ok result ∧
      result=KernelForward.final RealProof.ops p Memory.realAPI Neg.neg c (SourceIdealInput.epsilon projection c) m ∧
      (KernelContract.launched M N=Bool.true → ∀ k i,
        logical Memory.realAPI Neg.neg result (c.buffers.output k).tensor i.val=
          KernelForward.realValues c (SourceIdealInput.epsilon projection c) m k i) ∧
      (KernelContract.launched M N=Bool.false → result=m) ∧
      (∀ key,Outside c.buffers key → observe Memory.realAPI result key=observe Memory.realAPI m key) ∧
      (∀ key,caller [c.x.dense.tensor,c.w.dense.tensor,c.bias.dense.tensor] key →
        observe Memory.realAPI result key=observe Memory.realAPI m key) := by
  simpa only [KernelForward.values_real] using
    KernelForward.executes RealProof.ops p Memory.realAPI Neg.neg c (SourceIdealInput.epsilon projection c) m

theorem layer_norm_backward_graph_real (p : KernelGraph.BackwardPlan N B)
    (x w dy r mu : Input) (c : KernelContract.Backward M N x w dy r mu) (m : World (Nat → ℝ)) :
    ∃ result,KernelBackward.exec RealProof.ops p Memory.realAPI c m=.ok result ∧
      result=KernelBackward.final RealProof.ops p Memory.realAPI c m ∧
      (∀ k i,logical Memory.realAPI Neg.neg result (KernelBackward.outputs c k).tensor i.val=
        if KernelContract.launched M N then KernelBackward.realValues c m k i
        else logical Memory.realAPI Neg.neg m (KernelBackward.outputs c k).tensor i.val) ∧
      (KernelContract.launched M N=Bool.false → result=m) ∧
      (KernelContract.launched M N=Bool.true → ∀ k i,
        logical Memory.realAPI Neg.neg result (c.first.buffers.output k).tensor i.val=
          Execution.groupValues RealProof.ops p.base Memory.realAPI c.first
            x.dense w.dense dy.dense r.dense mu.dense m k i) ∧
      (∀ key,KernelBackward.Outside c key → observe Memory.realAPI result key=observe Memory.realAPI m key) ∧
      (∀ key,caller [x.dense.tensor,w.dense.tensor,dy.dense.tensor,r.dense.tensor,mu.dense.tensor] key →
        observe Memory.realAPI result key=observe Memory.realAPI m key) := by
  simpa only [KernelBackward.nativeValues_real] using
    KernelBackward.executes RealProof.ops p Memory.realAPI Neg.neg c m


theorem layer_norm_backward_source_real (model : SourceModel.Model ℝ) (laws : SourceFinalReal.Laws model) (B : Nat)
    (x w dy r mu : Input) (c : KernelContract.Backward M N x w dy r mu) (memory : World (Nat → ℝ)) (view : SourceStrideRefinement.MatrixView x M N) (hb : N≤B) :
    ∃ result,SourceObserved.backward model B Memory.realAPI Neg.neg c memory view=some result ∧
      result=SourceBackwardStorage.final model B Memory.realAPI Neg.neg c memory ∧
      (∀ k i,logical Memory.realAPI Neg.neg result (KernelBackward.outputs c k).tensor i.val=
        if KernelContract.launched M N then KernelBackward.realValues c memory k i
        else logical Memory.realAPI Neg.neg memory (KernelBackward.outputs c k).tensor i.val) ∧
      (KernelContract.launched M N=Bool.false → result=memory) ∧
      (KernelContract.launched M N=Bool.true → ∀ k i,
        logical Memory.realAPI Neg.neg result (c.first.buffers.output k).tensor i.val=
          SourceBackwardGroupReal.values c memory k i) ∧
      (∀ key,KernelBackward.Outside c key → observe Memory.realAPI result key=observe Memory.realAPI memory key) ∧
      (∀ key,caller [x.dense.tensor,w.dense.tensor,dy.dense.tensor,r.dense.tensor,mu.dense.tensor] key →
        observe Memory.realAPI result key=observe Memory.realAPI memory key) := by
  simpa only [SourceObserved.backward_eq] using SourceBackwardPublic.real_executes model laws B c memory hb

#axiomsClean layer_norm_backward_source_real
theorem layer_norm_composed_source_real (model : SourceModel.Model ℝ) (laws : SourceFinalReal.Laws model) (B : Nat) (projection : SourceIdealInput.ScalarProjection)
    (dy r mu : Input) (f : KernelContract.Forward M N) (b : KernelContract.Backward M N f.x f.w dy r mu)
    (rBinding : r.dense=f.buffers.output 0) (muBinding : mu.dense=f.buffers.output 1)
    (memory : World (Nat → ℝ)) (view : SourceStrideRefinement.MatrixView f.x M N) (hb : N≤B) :
    ∃ forward backward,
      SourceObserved.forward model B (SourceIdealInput.epsilon projection f) Memory.realAPI Neg.neg f memory view=some forward ∧
      SourceObserved.backward model B Memory.realAPI Neg.neg b forward view=some backward ∧
      forward=SourceForwardStorage.final model B (SourceIdealInput.epsilon projection f) Memory.realAPI Neg.neg f memory ∧
      backward=SourceBackwardStorage.final model B Memory.realAPI Neg.neg b forward ∧
      (KernelContract.launched M N=Bool.true → ∀ k i,
        logical Memory.realAPI Neg.neg forward (f.buffers.output k).tensor i.val=KernelForward.realValues f (SourceIdealInput.epsilon projection f) memory k i) ∧
      (∀ k i,logical Memory.realAPI Neg.neg backward (KernelBackward.outputs b k).tensor i.val=
        if KernelContract.launched M N then KernelBackward.realValues b forward k i
        else logical Memory.realAPI Neg.neg forward (KernelBackward.outputs b k).tensor i.val) ∧
      (KernelContract.launched M N=Bool.true → ∀ row,row<M →
        KernelBackward.initialStats Memory.realAPI Neg.neg r mu forward row=
          ⟨RealProof.Spec.mean N (KernelForward.row Memory.realAPI Neg.neg f.x memory N row),
           RealProof.Spec.inverse N (KernelForward.row Memory.realAPI Neg.neg f.x memory N row) (SourceIdealInput.epsilon projection f)⟩) ∧
      (KernelContract.launched M N=Bool.false → forward=memory ∧ backward=memory) ∧
      (∀ key,Outside f.buffers key → KernelBackward.Outside b key →
        observe Memory.realAPI backward key=observe Memory.realAPI memory key) := by
  simpa only [SourceObserved.forward_eq model B (SourceIdealInput.epsilon projection f) Memory.realAPI Neg.neg f memory view hb,
    SourceObserved.backward_eq] using SourceBackwardPublic.composed model laws B (SourceIdealInput.epsilon projection f) f b rBinding muBinding memory hb

#axiomsClean layer_norm_composed_source_real
#axiomsClean layer_norm_forward_graph_real
#axiomsClean layer_norm_forward_source_real
#axiomsClean layer_norm_backward_graph_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.TensorBridge.Correctness

/-! Actual typed-source FP32/native equivalence. Native execution first builds
source expression trees and then applies selected structural FMA, reduction and
operand-order policies. Physical kernel boundaries materialize words before
subsequent loads. The earlier strict UInt32 and graph-only entries remain
explicitly named. Historical whole-host results are under Proofs/HostFPEquiv. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.TensorBridge.Equivalence
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm.Storage
open KernelMemory SourceNative.Native SourceNativeStorage SourceStrideRefinement
noncomputable section

/-- Both actual backward source kernels execute, and the second reads the
scratch Tensor storage produced by the first. Every output and frame is given. -/
theorem layer_norm_backward_source_fp32 (model : SourceModel.Model UInt32)
    (api : Memory.API UInt32 Store UInt8) (neg : UInt32 → UInt32)
    (x w dy r mu : Input) (c : KernelContract.Backward M N x w dy r mu)
    (B W : Nat) (_settings : Compiler.settings N=.ok (B,W)) (memory : World Store) :
    ∃ result,SourceBackwardStorage.exec model B api neg c memory=some result ∧
      result=SourceBackwardStorage.final model B api neg c memory ∧
      (∀ k i,logical api neg result (KernelBackward.outputs c k).tensor i.val=
        if KernelContract.launched M N then SourceBackwardResult.values model B api neg c memory k i
        else logical api neg memory (KernelBackward.outputs c k).tensor i.val) ∧
      (KernelContract.launched M N=Bool.false → result=memory) ∧
      (KernelContract.launched M N=Bool.true → ∀ k i,
        logical api neg result (c.first.buffers.output k).tensor i.val=
          SourceBackwardStorage.groupValues (SourceBackwardStorage.graphGroup model B api neg c memory) c k i) ∧
      (∀ key,KernelBackward.Outside c key → observe api result key=observe api memory key) ∧
      (∀ key,caller [x.dense.tensor,w.dense.tensor,dy.dense.tensor,r.dense.tensor,mu.dense.tensor] key →
        observe api result key=observe api memory key) :=
  SourceBackwardResult.executes model B api neg c memory

/-- Actual source execution over all FP32 words and a whole-state equivalent
unrolled candidate. The epsilon ABI condition is local to the scalar literal. -/
theorem layer_norm_forward_source_fp32 (model : SourceModel.Model UInt32)
    (epsilonConversion : UInt64 → UInt32) (eps : ℝ)
    (api : Memory.API UInt32 Store UInt8) (neg : UInt32 → UInt32)
    (c : KernelContract.Forward M N) (B W : Nat) (settings : Compiler.settings N=.ok (B,W))
    (epsilonABI : SourceForwardPublic.EpsilonABI model eps epsilonConversion c.epsilon) (memory : World Store) :
    SourceForwardGraph.lit model eps=epsilonConversion c.epsilon.toBits ∧
    (∃ result,SourceForwardStorage.exec model B eps api neg c memory=some result ∧
      result=SourceForwardStorage.final model B eps api neg c memory ∧
      (KernelContract.launched M N=Bool.true → ∀ k i,
        logical api neg result (c.buffers.output k).tensor i.val=SourceForwardStorage.graphValues model B eps api neg c memory k i) ∧
      (KernelContract.launched M N=Bool.false → result=memory) ∧
      (∀ key,Outside c.buffers key → observe api result key=observe api memory key) ∧
      (∀ key,caller [c.x.dense.tensor,c.w.dense.tensor,c.bias.dense.tensor] key →
        observe api result key=observe api memory key)) ∧
    (KernelContract.launched M N=Bool.true → ∀ r,r<M → ∃ t,
      SourceForwardStorage.runRow model B eps api neg c memory r=some t ∧
      SourceForwardStorage.candidateRunRow model B eps api neg c memory r=some t) := by
  have hb := (Compiler.settings_success N B W settings).2.1
  exact ⟨epsilonABI,SourceForwardStorage.executes model B eps api neg c memory hb,
    fun launch r _ => SourceForwardPublic.source_and_candidate model B eps api neg c memory launch hb r⟩

theorem layer_norm_forward_native_graph (ops : Ops UInt32)
    (epsilonConversion : UInt64 → UInt32) (p : KernelGraph.ForwardPlan N B)
    (api : Memory.API UInt32 Store UInt8) (neg : UInt32 → UInt32)
    (c : KernelContract.Forward M N) (m : World Store) :
    ∃ result,KernelForward.exec ops p api neg c (epsilonConversion c.epsilon.toBits) m=.ok result ∧
      result=KernelForward.final ops p api neg c (epsilonConversion c.epsilon.toBits) m ∧
      (KernelContract.launched M N=Bool.true → ∀ k i,
        logical api neg result (c.buffers.output k).tensor i.val=
          KernelForward.values ops p api neg c (epsilonConversion c.epsilon.toBits) m k i) ∧
      (KernelContract.launched M N=Bool.false → result=m) ∧
      (∀ key,Outside c.buffers key → observe api result key=observe api m key) ∧
      (∀ key,caller [c.x.dense.tensor,c.w.dense.tensor,c.bias.dense.tensor] key →
        observe api result key=observe api m key) :=
  KernelForward.executes ops p api neg c (epsilonConversion c.epsilon.toBits) m

theorem layer_norm_backward_native_graph (ops : Ops UInt32) (p : KernelGraph.BackwardPlan N B)
    (api : Memory.API UInt32 Store UInt8) (neg : UInt32 → UInt32)
    (x w dy r mu : Input) (c : KernelContract.Backward M N x w dy r mu) (m : World Store) :
    ∃ result,KernelBackward.exec ops p api c m=.ok result ∧
      result=KernelBackward.final ops p api c m ∧
      (∀ k i,logical api neg result (KernelBackward.outputs c k).tensor i.val=
        if KernelContract.launched M N then KernelBackward.nativeValues ops p api neg c m k i
        else logical api neg m (KernelBackward.outputs c k).tensor i.val) ∧
      (KernelContract.launched M N=Bool.false → result=m) ∧
      (KernelContract.launched M N=Bool.true → ∀ k i,
        logical api neg result (c.first.buffers.output k).tensor i.val=
          Execution.groupValues ops p.base api c.first x.dense w.dense dy.dense r.dense mu.dense m k i) ∧
      (∀ key,KernelBackward.Outside c key → observe api result key=observe api m key) ∧
      (∀ key,caller [x.dense.tensor,w.dense.tensor,dy.dense.tensor,r.dense.tensor,mu.dense.tensor] key →
        observe api result key=observe api m key) :=
  KernelBackward.executes ops p api neg c m


theorem layer_norm_forward_native (p : Policy) (B : Nat) (eps : ℝ) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Forward M N) (memory : World Store) (view : MatrixView c.x M N) (W : Nat) (settings : Compiler.settings N=.ok (B,W))
    (platform : SourceNativeBinding.Platform) (code : SourceNativeBinding.Selected platform .forward p)
    (_bound : SourceNativeBinding.Bound platform code view.stride M N B c.epsilon)
    (_epsilon : SourceNativeBinding.Epsilon platform code eps c.epsilon) :
    ∃ result,SourceNativeStorage.Forward.exec p B eps api neg c memory view=some result ∧
      result=SourceNativeStorage.Forward.final p B eps api neg c memory ∧
      (KernelContract.launched M N=Bool.true → ∀ k i,
        logical api neg result (c.buffers.output k).tensor i.val=SourceNativeStorage.Forward.values p (SourceNativeStorage.Forward.graphRow p B eps api neg c memory) c k i) ∧
      (KernelContract.launched M N=Bool.false → result=memory) ∧
      (∀ key,Outside c.buffers key → observe api result key=observe api memory key) ∧
      (∀ key,caller [c.x.dense.tensor,c.w.dense.tensor,c.bias.dense.tensor] key →
        observe api result key=observe api memory key) :=
  SourceNativeForwardResult.executes p B eps api neg c memory view (Compiler.settings_success N B W settings).2.1

#axiomsClean layer_norm_forward_native

theorem layer_norm_backward_native (p q : SourceNativeStorage.Policy) (B : Nat) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (x w dy r mu : Input) (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (view : MatrixView x M N)
    (W : Nat) (_settings : Compiler.settings N=.ok (B,W))
    (platform : SourceNativeBinding.Platform)
    (grouped : SourceNativeBinding.Selected platform .groupedBackward p)
    (affine : SourceNativeBinding.Selected platform .finalBackward q)
    (hostEpsilon : Float)
    (_groupedBound : SourceNativeBinding.Bound platform grouped view.stride M N B hostEpsilon)
    (_affineBound : SourceNativeBinding.Bound platform affine view.stride M N B hostEpsilon) :
    ∃ result,SourceNativeStorage.Backward.exec p q B api neg c memory view=some result ∧
      result=SourceNativeStorage.Backward.final p q B api neg c memory ∧
      (∀ k i,logical api neg result (KernelBackward.outputs c k).tensor i.val=
        if KernelContract.launched M N then SourceNativeResult.values p q B api neg c memory k i
        else logical api neg memory (KernelBackward.outputs c k).tensor i.val) ∧
      (KernelContract.launched M N=Bool.false → result=memory) ∧
      (KernelContract.launched M N=Bool.true → ∀ k i,
        logical api neg result (c.first.buffers.output k).tensor i.val=SourceNativeStorage.Backward.groupValues p (SourceNativeStorage.Backward.graphGroup p B api neg c memory) c k i) ∧
      (∀ key,KernelBackward.Outside c key → observe api result key=observe api memory key) ∧
      (∀ key,caller [x.dense.tensor,w.dense.tensor,dy.dense.tensor,r.dense.tensor,mu.dense.tensor] key →
        observe api result key=observe api memory key) :=
  SourceNativeResult.executes p q B api neg c memory view

#axiomsClean layer_norm_backward_native

theorem layer_norm_composed_native (p q z : Policy) (B : Nat) (eps : ℝ) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (f : KernelContract.Forward M N) (dy r mu : Input) (b : KernelContract.Backward M N f.x f.w dy r mu)
    (rBinding : r.dense=f.buffers.output 0) (muBinding : mu.dense=f.buffers.output 1)
    (memory : World Store) (view : MatrixView f.x M N) (W : Nat) (settings : Compiler.settings N=.ok (B,W))
    (platform : SourceNativeBinding.Platform)
    (forwardCode : SourceNativeBinding.Selected platform .forward p)
    (groupedCode : SourceNativeBinding.Selected platform .groupedBackward q)
    (affineCode : SourceNativeBinding.Selected platform .finalBackward z)
    (_forwardBound : SourceNativeBinding.Bound platform forwardCode view.stride M N B f.epsilon)
    (_groupedBound : SourceNativeBinding.Bound platform groupedCode view.stride M N B f.epsilon)
    (_affineBound : SourceNativeBinding.Bound platform affineCode view.stride M N B f.epsilon)
    (_epsilon : SourceNativeBinding.Epsilon platform forwardCode eps f.epsilon) :
    ∃ forward backward,
      Forward.exec p B eps api neg f memory view=some forward ∧
      Backward.exec q z B api neg b forward view=some backward ∧
      forward=Forward.final p B eps api neg f memory ∧
      backward=Backward.final q z B api neg b forward ∧
      (KernelContract.launched M N=Bool.true → ∀ k i,
        logical api neg forward (f.buffers.output k).tensor i.val=Forward.values p (Forward.graphRow p B eps api neg f memory) f k i) ∧
      (∀ k i,logical api neg backward (KernelBackward.outputs b k).tensor i.val=
        if KernelContract.launched M N then SourceNativeResult.values q z B api neg b forward k i
        else logical api neg forward (KernelBackward.outputs b k).tensor i.val) ∧
      (KernelContract.launched M N=Bool.true → ∀ row,row<M →
        KernelBackward.initialStats api neg r mu forward row=
          ⟨p.read (Forward.graphRow p B eps api neg f memory row) "Mean" row,
           p.read (Forward.graphRow p B eps api neg f memory row) "Rstd" row⟩) ∧
      (KernelContract.launched M N=Bool.false → forward=memory ∧ backward=memory) ∧
      (∀ key,Outside f.buffers key → KernelBackward.Outside b key → observe api backward key=observe api memory key) :=
  SourceNativeComposition.executes p q z B eps api neg f dy r mu b rBinding muBinding memory view (Compiler.settings_success N B W settings).2.1

#axiomsClean layer_norm_composed_native

theorem layer_norm_candidate_native_memory (p : Policy) (B : Nat) (eps : ℝ)
    (api : Memory.API Word Store Byte) (neg : Word → Word) (c : KernelContract.Forward M N)
    (memory : World Store) (view : MatrixView c.x M N) (W : Nat) (settings : Compiler.settings N=.ok (B,W)) :
    ∃ result,Forward.exec p B eps api neg c memory view=some result ∧
      SourceNativeCandidate.exec p B eps api neg c memory view=some result ∧ result=Forward.final p B eps api neg c memory :=
  SourceNativeCandidate.executes p B eps api neg c memory view (Compiler.settings_success N B W settings).2.1

#axiomsClean layer_norm_candidate_native_memory

#axiomsClean layer_norm_forward_native_graph
#axiomsClean layer_norm_forward_source_fp32
#axiomsClean layer_norm_backward_source_fp32
#axiomsClean layer_norm_backward_native_graph
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.TensorBridge.Equivalence

/-! Retained saved-statistic Tensor readback applications. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.TensorBridge
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm.Storage
open KernelMemory
noncomputable section

theorem layer_norm_saved_source_real (model : SourceModel.Model ℝ) (laws : SourceForwardReal.Laws model)
    (B : Nat) (projection : SourceIdealInput.ScalarProjection) (c : KernelContract.Forward M N) (memory : World (Nat → ℝ)) (r mu : Input)
    (rBinding : r.dense=c.buffers.output 0) (muBinding : mu.dense=c.buffers.output 1)
    (launch : KernelContract.launched M N=Bool.true) (hb : N≤B) (row : Nat) (hr : row<M) :
    KernelBackward.initialStats Memory.realAPI Neg.neg r mu
      (SourceForwardStorage.final model B (SourceIdealInput.epsilon projection c) Memory.realAPI Neg.neg c memory) row=
      ⟨RealProof.Spec.mean N (KernelForward.row Memory.realAPI Neg.neg c.x memory N row),
       RealProof.Spec.inverse N (KernelForward.row Memory.realAPI Neg.neg c.x memory N row) (SourceIdealInput.epsilon projection c)⟩ :=
  SourceSaved.saved_real model laws B (SourceIdealInput.epsilon projection c) c memory r mu rBinding muBinding launch hb row hr

theorem layer_norm_saved_graph_real (p : KernelGraph.ForwardPlan N B)
    (c : KernelContract.Forward M N) (projection : SourceIdealInput.ScalarProjection) (m : World (Nat → ℝ)) (r mu : Input)
    (rBinding : r.dense=c.buffers.output 0) (muBinding : mu.dense=c.buffers.output 1)
    (launch : KernelContract.launched M N=Bool.true) (row : Nat) (hr : row<M) :
    KernelBackward.initialStats Memory.realAPI Neg.neg r mu
      (KernelForward.final RealProof.ops p Memory.realAPI Neg.neg c (SourceIdealInput.epsilon projection c) m) row=
      ⟨RealProof.Spec.mean N (KernelForward.row Memory.realAPI Neg.neg c.x m N row),
       RealProof.Spec.inverse N (KernelForward.row Memory.realAPI Neg.neg c.x m N row) (SourceIdealInput.epsilon projection c)⟩ :=
  KernelSaved.saved_real p c (SourceIdealInput.epsilon projection c) m r mu rBinding muBinding launch row hr

theorem layer_norm_saved_native_graph (ops : Ops UInt32) (epsilonConversion : UInt64 → UInt32)
    (p : KernelGraph.ForwardPlan N B) (api : Memory.API UInt32 Store UInt8) (neg : UInt32 → UInt32)
    (c : KernelContract.Forward M N) (m : World Store) (r mu : Input)
    (rBinding : r.dense=c.buffers.output 0) (muBinding : mu.dense=c.buffers.output 1)
    (launch : KernelContract.launched M N=Bool.true) (row : Nat) (hr : row<M) :
    KernelBackward.initialStats api neg r mu
      (KernelForward.final ops p api neg c (epsilonConversion c.epsilon.toBits) m) row=
      KernelGraph.stats ops p (KernelForward.row api neg c.x m N row) (epsilonConversion c.epsilon.toBits) 0 :=
  KernelSaved.saved ops p api neg c (epsilonConversion c.epsilon.toBits) m r mu rBinding muBinding launch row hr

#axiomsClean layer_norm_saved_source_real
#axiomsClean layer_norm_saved_graph_real
#axiomsClean layer_norm_saved_native_graph
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.TensorBridge
