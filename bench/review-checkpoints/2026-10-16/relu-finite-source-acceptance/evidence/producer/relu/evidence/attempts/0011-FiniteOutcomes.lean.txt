import bench.tritonbench_g.relu_triton_kernel.Proofs.FiniteHost

namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteHost
open VeriTile.Triton HostWrapper GatedGrid FP64Host FiniteBoundary
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def Ready {α : Type} (views : TensorSemantics) (x : Tensor) (s : Runtime α) : Prop :=
  match admission x.dtype with
  | .typed p => ∃ storage, TypedInput p x (views x) storage s
  | .fp64 => ∃ storage, WideInput x (views x) storage s
  | .comparisonRejected | .signatureRejected => True
  | .outsideDomain => False

/-- An independent postcondition. Error constructors retain the first failing
primitive and its exact source event prefix. Success constructors contain
storage-relative values, metadata, finite addresses and the original tail bug.
It contains no interpreter, assumed kernel result or assumed launch success. -/
inductive Outcome {α : Type} [Inhabited α] (views : TensorSemantics)
    (P : Primitives (Runtime α)) (M : FP.Structural.Algebra α)
    (cmp16 cmp32 : α → α → Bool) (num : FP64Source.Numerics)
    (fault : Option (Error × Runtime α)) (x : Tensor) (s : Runtime α) (trace : List Event) :
    Result (HostState (Runtime α)) Tensor → Prop where
  | deviceError (e t) (h : P.device x s = .error e t) :
      Outcome views P M cmp16 cmp32 num fault x s trace
        (.error e ⟨t, trace ++ [.device x]⟩)
  | allocationError (device d e t) (hd : P.device x s = .ok device d)
      (ha : P.emptyLike x "float32" device d = .error e t) :
      Outcome views P M cmp16 cmp32 num fault x s trace
        (.error e ⟨t, trace ++ [.device x, .emptyLike x "float32" device]⟩)
  | numelError (device d output a e t) (hd : P.device x s = .ok device d)
      (ha : P.emptyLike x "float32" device d = .ok output a)
      (hm : P.numel output a = .error e t) :
      Outcome views P M cmp16 cmp32 num fault x s trace
        (.error e ⟨t, trace ++ [.device x, .emptyLike x "float32" device, .numel output]⟩)
  | launchError (device d output a n m e t) (hd : P.device x s = .ok device d)
      (ha : P.emptyLike x "float32" device d = .ok output a)
      (hm : P.numel output a = .ok n m) (hf : fault = some (e,t)) :
      Outcome views P M cmp16 cmp32 num fault x s trace
        (.error e ⟨t, trace ++ (successTrace x output device n).take 5⟩)
  | comparisonRejected (device d output a n m) (hd : P.device x s = .ok device d)
      (ha : P.emptyLike x "float32" device d = .ok output a)
      (hm : P.numel output a = .ok n m) (hf : fault = none)
      (rejected : admission x.dtype = .comparisonRejected) :
      Outcome views P M cmp16 cmp32 num fault x s trace
        (.error (.native "frontend: fp8 ordered comparison rejected")
          ⟨m, trace ++ (successTrace x output device n).take 5⟩)
  | signatureRejected (device d output a n m) (hd : P.device x s = .ok device d)
      (ha : P.emptyLike x "float32" device d = .ok output a)
      (hm : P.numel output a = .ok n m) (hf : fault = none)
      (rejected : admission x.dtype = .signatureRejected) :
      Outcome views P M cmp16 cmp32 num fault x s trace
        (.error (.native "frontend: fp8 signature rejected")
          ⟨m, trace ++ (successTrace x output device n).take 5⟩)
  | typed (device d output a n m final p storage) (hd : P.device x s = .ok device d)
      (ha : P.emptyLike x "float32" device d = .ok output a)
      (hm : P.numel output a = .ok n m) (hf : fault = none)
      (input : TypedInput p x (views x) storage s)
      (alloc : Allocation views x output s a) (access : AccessSummary views x output)
      (post : TypedPost M cmp16 cmp32 p x output (views x) storage s a final) :
      Outcome views P M cmp16 cmp32 num fault x s trace
        (.ok output ⟨final, trace ++ successTrace x output device (numel x)⟩)
  | wide (device d output a n m final storage) (hd : P.device x s = .ok device d)
      (ha : P.emptyLike x "float32" device d = .ok output a)
      (hm : P.numel output a = .ok n m) (hf : fault = none)
      (input : WideInput x (views x) storage s)
      (alloc : Allocation views x output s a) (access : AccessSummary views x output)
      (post : WidePost num x output (views x) storage s a final) :
      Outcome views P M cmp16 cmp32 num fault x s trace
        (.ok output ⟨final, trace ++ successTrace x output device (numel x)⟩)

theorem all_outcomes {α : Type} [Inhabited α] (views : TensorSemantics)
    (P : Primitives (Runtime α)) (C : PrimitivesContract views P)
    (M : FP.Structural.Algebra α) (cmp16 cmp32 : α → α → Bool)
    (num : FP64Source.Numerics) (undef : Nat → BitVec 64) (order : Nat → List Nat)
    (fault : Option (Error × Runtime α)) (x : Tensor) (s : Runtime α) (trace : List Event)
    (domain : SourceDomain x) (ready : Ready views x s)
    (complete : (order (gridCount (numel x) 1024)).Perm (List.range (gridCount (numel x) 1024))) :
    Outcome views P M cmp16 cmp32 num fault x s trace
      (relu (withLaunch P M cmp16 cmp32 num undef order fault) x ⟨s, trace⟩) := by
  cases hd : P.device x s with
  | error e t =>
    rw [device_error_first (withLaunch P M cmp16 cmp32 num undef order fault) x s t trace e hd]
    exact .deviceError e t hd
  | ok device d =>
    cases ha : P.emptyLike x "float32" device d with
    | error e t =>
      rw [allocation_error_after_device (withLaunch P M cmp16 cmp32 num undef order fault) x s d t trace device e hd ha]
      exact .allocationError device d e t hd ha
    | ok output a =>
      cases hm : P.numel output a with
      | error e t =>
        rw [numel_error_after_allocation (withLaunch P M cmp16 cmp32 num undef order fault) x output s d a t trace device e hd ha hm]
        exact .numelError device d output a e t hd ha hm
      | ok n m =>
        cases fault with
        | some f =>
          rcases f with ⟨e,t⟩
          rw [launch_error_stops_return (withLaunch P M cmp16 cmp32 num undef order (some (e,t))) x output s d a m t trace device n e hd ha hm (by rfl)]
          exact .launchError device d output a n m e t hd ha hm rfl
        | none =>
          rw [no_fault]
          rcases finite_classification x domain with ⟨p, selected⟩ | selected | selected | selected
          · obtain ⟨storage,input⟩ : ∃ storage, TypedInput p x (views x) storage s := by
              simpa [Ready, selected] using ready
            obtain ⟨final,run,alloc,post⟩ := typed_correct views P C M cmp16 cmp32 num undef order
              p x output s d a m trace device n storage input hd ha hm
            rw [run]
            exact .typed device d output a n m final p storage hd ha hm rfl input alloc
              (access_summary views x output s a domain input.valid input.readable alloc) post
          · obtain ⟨storage,input⟩ : ∃ storage, WideInput x (views x) storage s := by
              simpa [Ready, selected] using ready
            obtain ⟨final,run,alloc,post⟩ := wide_correct views P C M cmp16 cmp32 num undef order
              x output s d a m trace device n storage input complete hd ha hm
            rw [run]
            exact .wide device d output a n m final storage hd ha hm rfl input alloc
              (access_summary views x output s a domain input.valid input.readable alloc) post
          · rw [relu_comparison_rejection P M cmp16 cmp32 num undef order x output s d a m trace
              device n selected hd ha hm]
            exact .comparisonRejected device d output a n m hd ha hm rfl selected
          · rw [relu_signature_rejection P M cmp16 cmp32 num undef order x output s d a m trace
              device n selected hd ha hm]
            exact .signatureRejected device d output a n m hd ha hm rfl selected

#specNonCircular Ready avoiding [HostWrapper.relu, FP64Host.logicalLaunch]
#specNonCircular Outcome avoiding [HostWrapper.relu, FP64Host.logicalLaunch]
#axiomsClean all_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteHost
