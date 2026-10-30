import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.GridGeometry
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.CarrySafety
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.CarryNativeLoop
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.CarryIO

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.CarryFiniteGrid
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.AttentionLlama HostGrid MaskCoverage NativeSchedule
open Backing FiniteInputs GridGeometry
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

/-- Full physical source grid over finite logical input roles. This theorem
reuses the accepted source execution and native formula connection; it removes
only irrelevant byte conditions on cells the source never reads. -/
theorem rows (A : Target.Ops α) (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument)
    (B : Nat) (c : Config) (hD : 0<c.D) (hN : 0<c.N)
    (hinj : Function.Injective (IOLayout.output (B:=B) c))
    (vals : ChannelValues α (IOLayout.inputs B c)) (s : State (Term α))
    (hin : (CarryIO.io A C arg B c).InputBinding vals s)
    (bytes : IOLayout.nativeDomain B c vals) :
    ∃ states : CarryIO.Program B c→State (Term α),
      (∀ p,(CarryIO.invocation C arg B c p).run s=some (states p)) ∧
      (∀ i : TileIndex [B,c.NH,c.H,c.D],
        mapCell (Native.lower A c.fp8 c.D (lane i.2.2.2.1.val))
          ((states (owner (index i))).mem c.Out (IOLayout.output c i))=
            .mk .fp16 (Mathematical.native A C arg B c vals 0 i)) ∧
      (∀ p r o,¬CarryIO.writes B c p r o → (states p).mem r o=s.mem r o) := by
  have runs:=fun p : CarryIO.Program B c=>
    CarrySource.values_and_frame_of_reached c (Tensor.initialFor c s (anchor hD p)) (Native.argument C arg)
      (reached A C arg B c vals s hin bytes (anchor hD p))
      (Tensor.block_injective c s (tensor_injective c hinj) (anchor hD p))
  choose states he hv hf using runs
  refine ⟨states,?_,?_,?_⟩
  · intro p
    unfold Invocation.run
    rw [CarryIO.invocation_initial C arg B c hD p s]
    exact he p
  · intro j
    let i:=index j
    have active : Source.rows (Tensor.initialFor c s i) (Tensor.laneIndex i).1<c.H := by
      rw [Tensor.row_eq]
      exact i.query.isLt
    have vi:=hv (owner i) (Tensor.laneIndex i)
    rw [initial_owner c s hD i] at vi
    have address_eq : IOLayout.output c j=address c (Tensor.initialFor c s i) (Tensor.laneIndex i) :=
      (Tensor.address_eq c s i).symm
    rw [address_eq,vi active]
    change Cell.mk .fp16 (Native.lower A c.fp8 c.D (lane i.channel.val)
      (.half ((CarrySource.accumulated c (Tensor.initialFor c s i) (Native.argument C arg)
        (count (stop c (Tensor.initialFor c s i)))).accumulator (Tensor.laneIndex i))))=_
    have result:=CarryNative.output_binding A C arg c (Tensor.initialFor c s i) (Tensor.laneIndex i).1
      ((tensorInputs A B c vals).at i) (Tensor.bindings c s _ (binding A C arg B c vals s hin) i)
      active hN i.channel
    apply congrArg (Cell.mk .fp16)
    exact result
  · intro p r o miss
    apply hf p r o
    by_cases hr:r=c.Out
    · right
      intro j hj heq
      apply miss
      rw [hr,heq]
      exact active_writes c s hD p j hj
    · exact Or.inl hr

/-- Safety is derived from exactly the same declared finite windows and same
runtime scalar source as the successful executions above. -/
theorem safe (A : Target.Ops α) (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument)
    (B : Nat) (c : Config) (hD : 0<c.D)
    (vals : ChannelValues α (IOLayout.inputs B c)) (s : State (Term α))
    (hin : (CarryIO.io A C arg B c).InputBinding vals s)
    (bytes : IOLayout.nativeDomain B c vals) (capacity : RegionName→Nat)
    (hb : (CarryIO.io A C arg B c).InBounds capacity vals) (p : CarryIO.Program B c) :
    (CarryIO.invocation C arg B c p).Safe capacity s := by
  unfold Invocation.Safe
  rw [CarryIO.invocation_initial C arg B c hD p s]
  exact CarrySource.source_safe c (Tensor.initialFor c s (anchor hD p)) (Native.argument C arg)
    (reached A C arg B c vals s hin bytes (anchor hD p)) capacity
    (FiniteInputs.bounds A C arg B c vals s capacity hb (anchor hD p))

#axiomsClean rows
#axiomsClean safe
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.CarryFiniteGrid
