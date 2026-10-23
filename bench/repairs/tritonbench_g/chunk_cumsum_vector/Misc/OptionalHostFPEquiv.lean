import bench.repairs.tritonbench_g.chunk_cumsum_vector.Kernels
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.Contract
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.FPWrapper
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.NativeGraph

/-! Complete source FP result and observed accumulating-dot graph specialization.
Both retain configured reduction, cast/store order and explicit zero padding.
No split-versus-fused floating equality is used. -/
namespace VeriTile.Bench.Repairs.ChunkCumsumVector.OptionalHostFPEquiv
open VeriTile.Triton VeriTile.Torch FP.Structural FPValues
noncomputable section
variable {α : Type} [Inhabited α]

def defaultPrecision : ComputeDType := .fp32
def sourceReference (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config)
    (positive : 0<c.BT) (prog : Geometry.Program h)
    (logical : Nat → Nat → Nat → α) (time : Nat) (j : Fin BS) : Cell α :=
  FPReference.expected M dot h c positive prog logical
    (fun d _ _ => defaultValue d) time j

inductive NumericalSite where
  | literal | loadConversion | paddingConversion | accumulatingDotNoTF32 | storeConversion | reduction | carryAdd
  deriving DecidableEq
def numericalSites : List NumericalSite := [.literal,.loadConversion,.paddingConversion,.accumulatingDotNoTF32,
  .storeConversion,.reduction,.carryAdd]

theorem wrapper_source_fp32 (s : TensorMetadata) (dtype : Option DType)
    (contract : InputContract s dtype) (h : Call) (storage : Storage.Placement h)
    (binding : TensorBinding s h storage) (M : Config → Algebra α) (dot : Config → DotBackend α)
    (padding : FPWrapper.Oracle α h) (api : Host.API (Cell α))
    (tried : List Config) (valid : ∀ c∈tried, c∈configurations)
    (selected : Config) (chosen : selected∈configurations)
    (caller : FPStorage.Heap α) (live : Nat → Prop) (logical : Nat → Nat → Nat → α)
    (ordinaryAPI : Host.PreparationLaws api h storage (FPReference.read h) logical caller live)
    (allocation : Host.Allocated (Cell α)) (allocated : api.emptyLike caller=.ok allocation)
    (device : allocation.device=s.device) (launchDevice : api.launchDevice=s.device)
    (setups : allocation.shape.prod≠0 → Host.Setups api h 0 (tried++[selected])) :
    ∃ result, Host.guarded s dtype api (FPWrapper.grid M dot h storage padding) h tried selected caller=
        .ok {allocation with heap:=result} ∧
      Host.metadata h storage {allocation with heap:=result} ∧
      (∀ (row : Fin (h.B*h.H)) (time : Fin h.T) (channel : Fin h.S),
        result (Storage.address storage.backing.output (h.outputStrides.address h.H row.val time.val channel.val))=
          sourceReference (M selected) (dot selected) h selected (Kernels.configurations_positive selected chosen).1
            (Grid.owner h row channel) logical
            time.val (Grid.lane h channel)) ∧
      (∀ addr, live addr → result addr=caller addr) ∧
      (∀ addr, ¬Host.preparationWrites h storage addr → result addr=caller addr) ∧
      (∀ addr, ¬Storage.writes h storage (Grid.programs h) addr → result addr=allocation.heap addr) ∧
      allocation.device=s.device ∧ api.launchDevice=allocation.device := by
  have prepared : Host.prepare api Bool.true caller=.ok allocation := by simp [Host.prepare,allocated]
  obtain ⟨result,returned,metadata,out,liveFrame,frame,outputFrame⟩ :=
    FPWrapper.complete M dot h binding.outputLayout storage padding api Bool.true tried valid selected chosen
      caller live logical ordinaryAPI allocation prepared setups
  refine ⟨result,(Host.guarded_eq s dtype contract api _ h tried selected caller).trans returned,
    metadata,?_,liveFrame,frame,outputFrame,device,launchDevice.trans device.symm⟩
  intro row time channel
  exact (out row time channel).trans (FPReference.expected_padding_irrelevant
    (M selected) (dot selected) h selected _ _ logical _ _ _ _)

/-- Native graph reference: the incoming carry is the dot accumulator.
The graph is independent of the execution result and has raw FP32 word leaves. -/
def reference (rest : Algebra NativeGraph.Expr) (h : Call) (c : Config)
    (positive : 0<c.BT) (prog : Geometry.Program h)
    (words : Nat → Nat → Nat → BitVec 32) (time : Nat) (j : Fin BS) : Cell NativeGraph.Expr :=
  sourceReference (NativeGraph.algebra rest c) (NativeGraph.backend c) h c positive prog
    (fun r t s => .leaf (words r t s)) time j

theorem wrapper_fp32 (s : TensorMetadata) (dtype : Option DType)
    (contract : InputContract s dtype) (h : Call) (storage : Storage.Placement h)
    (binding : TensorBinding s h storage) (rest : Algebra NativeGraph.Expr)
    (padding : FPWrapper.Oracle NativeGraph.Expr h) (api : Host.API (Cell NativeGraph.Expr))
    (tried : List Config) (valid : ∀ c∈tried, c∈configurations)
    (selected : Config) (chosen : selected∈configurations)
    (caller : FPStorage.Heap NativeGraph.Expr) (live : Nat → Prop)
    (words : Nat → Nat → Nat → BitVec 32)
    (ordinaryAPI : Host.PreparationLaws api h storage (FPReference.read h)
      (fun r t s => NativeGraph.Expr.leaf (words r t s)) caller live)
    (allocation : Host.Allocated (Cell NativeGraph.Expr)) (allocated : api.emptyLike caller=.ok allocation)
    (device : allocation.device=s.device) (launchDevice : api.launchDevice=s.device)
    (setups : allocation.shape.prod≠0 → Host.Setups api h 0 (tried++[selected])) :
    ∃ result, Host.guarded s dtype api
        (FPWrapper.grid (NativeGraph.algebra rest) NativeGraph.backend h storage padding) h tried selected caller=
        .ok {allocation with heap:=result} ∧
      Host.metadata h storage {allocation with heap:=result} ∧
      (∀ (row : Fin (h.B*h.H)) (time : Fin h.T) (channel : Fin h.S),
        result (Storage.address storage.backing.output (h.outputStrides.address h.H row.val time.val channel.val))=
          reference rest h selected (Kernels.configurations_positive selected chosen).1
            (Grid.owner h row channel) words time.val (Grid.lane h channel)) ∧
      (∀ addr, live addr → result addr=caller addr) ∧
      (∀ addr, ¬Host.preparationWrites h storage addr → result addr=caller addr) ∧
      (∀ addr, ¬Storage.writes h storage (Grid.programs h) addr → result addr=allocation.heap addr) ∧
      allocation.device=s.device ∧ api.launchDevice=allocation.device ∧
      (∀ (row : Fin (h.B*h.H)) (time : Fin h.T) (channel : Fin h.S),
        NativeGraph.Materialized ((result (Storage.address storage.backing.output
          (h.outputStrides.address h.H row.val time.val channel.val))).read .fp32)) := by
  obtain ⟨result,returned,metadata,out,liveFrame,frame,outputFrame,dev,launch⟩ :=
    wrapper_source_fp32 s dtype contract h storage binding (NativeGraph.algebra rest) NativeGraph.backend
      padding api tried valid selected chosen caller live (fun r t s => NativeGraph.Expr.leaf (words r t s))
      ordinaryAPI allocation allocated device launchDevice setups
  refine ⟨result,returned,metadata,out,liveFrame,frame,outputFrame,dev,launch,?_⟩
  intro row time channel
  rw [out row time channel]
  exact NativeGraph.expected_materialized rest h binding.outputDType selected _ (Grid.owner h row channel)
    (fun r t s => NativeGraph.Expr.leaf (words r t s)) (fun _ _ _ => True.intro) _ time.val (Grid.lane h channel)

def native_pid := TensorBinding.pid
abbrev native_addresses := NativeIndex.source_addresses
abbrev native_coordinates := NativeIndex.source_coordinates
abbrev native_indices := NativeIndex.source_indices

#axiomsClean wrapper_fp32
#print axioms wrapper_fp32
#axiomsClean wrapper_source_fp32
#print axioms wrapper_source_fp32
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.OptionalHostFPEquiv
