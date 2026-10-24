import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.KernelResult
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.KernelAddresses

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.KernelPhysical
open VeriTile.Torch VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open KernelDomain KernelBinding KernelModel
noncomputable section
local instance : Inhabited ValueProofs.Word := ⟨FP.Value.zero .fp32⟩
local instance : Inhabited SourceGraph.Expr := ⟨SourceGraph.Expr.zero⟩

/-- The reused complete-grid interpreter executes precisely Kernels.source. -/
theorem source_binding (p : Parameters) : SourceGraph.source p=source p := rfl

/-- The actual output Tensor also binds the initial tagged memory. Its initial
contents may be arbitrary: every output coordinate is overwritten, including
K=0. The Python zero allocation is therefore a supported special case. -/
def allocatedScratch (z : Allocated shape) (heap : Heap) (scratch : State ValueProofs.Word) :
    State ValueProofs.Word :=
  { scratch with mem := fun r i => if r="out" then .mk .real (z.value heap i) else scratch.mem r i }

/-- Complete logical/physical word binding on all three actual Tensor windows. -/
def Realizes (a : Arguments) (x : Allocated [a.M,a.K]) (y : Allocated [a.M,a.N,a.K])
    (z : Allocated [a.M,a.N]) (heap : Heap) (mem : ValueProofs.Memory) : Prop :=
  (∀ i,i<x.count → mem "a" i=.mk .real (x.value heap i)) ∧
  (∀ i,i<y.count → mem "b" i=.mk .real (y.value heap i)) ∧
  (∀ i,i<z.count → mem "out" i=.mk .real (z.value heap i))

theorem initial_realizes (a : Arguments) (x : Allocated [a.M,a.K]) (y : Allocated [a.M,a.N,a.K])
    (z : Allocated [a.M,a.N]) (heap : Heap) (scratch : State ValueProofs.Word) :
    Realizes a x y z heap (inputMemory (x.value heap) (y.value heap) (allocatedScratch z heap scratch).mem) := by
  simp [Realizes,inputMemory,allocatedScratch]

theorem input_value_frame (x : Allocated inputShape) (z : Allocated outputShape)
    (sep : x.tensor.storage.base/4+x.tensor.storage.elements≤z.base ∨
      z.base+z.count≤x.tensor.storage.base/4)
    (heap : Heap) (values : Nat → ValueProofs.Word) (i : Nat) (inside : i<x.count) :
    x.value (physical z heap values) i=x.value heap i := by
  rw [x.value_observe,x.value_observe]
  unfold Allocated.observe FP32.Storage.address
  have eqn : (x.tensor.storage.base+4*x.tensor.physicalIndex i)/4=
      x.tensor.storage.base/4+x.tensor.physicalIndex i := by omega
  rw [eqn]
  exact input_storage_frame z x.tensor.storage sep heap values _ (x.laws.inBounds i inside)

theorem final_realizes (cfg : FP.Config) (a : Arguments)
    (x : Allocated [a.M,a.K]) (y : Allocated [a.M,a.N,a.K]) (z : Allocated [a.M,a.N])
    (binding : Binding a x y z) (heap : Heap) (scratch : State ValueProofs.Word)
    (final : State SourceGraph.Expr)
    (frame : ∀ r i,SourceGraph.GridOutside (parameters a) r i →
      SourceGraph.Native.decodeMemory cfg final.mem r i=
        inputMemory (x.value heap) (y.value heap) (allocatedScratch z heap scratch).mem r i)
    (typed : ∀ i,i<z.count → ∃ v,final.mem "out" i=.mk .real v) :
    Realizes a x y z (physical z heap (nativeValues cfg final))
      (SourceGraph.Native.decodeMemory cfg final.mem) := by
  refine ⟨?_,?_,?_⟩
  · intro i hi
    rw [frame "a" i (Or.inl (by change ("a" : RegionName)≠"out"; decide)),
      input_value_frame x z binding.separateA heap _ i hi]
    simp [inputMemory]
  · intro i hi
    rw [frame "b" i (Or.inl (by change ("b" : RegionName)≠"out"; decide)),
      input_value_frame y z binding.separateB heap _ i hi]
    simp [inputMemory]
  · intro i hi
    rw [output_value z heap _ i hi]
    obtain ⟨v,hv⟩ := typed i hi
    unfold nativeValues SourceGraph.Native.decodeMemory
    rw [hv]
    rfl

/-- Full-grid coverage establishes output dtype as well as its value. -/
theorem output_typed (rest : Algebra SourceGraph.Expr) (plans : SourceGraph.PlanSelector)
    (a : Arguments) (h : KernelContract a) (native : LaunchABI a)
    (x : Allocated [a.M,a.K]) (y : Allocated [a.M,a.N,a.K]) (z : Allocated [a.M,a.N])
    (heap : Heap) (scratch : State ValueProofs.Word) (final : State SourceGraph.Expr)
    (run : SourceGraph.runGrid rest plans (parameters a)
      (initial (x.value heap) (y.value heap) scratch)=some final) :
    ∀ i,i<z.count → ∃ v,final.mem "out" i=.mk .real v := by
  obtain ⟨other,he,_,hc,_⟩ := KernelResult.graph_correct rest plans a h native x y z heap scratch
  have same : other=final := Option.some.inj (he.symm.trans run)
  subst other
  intro i hi
  have bound : i<a.M*a.N := by simpa using hi
  have pos : 0<a.N := by nlinarith
  let m : Fin a.M := ⟨i/a.N,(Nat.div_lt_iff_lt_mul pos).2 bound⟩
  let n : Fin a.N := ⟨i%a.N,Nat.mod_lt i pos⟩
  have eqn : m.val*a.N+n.val=i := by
    dsimp [m,n]
    simpa only [Nat.mul_comm] using Nat.div_add_mod i a.N
  rw [←eqn]
  exact ⟨_,hc m n⟩

#axiomsClean initial_realizes
#axiomsClean source_binding
#axiomsClean input_value_frame
#axiomsClean final_realizes
#axiomsClean output_typed
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.KernelPhysical
