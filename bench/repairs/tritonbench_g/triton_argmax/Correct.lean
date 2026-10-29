import bench.repairs.tritonbench_g.triton_argmax.Proofs.NativeSafety
import bench.repairs.tritonbench_g.triton_argmax.Proofs.RealScratch
import bench.repairs.tritonbench_g.triton_argmax.Proofs.RealMemory

namespace VeriTile.Bench.Repairs.TritonArgmax.Correct
open Source Memory TensorBinding Reduction
noncomputable section

/-- Arbitrary independent Real interpretation at the supplied Tensor's actual
physical coordinates; ordinary negative-view semantics is retained. -/
def logical (x : Input) (heap : Address → ℝ) (i : Nat) : ℝ :=
  let value := heap (address x.storage 4 (x.tensor.physicalIndex i))
  if x.tensor.metadata.isNeg then -value else value

def reference (x : Input) (heap : Address → ℝ) (positive : 0 < x.tensor.metadata.numel) : Nat :=
  (Math.argmax x.tensor.metadata.numel positive (fun i => logical x heap i.val)).val

def initial (heap : Address → ℝ) (indices : Address → Nat) : RealScratch.State :=
  ⟨fun point => ↑(heap point),indices⟩

structure RealResult where
  returned : AllocatedTensor
  state : RealScratch.State

/-- Complete Real projection of the accepted precision-adapter macro:
zeroed index padding, every first-stage store, second-stage loads/reduction,
indirect index gather and final store. No native output is an input. -/
def globalInvoke (x : Input) (o : Checked.Observation) (adapter : RealScratch.Adapter)
    (g : NativeGraph.GlobalPrograms adapter.format o)
    (a : GlobalAllocation x o adapter.format callers)
    (heap : Address → ℝ) (indices : Address → Nat) : Except String RealResult :=
  if Checked.check o then .ok ⟨a.output.returned,
    RealScratch.run adapter g.config a.buffers (logical x heap) (initial heap indices)⟩
  else .error "input contract rejected"

theorem point_ne (word point : Address) (width : Nat) (positive : 0 < width)
    (apart : Memory.Disjoint word width point 1) : point ≠ word := by
  intro same
  subst point
  rcases apart with owners | left | right
  · exact owners rfl
  · omega
  · omega

theorem argmax_size (n m : Nat) (hn : 0 < n) (hm : 0 < m) (same : n = m)
    (input : Nat → ℝ) :
    (Math.argmax n hn (fun i => input i.val)).val =
      (Math.argmax m hm (fun i => input i.val)).val := by
  subst m
  rfl

theorem global_tensor (x : Input) (o : Checked.Observation)
    (observed : o.input = x.tensor.metadata) (h : Checked.InputContract o) (branch : o.dim = none)
    (adapter : RealScratch.Adapter) (dtype : o.dtype.adapter = some adapter)
    (g : NativeGraph.GlobalPrograms adapter.format o)
    (a : GlobalAllocation x o adapter.format callers)
    (heap : Address → ℝ) (indices : Address → Nat) :
    ∃ result,globalInvoke x o adapter g a heap indices = .ok result ∧
      result.returned = a.output.returned ∧
      result.returned.metadata.shape = Host.outputShape x.tensor.metadata.shape none o.keepdim ∧
      result.returned.dtypeName = "torch.int64" ∧
      result.returned = a.output.allocated ∧ NativeSafety.Global x o a ∧
      (∃ positive : 0 < x.tensor.metadata.numel,
        result.state.indices (address result.returned.storage 8 0) = reference x heap positive) ∧
      (∀ caller,caller ∈ callers → ∀ point,caller.contains point →
        result.state.values point = ↑(heap point) ∧ result.state.indices point = indices point) := by
  have hg : Checked.GlobalContract o := by simpa only [branch] using h.2
  have coverage := Checked.global_coverage o hg (fun pid => (g.first pid).tree) g.second.tree
    (fun pid _ => (g.first pid).covers) g.second.covers
  let state := RealScratch.run adapter g.config a.buffers (logical x heap) (initial heap indices)
  have positive : 0 < x.tensor.metadata.numel := observed ▸ hg.2.2.1
  refine ⟨⟨a.output.returned,state⟩,by simp [globalInvoke,(Checked.check_iff o).mpr h,state],
    rfl,a.outputShape,a.output.int64,a.directOutput,
    NativeSafety.global x o observed hg adapter dtype a,⟨positive,?_⟩,?_⟩
  · have run := RealScratch.gathered_index adapter g.config coverage a.buffers (logical x heap)
      (initial heap indices)
    change state.indices (address a.output.returned.storage 8 0) = _ at run
    rw [run]
    have unique := least_maximum_unique (fun i : Fin g.config.size => logical x heap i.val)
      ⟨Math.globalIndex g.config (logical x heap),Math.global_valid g.config coverage (logical x heap)⟩
      (Math.argmax g.config.size coverage.sizePositive (fun i => logical x heap i.val))
      (Math.global_mathematical g.config coverage (logical x heap))
      (Math.argmax_spec _ _ _)
    have equality := congrArg Fin.val unique
    exact equality.trans (argmax_size _ _ _ positive (congrArg _ observed) (logical x heap))
  · intro caller member point inside
    apply RealScratch.frame adapter g.config a.buffers (logical x heap) (initial heap indices) point
    · intro i hi
      apply point_ne _ _ _ adapter.format.positive
      have cap := a.midValue.capacity
      change a.midValue.width*a.midValue.metadata.shape.prod ≤ _ at cap
      rw [a.midWidth,a.midShape] at cap
      simp only [List.prod_cons,List.prod_nil,Nat.mul_one] at cap
      exact HostExecution.word_point_apart _ _ (a.freshMid caller member) _ i
        ((Nat.mul_le_mul_left _ (Nat.succ_le_iff.mpr hi)).trans cap) point inside
    · intro i hi
      apply point_ne _ _ 8 (by omega)
      have cap := a.midIndex.capacity
      change a.midIndex.width*a.midIndex.metadata.shape.prod ≤ _ at cap
      rw [a.indexWidth,a.indexShape] at cap
      simp only [List.prod_cons,List.prod_nil,Nat.mul_one] at cap
      exact HostExecution.word_point_apart _ _ (a.freshIndex caller member) 8 i
        ((Nat.mul_le_mul_left _ (Nat.succ_le_iff.mpr hi)).trans cap) point inside
    · exact hg.2.2.2.2.2.2.2.1
    · apply point_ne _ _ 8 (by omega)
      have cap := a.output.returned.capacity
      change a.output.returned.width*a.output.returned.metadata.shape.prod ≤ _ at cap
      rw [a.output.width,a.outputShape,RequestBinding.global_shape_product] at cap
      exact HostExecution.word_point_apart _ _ (a.freshOutput caller member) 8 0
        (by simpa using cap) point inside

def dimensionInvoke (mem : Memory.Primitives M) (before : M) (x : Input)
    (o : Checked.Observation) (ax : Nat) (g : NativeGraph.DimensionPrograms)
    (call : ContiguousCall mem before x callers)
    (a : DimensionAllocation x o ax call.result callers) (heap : Address → ℝ) :
    Except String (TensorBinding.Result M) :=
  if Checked.check o then .ok ⟨a.output.returned,
    RealMemory.dimensionRun mem (g.config o ax) (logical x heap) a.output.returned.storage call.state⟩
  else .error "input contract rejected"

theorem dimension_tensor (mem : Memory.Primitives M) (before : M)
    (x : Input) (o : Checked.Observation) (observed : o.input = x.tensor.metadata)
    (h : Checked.InputContract o) (dim : Int) (branch : o.dim = some dim)
    (ax : Nat) (normalized : Host.normalizeDim o.input.shape.length dim = some ax)
    (g : NativeGraph.DimensionPrograms) (call : ContiguousCall mem before x callers)
    (prepared : o.prepared = call.result.tensor.metadata)
    (a : DimensionAllocation x o ax call.result callers) (heap : Address → ℝ) :
    ∃ result,dimensionInvoke mem before x o ax g call a heap = .ok result ∧
      result.returned = a.output.returned ∧
      result.returned.metadata.shape = Host.outputShape x.tensor.metadata.shape (some ax) o.keepdim ∧
      result.returned.dtypeName = "torch.int64" ∧
      (o.keepdim = Bool.true → result.returned = a.output.allocated) ∧
      NativeSafety.Dimension o ax call.result result.returned ∧
      (∃ nonempty : 0 < Checked.extent o ax,
        ∀ row k,row < Checked.outer o ax → k < Checked.inner o ax →
          mem.read 8 result.state (address result.returned.storage 8 (row*Checked.inner o ax+k)) =
            BitVec.ofNat 64 (Math.argmax (Checked.extent o ax) nonempty
              (fun i => logical x heap (dimInputOffset (g.config o ax) row i.val k))).val) ∧
      (∀ caller,caller ∈ callers → ∀ point,caller.contains point →
        mem.byte result.state point = mem.byte before point) := by
  have hd : Checked.DimensionContract o ax := by simpa only [branch,normalized] using h.2
  have valid : ax < x.tensor.metadata.shape.length := by
    rw [←observed]
    exact Checked.normalized_valid _ _ _ normalized
  let setup := a.setup mem before x o observed ax valid g call
  refine ⟨⟨a.output.returned,RealMemory.dimensionRun mem (g.config o ax) (logical x heap)
      a.output.returned.storage call.state⟩,
    by simp [dimensionInvoke,(Checked.check_iff o).mpr h],rfl,a.shape,a.output.int64,a.keptOutput,
    NativeSafety.dimension x o observed h.1 ax valid hd call prepared a,⟨hd.1,?_⟩,?_⟩
  · intro row k hr hk
    exact RealMemory.dimension_returned mem (g.config o ax) (logical x heap)
      a.output.returned.storage call.state row k hr hk hd.1 (fun tile => (g.tileProgram row k tile).covers)
  · intro caller member point inside
    rw [RealMemory.dimension_frame]
    · exact HostExecution.packed_frame mem before x.host call.packed callers call.fresh caller member point inside
    · intro row k hr hk
      have bound := Native.dim_output_bound _ _ _ _ hr hk
      exact HostExecution.word_point_apart _ _ (a.fresh caller member) 8 _
        ((Nat.mul_le_mul_left _ (Nat.succ_le_iff.mpr bound)).trans setup.layout.outputCapacity) point inside

#axiomsClean global_tensor
#axiomsClean dimension_tensor
#specNonCircular reference avoiding [globalInvoke,RealScratch.run,RealMemory.dimensionRun]
end
end VeriTile.Bench.Repairs.TritonArgmax.Correct
