import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.FPWorkset
import bench.tritonbench_g.bgmv_shrink_kernel.Contract
import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.TargetBridge
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Real
open VeriTile.Triton FP.Structural
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 3000000

/-- The ordinary Real erasure of scalar arithmetic. Unsupported operations
are irrelevant: the constructive full-source theorem uses only add and mul. -/
def algebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .add => a+b | .sub => a-b | .mul => a*b | .div => a/b | .max => max a b | .pow => Real.rpow a b
  unary := fun _ _ x => x
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0

def primitives : VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.Primitives ℝ where
  reduction.apply := fun _ dtype {shape} ax keep xs out => ofFloat dtype
    (∑ k : Fin (TileShape.axisDim shape ax), toFloat dtype
      (xs (match keep with
        | Bool.false => TileShape.insertAxisIndex shape ax out k
        | Bool.true => TileShape.replaceAxisIndex shape ax out k)))
  atomic := fun _ a b => a+b

def data (h : Call) (u : BlockState) (indices : Nat → Int) : Target.Data ℝ where
  input := u.readMem "X"
  weights := u.readMem "W"
  output := u.readMem "Y"
  indices := indices
  scaling := h.scaling
  other := fun _ _ => .mk .real 0
  padding := fun dtype _ _ => defaultValue dtype

def summand (h : Call) (u : BlockState) (batch slot split c : Nat) (n : Fin h.BN) : ℝ :=
  ∑ e : Fin 256, if n.val < h.N ∧ VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.rank split c e.val < h.K then
    u.readMem "X" (VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.inputOffset h batch (VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.rank split c e.val))*
      u.readMem "W" (VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.weightOffset h slot n.val (VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.rank split c e.val)) else 0

theorem sums (h : Call) (u : BlockState) (indices : Nat → Int) (b slot split c : Nat) (n : Fin h.BN) :
    VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.sums algebra primitives h (data h u indices) b slot split c n=
      summand h u b slot split c n := by
  rcases h with ⟨batches,adapters,N,K,BN,columns,positive,covered,outputCovered,storage,outputType,scaling⟩
  cases storage <;>
    simp only [VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.sums,
      VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.products,
      VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.loaded,
      VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.weight,
      VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.dtype,
      primitives,algebra,data,ofFloat,toFloat,summand,TileShape.axisDim,TileShape.insertAxisIndex,id_eq]
  all_goals
    apply Finset.sum_congr rfl
    intro e _
    by_cases hn : n.val < N <;> by_cases hk : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.rank split c e.val < K <;> simp [hn,hk] <;> norm_num

theorem accumulator (h : Call) (u : BlockState) (indices : Nat → Int) (b slot split c : Nat) (n : Fin h.BN) :
    VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.accumulator algebra primitives h (data h u indices) b slot split c n=
      ∑ j ∈ Finset.range c,summand h u b slot split j n := by
  induction c with
  | zero => simp [VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.accumulator,algebra]
  | succ c ih =>
    change VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.accumulator algebra primitives h (data h u indices) b slot split c n+
      VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.sums algebra primitives h (data h u indices) b slot split c n=_
    rw [ih,sums,Finset.sum_range_succ]

theorem payload (h : Call) (u : BlockState) (indices : Nat → Int) (b slot split : Nat) (n : Fin h.N) :
    VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.payload algebra primitives h (data h u indices)
      b slot split ⟨n.val,n.isLt.trans_le h.rowsCovered⟩=VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.RealExecution.contribution h u b slot n.val split := by
  change VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.accumulator algebra primitives h (data h u indices)
    b slot split h.cycles ⟨n.val,n.isLt.trans_le h.rowsCovered⟩*h.scaling=_
  rw [accumulator]
  simp only [summand,n.isLt,true_and,VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.RealExecution.contribution]
  ring

abbrev Run (h : Call) (orders : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Orders h)
    (u : BlockState) (indices : Nat → Int) := FPWorkset.Run algebra primitives h orders (data h u indices)

theorem readback (h : Call) (orders : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Orders h)
    (u : BlockState) (indices : Nat → Int) (run : Run h orders u indices) (b : Fin h.batches) (n : Fin h.N) :
    run.output b n=Target.real h u indices b n := by
  rw [FPWorkset.readback]
  unfold VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.target Target.real
  by_cases skip : indices b.val=-1
  · simp [data,skip,VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.outputOffset]
  · have dataIndices : (data h u indices).indices=indices := rfl
    simp only [dataIndices,skip,if_false,payload]
    exact VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.AtomicOrder.contribution_readback h orders u b n (indices b.val).toNat

#axiomsClean payload
#axiomsClean readback
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Real
