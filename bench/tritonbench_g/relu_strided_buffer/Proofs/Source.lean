import bench.tritonbench_g.relu_strided_buffer.Kernels
import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_StructuralTile

/-! Source execution bridge. The readable DSL retains signed pointer-register
values; the established signed tile uses natural cell-index registers. Those
private registers differ, while every typed memory cell and undef observation
agree. Both conversions take place after the full signed address sum. -/
namespace VeriTile.Bench.Repairs.ReluStridedBuffer
open VeriTile.Triton FP.Structural
open TritonBenchG.ReluStridedBuffer
open StructuralTile SignedView SourceSignedBranch
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

def parameters (a : HostPlan.Arguments) : Parameters :=
  ⟨⟨a.input.region,a.input.base,a.input.stride⟩,
   ⟨a.output.region,a.output.base,a.output.stride⟩,a.n,a.B,a.T,a.oneTile⟩

def views (c : Parameters) : SignedViews :=
  ⟨⟨c.input.region,c.input.base,c.input.stride⟩,
   ⟨c.output.region,c.output.base,c.output.stride⟩,c.n,c.B⟩

/-- Extracted from the actual one-tile source, not a second handwritten AST. -/
def perTile (c : Parameters) : List ComputeStmt := (oneTileKernel c).surfaceBody.drop 3

def beforeStore [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (c : Parameters) (T : Nat) (s : State α) : State α :=
  let s := s.setReg "tile_id0" .nat [] (fun _ => T)
  let s := s.setReg "offset0" .nat [] (fun _ => T * c.B)
  let s := s.setReg "in0_bptr" .int [c.B]
    (fun i => c.input.base + ((T * c.B + i.1.val : Nat) : Int) * c.input.stride)
  let s := s.setReg "in0" .fp32 [c.B] (inputValues M (views c) T s)
  let s := s.setReg "out0" .fp32 [c.B] (selectedValues M cmp (views c) T s)
  s.setReg "out0_bptr" .int [c.B]
    (fun i => c.output.base + ((T * c.B + i.1.val : Nat) : Int) * c.output.stride)

def sourceFinal [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (c : Parameters) (T : Nat) (s : State α) : State α :=
  (TileShape.allIndices [c.B]).foldl (fun current i =>
    if T * c.B + i.1.val < c.n then
      current.write c.output.region ((views c).outputView.address (T * c.B + i.1.val))
        (.mk .fp32 (scalarResult M cmp (rawValues (views c) T s i)))
    else current) (beforeStore M cmp c T s)

theorem perTile_exact [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (hc : M.compareLt none .fp32 = some cmp) (c : Parameters) (T : Nat) (s : State α)
    (ht : s.regs .nat [] "tile_id" = some (fun _ => T)) :
    run M (perTile c) s = some (sourceFinal M cmp c T s) := by
  simp [perTile,oneTileKernel,ComputeKernel.surfaceBody,run,step,stepAlg,evalExpr,
    evalOp_unfold,numeric,numericLt,natLt,FP.Structural.bop,store,hc,ht,
    sourceFinal,beforeStore,views,rawValues,StructuralTile.inputValues,StructuralTile.selectedValues,
    FloatDType.comparableDType,
    scalarResult,scalarZero,View.address,View.signedAddress,Region.cast,ofFloat,toFloat]
  norm_num only
  rfl

theorem sourceFinal_mem [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (c : Parameters) (T : Nat) (s : State α) :
    (sourceFinal M cmp c T s).mem = tileOutput M cmp (views c) T s.mem := by
  unfold sourceFinal tileOutput
  rw [scatter_mem]
  change _ = List.foldl _ s.mem _
  apply List.foldl_ext
  intro mem i _
  by_cases hi : T * c.B + i.1.val < c.n <;> simp [hi,rawValues,views]

theorem tile_implementation [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (hc : M.compareLt none .fp32 = some cmp) (c : Parameters) :
    StructuralControl.TileImplementation M (tileOutput M cmp (views c)) (perTile c) := by
  intro T s ht
  have control := scatter_control c.output.region
    (fun i : TileIndex [c.B] => (views c).outputView.address (T * c.B + i.1.val))
    (fun i => Cell.mk .fp32 (scalarResult M cmp (rawValues (views c) T s i)))
    (fun i => T * c.B + i.1.val < c.n) (TileShape.allIndices [c.B]) (beforeStore M cmp c T s)
  refine ⟨sourceFinal M cmp c T s,perTile_exact M cmp hc c T s ht,
    ⟨sourceFinal_mem M cmp c T s,control.2.1⟩,?_,?_⟩
  · rw [show (sourceFinal M cmp c T s).regs = (beforeStore M cmp c T s).regs from control.1]
    simp [beforeStore]
  · rw [show (sourceFinal M cmp c T s).regs = (beforeStore M cmp c T s).regs from control.1]
    simp [beforeStore]

#axiomsClean perTile_exact
#axiomsClean sourceFinal_mem
#axiomsClean tile_implementation
end
end VeriTile.Bench.Repairs.ReluStridedBuffer
