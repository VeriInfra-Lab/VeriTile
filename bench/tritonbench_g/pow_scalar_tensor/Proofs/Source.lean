import bench.tritonbench_g.pow_scalar_tensor.Kernels
import bench.tritonbench_g.pow_scalar_tensor.Proofs.Mathematical_WholeTile

/-! Execute the readable signed source using the established opaque-power tile
semantics. Pointer temporaries keep signed offsets until the actual access.
Only those private registers differ from the historical cell-index lowering. -/
namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.PowScalarTensor WholeTile
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

def parameters (base : ℝ) (a : HostPlan.Arguments) : Parameters :=
  ⟨⟨a.input.region,a.input.base,a.input.stride⟩,
   ⟨a.output.region,a.output.base,a.output.stride⟩,a.n,base,a.B,a.T,a.oneTile⟩

def config (c : Parameters) (candidate : Bool) : WholeSource.Config :=
  ⟨⟨c.input.region,c.input.base,c.input.stride⟩,
   ⟨c.output.region,c.output.base,c.output.stride⟩,c.n,c.B,c.base,candidate⟩

def sourceKernel (c : Parameters) (candidate : Bool) : ComputeKernel :=
  if candidate then commutedKernel c else kernel c

/-- Tile extraction from the canonical source, not a second handwritten AST. -/
def perTile (c : Parameters) (candidate : Bool) : List ComputeStmt :=
  (if candidate then oneTileCommuted c else oneTileKernel c).surfaceBody.drop 3

def beforeStore [Inhabited α] (M : Algebra α) (c : Parameters) (candidate : Bool)
    (T : Nat) (s : State α) : State α :=
  let s := s.setReg "tile_id0" .nat [] (fun _ => T)
  let s := s.setReg "offset0" .nat [] (fun _ => if candidate then c.B * T else T * c.B)
  let s := s.setReg "in0_bptr" .int [c.B]
    (fun i => c.input.base + ((T * c.B + i.1.val : Nat) : Int) * c.input.stride)
  let s := s.setReg "in0" .real [c.B] (inputValues M (config c candidate) T s)
  let s := s.setReg "out0" .real [c.B] (selectedValues M (config c candidate) T s)
  s.setReg "out0_bptr" .int [c.B]
    (fun i => c.output.base + ((T * c.B + i.1.val : Nat) : Int) * c.output.stride)

def sourceFinal [Inhabited α] (M : Algebra α) (c : Parameters) (candidate : Bool)
    (T : Nat) (s : State α) : State α :=
  (TileShape.allIndices [c.B]).foldl (fun current i =>
    if T * c.B + i.1.val < c.n then
      current.write c.output.region ((config c candidate).outputView.address (T * c.B + i.1.val))
        (.mk .fp32 (scalarResult M c.base (rawValues (config c candidate) T s i)))
    else current) (beforeStore M c candidate T s)

theorem perTile_exact [Inhabited α] (M : Algebra α) (c : Parameters) (candidate : Bool)
    (T : Nat) (s : State α) (ht : s.regs .nat [] "tile_id" = some (fun _ => T)) :
    run M (perTile c candidate) s = some (sourceFinal M c candidate T s) := by
  cases candidate <;>
    simp [perTile,oneTileKernel,oneTileCommuted,ComputeKernel.surfaceBody,run,step,stepAlg,
      evalExpr,evalComputeOp,evalOp_unfold,numeric,natLt,FP.Structural.bop,store,ht,
      sourceFinal,beforeStore,config,rawValues,WholeTile.inputValues,WholeTile.selectedValues,
      scalarResult,scalarBase,SignedView.View.address,SignedView.View.signedAddress,
      Region.cast,ofFloat,toFloat,Nat.mul_comm]
  all_goals try rfl
  have hm : (FP.Structural.bop (fun x y : Nat => x * y) Broadcast.nil
      (fun _ => c.B) (fun _ => T)) = (fun _ => T * c.B) := by
    funext i
    exact Nat.mul_comm c.B T
  rw [hm]
  rfl

theorem sourceFinal_mem [Inhabited α] (M : Algebra α) (c : Parameters) (candidate : Bool)
    (T : Nat) (s : State α) :
    (sourceFinal M c candidate T s).mem = tileOutput M (config c candidate) T s.mem := by
  unfold sourceFinal tileOutput
  rw [scatter_mem]
  change _ = List.foldl _ s.mem _
  apply List.foldl_ext
  intro mem i _
  by_cases hi : T * c.B + i.1.val < c.n <;> simp [hi,rawValues,config]

theorem tile_implementation [Inhabited α] (M : Algebra α) (c : Parameters) (candidate : Bool) :
    StructuralControl.TileImplementation M (tileOutput M (config c candidate)) (perTile c candidate) := by
  intro T s ht
  have control := scatter_control c.output.region
    (fun i : TileIndex [c.B] => (config c candidate).outputView.address (T * c.B + i.1.val))
    (fun i => Cell.mk .fp32 (scalarResult M c.base (rawValues (config c candidate) T s i)))
    (fun i => T * c.B + i.1.val < c.n) (TileShape.allIndices [c.B]) (beforeStore M c candidate T s)
  refine ⟨sourceFinal M c candidate T s,perTile_exact M c candidate T s ht,
    ⟨sourceFinal_mem M c candidate T s,control.2.1⟩,?_,?_⟩
  · rw [show (sourceFinal M c candidate T s).regs = (beforeStore M c candidate T s).regs from control.1]
    simp [beforeStore]
  · rw [show (sourceFinal M c candidate T s).regs = (beforeStore M c candidate T s).regs from control.1]
    simp [beforeStore]

#axiomsClean perTile_exact
#axiomsClean sourceFinal_mem
#axiomsClean tile_implementation
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked
