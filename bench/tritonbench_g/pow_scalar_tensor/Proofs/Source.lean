import bench.tritonbench_g.pow_scalar_tensor.Proofs.PastedTile
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
set_option linter.unusedSimpArgs false
set_option maxRecDepth 16000

def parameters (base : ℝ) (a : HostPlan.Arguments) : Parameters :=
  ⟨⟨a.input.region,a.input.base,a.input.stride⟩,
   ⟨a.output.region,a.output.base,a.output.stride⟩,a.n,base,a.B,a.T,a.oneTile⟩

def config (c : Parameters) (candidate : Bool) : WholeSource.Config :=
  ⟨⟨c.input.region,c.input.base,c.input.stride⟩,
   ⟨c.output.region,c.output.base,c.output.stride⟩,c.n,c.B,c.base,candidate⟩

def sourceKernel (c : Parameters) (candidate : Bool) : ComputeKernel :=
  if candidate then commutedKernel c else kernel c

/-- Extracted from the two complete source branches, including helper calls. -/
def perTile (c : Parameters) (candidate : Bool) : List ComputeStmt := PastedTile.body c candidate

def sourceFinal [Inhabited α] (M : Algebra α) (c : Parameters) (candidate : Bool)
    (T : Nat) (s : State α) (ht : s.regs .nat [] "tile_id"=some (fun _ => T)) : State α :=
  (PastedTile.execution M c candidate T s ht).val

theorem perTile_exact [Inhabited α] (M : Algebra α) (c : Parameters) (candidate : Bool)
    (T : Nat) (s : State α) (ht : s.regs .nat [] "tile_id"=some (fun _ => T)) :
    run M (perTile c candidate) s = some (sourceFinal M c candidate T s ht) :=
  (PastedTile.execution M c candidate T s ht).property

theorem sourceFinal_mem [Inhabited α] (M : Algebra α) (c : Parameters) (candidate : Bool)
    (T : Nat) (s : State α) (ht : s.regs .nat [] "tile_id"=some (fun _ => T)) :
    (sourceFinal M c candidate T s ht).mem = tileOutput M (config c candidate) T s.mem := by
  rcases c with ⟨input,output,n,base,lanes,tiles,branch⟩
  cases candidate <;> cases branch
  all_goals
    simp only [sourceFinal,PastedTile.execution]
    simp only [tileOutput,scatter_mem]
    simp [config,rawValues,scalarResult,scalarBase,
      SignedView.View.address,SignedView.View.signedAddress,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex,
      ← Int.natCast_add,← Int.natCast_mul,-Nat.cast_add,-Nat.cast_mul]
    apply List.foldl_ext
    intro mem i _
    by_cases active : T*lanes+i.1.val<n <;> simp [active,Nat.mul_comm,Int.mul_comm] <;> norm_num

theorem tile_implementation [Inhabited α] (M : Algebra α) (c : Parameters) (candidate : Bool) :
    StructuralControl.TileImplementation M (tileOutput M (config c candidate)) (perTile c candidate) := by
  intro T s ht
  refine ⟨sourceFinal M c candidate T s ht,perTile_exact M c candidate T s ht,
    ⟨sourceFinal_mem M c candidate T s ht,?_⟩,?_,?_⟩
  all_goals
    rcases c with ⟨input,output,n,base,lanes,tiles,branch⟩
    cases candidate <;> cases branch
    all_goals
      simp only [sourceFinal,PastedTile.execution]
      simp [PastedTile.masked_regs,PastedTile.masked_undef,StructuralControl.flatState]

#axiomsClean perTile_exact
#axiomsClean sourceFinal_mem
#axiomsClean tile_implementation
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked
