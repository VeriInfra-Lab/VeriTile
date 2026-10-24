import bench.repairs.tritonbench_g.triton_matmul.Proofs.KernelExecution
import bench.repairs.tritonbench_g.triton_matmul.Proofs.ByteHost
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelNative
open VeriTile.Triton FP.Structural
open FPValue (Storage DotAcc)
open Contract TensorBinding InputAdapter KernelContract OutputBytes
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 2000000

/-- Source block padding, fixed K16/K32 MMA groups, live FP32 carry and the
original output encoding, computed only from original logical inputs. -/
def expected (I : Native.Instructions) (read : Read) (a b : Tensor)
    (la : Laws (storage a.metadata) a) (lb : Laws (storage b.metadata) b) (i : OutputIndex a b) : Word :=
  logicalReference (Native.numerics I (storage a.metadata)) (request a b)
    (CheckedHost.left read id a b la) (CheckedHost.right read id a b lb) i.1 i.2

theorem correct (rest : Algebra Word) (I : Native.Instructions) (read : Read) (a b c : Tensor)
    (la : Laws (storage a.metadata) a) (lb : Laws (storage b.metadata) b)
    (h : InputContract a.metadata b.metadata) (initial : State Word) (output : Output a b c)
    (bound : TensorBinding.Inputs read id a b initial)
    (outputBound : OutputBytes.Bound read (storage a.metadata) c initial) :
    ∃ actual typed bytes,
      FiniteGrid.grid (Native.algebra rest I) (Native.dotAcc I) (request a b)
        (List.range (programs (request a b))) initial=some actual ∧
      runPrograms (Native.algebra rest I) (Native.dotAcc I) (request a b)
        (List.range (programs (request a b))) initial=some typed ∧
      actual.mem=typed.mem ∧
      ByteExecution.grid (Native.algebra rest I) (Native.dotAcc I) (request a b) c
        (List.range (programs (request a b))) (initial,read)=some (actual,bytes) ∧
      CheckedHost.Post a b c initial actual (expected I read a b la lb) ∧
      ByteHost.Post read bytes a b c output.tensor (expected I read a b la lb) := by
  have input : KernelContract.Inputs (readWord read) a b initial := ⟨bound.left,bound.right⟩
  obtain ⟨actual,typed,ha,ht,hm,result⟩ := KernelSource.executes (Native.algebra rest I) (Native.dotAcc I)
    (Native.numerics I (storage a.metadata)) (readWord read) a b c h initial output input
    (Native.primitives rest I _)
  have logical : CheckedHost.Post a b c initial actual (expected I read a b la lb) := by
    apply KernelSource.post (Native.numerics I (storage a.metadata)) a b c initial actual output result
    intro i
    have logical := CheckedHost.source_inputs read id a b la lb h initial bound
    exact reference_logical _ (request a b) initial _ _ logical.1 logical.2 i.1 i.2
  refine ⟨actual,typed,render read (storage a.metadata) c actual,ha,ht,hm,?_,logical,?_⟩
  · by_cases empty : Contract.Empty a.metadata b.metadata
    · have np := KernelSource.empty_programs a b empty
      have eq : initial=actual := by simpa only [np,List.range_zero,FiniteGrid.grid,Option.some.injEq] using ha
      subst actual
      simp only [np,List.range_zero,ByteExecution.grid,render_initial read _ c initial outputBound]
    · let tensor := KernelSource.tensorAPI (readWord read) a b c h empty initial output input
      have launch : programs (request a b)<2^31 := (execution_nonempty _ _ h empty).1.2.2.2
      have base : tensor.baseC=c.metadata.dataPtr.toNat := by
        change InputAdapter.base c=c.metadata.dataPtr.toNat
        have size := output_numel a b c output.metadata
        have nz : c.metadata.numel≠0 := by
          unfold Contract.Empty at empty
          rw [size]
          exact Nat.mul_ne_zero (by omega) (by omega)
        simp only [InputAdapter.base,if_neg nz]
      have seq := ByteExecution.grid_render (Native.algebra rest I) (Native.dotAcc I) (request a b) c
        initial tensor launch rfl (output_numel a b c output.metadata).symm base read
        (List.range (programs (request a b))) (fun _ hp => List.mem_range.mp hp) initial
      have storage_eq : (request a b).storage=storage a.metadata := rfl
      rw [storage_eq,render_initial read _ c initial outputBound,ha,Option.map_some] at seq
      exact seq
  · exact ByteHost.post read read a b c output.tensor initial actual _ logical
      (ByteHost.expected_packed I a b _ _) (fun _ _ => rfl)

#specNonCircular expected avoiding [Source.typed,FP.Structural.exec,runPrograms,FiniteGrid.grid,ByteExecution.grid]
#axiomsClean correct
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelNative
