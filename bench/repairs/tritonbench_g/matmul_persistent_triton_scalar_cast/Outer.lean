import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Setup
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Loop
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Store

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution (Post)
noncomputable section
set_option maxHeartbeats 7000000
set_option maxRecDepth 28000
variable {α : Type} [Inhabited α]

theorem outer_decomposition (p : Parameters) :
    outer p=tilePrelude p++(innerStatement p::tileSuffix p) := by
  rcases p with ⟨⟨M,N,K,SMS,profile⟩,am,ak,bk,bn,cm,cn⟩
  cases profile <;> rfl

theorem outerExecution (P : Source.Arithmetic α α) (p : Parameters) (b : Buffers)
    (initial : State α) (entry : Memory α) (pid ordinal : Nat)
    (layout : Layout p.problem b) (strides : Strides p b) (inputs : Inputs p b initial entry)
    (zero : P.cast p.problem.profile P.accumulatorZero=P.inputZero)
    (regs : Registers p pid initial) (tileBound : p.problem.tile pid ordinal<p.problem.tiles) :
    ∃ final,run (backend P) (outer p) (initial.setReg "tile_ordinal" .nat [] (fun _ => ordinal))=some final ∧
      Post "C" (logicalAddress p) (referenceValue P p b entry) (Owns p (p.problem.tile pid ordinal)) initial final ∧
      Registers p pid final := by
  let setup := tilePreludeExecution P p pid ordinal initial regs
  have invariant := tile_initial P p pid ordinal initial regs tileBound
  have registers := tile_registers P p pid ordinal initial regs
  obtain ⟨last,ran,inv,reg⟩ := innerLoop P p initial setup.val pid (p.problem.tile pid ordinal) 0
    (Nat.zero_le _) invariant registers
  refine ⟨suffixResult P p initial last (p.problem.tile pid ordinal),?_,
    suffix_post P p b initial last entry (p.problem.tile pid ordinal) layout strides inputs zero inv,
    suffix_registers P p initial last pid (p.problem.tile pid ordinal) reg⟩
  rw [outer_decomposition,run_append,setup.property,Option.bind_some,run,
    dynamicInner P p setup.val pid registers,ran]
  exact suffixExecution P p initial last _ inv

#axiomsClean outerExecution
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
