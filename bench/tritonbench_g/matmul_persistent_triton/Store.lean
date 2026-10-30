import bench.tritonbench_g.matmul_persistent_triton.Suffix
import bench.tritonbench_g.matmul_persistent_triton.Geometry
import bench.tritonbench_g.matmul_persistent_triton.Compose

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution (Outside Post)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_regs masked_readback_active)
noncomputable section
set_option maxHeartbeats 7000000
set_option maxRecDepth 28000
variable {α : Type} [Inhabited α]

def referenceValue (P : Source.Arithmetic α α) (p : Parameters) (b : Buffers)
    (entry : Memory α) (x : LogicalIndex p) : Cell α :=
  pack p.problem.profile (Targets.result P p.problem b entry x.1.val x.2.val)

theorem inputs_outside (p : Parameters) (b : Buffers) (initial current : State α)
    (entry : Memory α) (inputs : Inputs p b initial entry) (outside : Outside "C" initial current) :
    Inputs p b current entry := by
  constructor
  · intro i; rw [outside "A" i (by decide)]; exact inputs.a i
  · intro i; rw [outside "B" i (by decide)]; exact inputs.b i

theorem written_post (P : Source.Arithmetic α α) (p : Parameters) (b : Buffers)
    (initial : State α) (entry : Memory α) (tile : Nat)
    (layout : Layout p.problem b) (strides : Strides p b) (inputs : Inputs p b initial entry)
    (zero : P.cast p.problem.profile P.accumulatorZero=P.inputZero) :
    Post "C" (logicalAddress p) (referenceValue P p b entry) (Owns p tile) initial
      (written P p initial tile) := by
  constructor
  · intro x hx
    obtain ⟨i,hr,hc⟩ := hx
    rw [← owned_address p tile x i hr hc]
    rw [written,masked_readback_active "C" (cOffset p tile) (outputValue P p initial tile)
      (active p tile) initial i (owned_active p tile x i hr hc)
      (lane_injective p b layout strides tile)]
    simp only [outputValue,value_reference P p b initial entry strides inputs zero,
      referenceValue,hr,hc]
  · intro r o miss
    apply State.masked_scatter_frame
    rcases miss with region|offset
    · exact .inl region
    · right
      intro i hi active
      rw [address_index p tile i active]
      exact offset (index p tile i active) (lane_owns p tile i active)

theorem suffix_post (P : Source.Arithmetic α α) (p : Parameters) (b : Buffers)
    (initial current : State α) (entry : Memory α) (tile : Nat)
    (layout : Layout p.problem b) (strides : Strides p b) (inputs : Inputs p b initial entry)
    (zero : P.cast p.problem.profile P.accumulatorZero=P.inputZero)
    (inv : InnerInvariant P p initial tile p.problem.kTiles current) :
    Post "C" (logicalAddress p) (referenceValue P p b entry) (Owns p tile) initial
      (suffixResult P p initial current tile) := by
  have post := written_post P p b initial entry tile layout strides inputs zero
  simpa only [Post,suffixMemory P p initial current tile inv] using post

theorem suffix_registers (P : Source.Arithmetic α α) (p : Parameters)
    (initial current : State α) (pid tile : Nat) (regs : Registers p pid current) :
    Registers p pid (suffixResult P p initial current tile) := by
  constructor <;>
    simp [suffixResult,masked_scatter_regs,storeRegisters,regs.start,regs.rows,regs.columns,
      regs.count,regs.work,regs.group,regs.span,regs.lanesM,regs.lanesN,regs.lanesK]

#axiomsClean written_post
#axiomsClean suffix_post
#axiomsClean suffix_registers

end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
