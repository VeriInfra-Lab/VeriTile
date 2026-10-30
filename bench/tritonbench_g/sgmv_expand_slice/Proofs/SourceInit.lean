import bench.tritonbench_g.sgmv_expand_slice.Proofs.SourceSetup
import bench.tritonbench_g.sgmv_expand_slice.Proofs.SourceBody

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceInit
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false

def coordinates (p : KernelArgs) (start adapter length pid : Nat) : SourceBody.Coordinates :=
  ⟨start,adapter,
    fun i => SourceSetup.rowIndex (pid / tiles p.columns) i % length,
    fun i => SourceSetup.columnIndex (pid % tiles p.columns) i % p.columns⟩

def zero {α : Type} (M : Algebra α) : Values α .real [32,32] :=
  fun _ => M.literal (some .fp32) .real 0

def guardedState {α : Type} [Inhabited α] (p : KernelArgs) (s : State α) (adapter : Nat) :=
  (SourcePrelude.prepared p s).setReg "lora_index" .int [] (fun _ => (adapter : Int))

def state {α : Type} [Inhabited α] (M : Algebra α) (p : KernelArgs) (s : State α)
    (start adapter length : Nat) : State α :=
  SourceSetup.prepared M start length p.columns (s.pids 0 / tiles p.columns)
    (s.pids 0 % tiles p.columns) (guardedState p s adapter)

theorem ready {α : Type} [Inhabited α] (M : Algebra α) (p : KernelArgs) (s : State α)
    (start adapter length : Nat) :
    SourceBody.Ready p (coordinates p start adapter length (s.pids 0))
      (state M p s start adapter length) (state M p s start adapter length) (zero M) := by
  constructor
  all_goals first | rfl | (intro d sh name hn; rfl) |
    simp [state,guardedState,SourceSetup.prepared,SourcePrelude.prepared,
      SourcePrelude.scalarBindings,coordinates,zero,State.setReg]

theorem prefix_run {α : Type} [Inhabited α] (M : Algebra α) (abi : SourceABI)
    (p : KernelArgs) (s : State α) (start adapter length : Nat)
    (hstart : (s.mem p.starts p.batch).read .int = (start : Int))
    (hlength : (s.mem p.lengths p.batch).read .int = (length : Int)) :
    run M (SourceParts.activePrelude abi p) (guardedState p s adapter) =
      some (state M p s start adapter length) := by
  apply SourceSetup.run_setup
  all_goals simp [guardedState,SourcePrelude.prepared,SourcePrelude.scalarBindings,
    State.setReg,hstart,hlength]

#axiomsClean ready
#axiomsClean prefix_run
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceInit
