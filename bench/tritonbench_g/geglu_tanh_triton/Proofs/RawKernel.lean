import bench.tritonbench_g.geglu_tanh_triton.Proofs.NativeSource
import bench.tritonbench_g.geglu_tanh_triton.Proofs.Real
import bench.tritonbench_g.geglu_tanh_triton.Proofs.Rewrite

/-! Raw allocation/layout kernel interfaces and complete source/grid certificates. -/
namespace VeriTile.Bench.Repairs.GegluTanhTriton.RawKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Physical
noncomputable section
set_option maxHeartbeats 4000000

structure Plan (p : Layout) where
  order : List Nat
  schedule : order.Perm (List.range (Grid.grid p.count))

def Outside (backward : Bool) (p : Layout) (owner address : Nat) : Prop :=
  ∀ ch, enabled backward ch → ∀ i, i < p.count →
    ¬(p.key (outputRole ch) i).covers owner address

def Values {M α β : Type} [Inhabited α] (ops : Ops M α β) (formula : Channel → α → α → α → α)
    (backward : Bool) (p : Layout) (entry final : M) : Prop :=
  ∀ ch, enabled backward ch → ∀ i, i < p.count →
    ops.read final (p.key (outputRole ch) i) =
      formula ch (ops.read entry (p.key .a i)) (ops.read entry (p.key .b i))
        (if backward then ops.read entry (p.key .third i) else default)

def RunSpec {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (formula : Channel → α → α → α → α) (backward : Bool) (p : Layout) (h : Valid p)
    (entry final : M) : Prop :=
  Values ops formula backward p entry final ∧
  (∀ owner address, Outside backward p owner address → ops.observe final owner address = ops.observe entry owner address)

theorem run_spec {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute formula : Channel → α → α → α → α) (same : ∀ ch a b d, compute ch a b d = formula ch a b d)
    (backward : Bool) (p : Layout) (h : Valid p) (plan : Plan p) (s : M) :
    RunSpec ops formula backward p h s (Physical.run ops compute backward p plan.order s) := by
  constructor
  · intro ch on i hi
    have result := Physical.run_values ops compute backward p h
      plan.order (plan.schedule.nodup_iff.mpr List.nodup_range) s ch on i hi
      (plan.schedule.mem_iff.mpr (List.mem_range.mpr (Grid.covered p.count i hi).1))
    simpa only [value,same] using result
  · intro owner address outside
    apply Physical.run_bytes
    intro pid hp ch on lane hl
    exact outside ch on (pid*128+lane) ((lanes_mem _ _ _).mp hl).2

/-- Successful execution of each actual selected source in the same physical
state sequence. The transition is tied to all source output observations;
source success is a conclusion and cannot be supplied by a caller. -/
def Executes {M α β : Type} [Inhabited α] (ops : Ops M α β) (algebra : Algebra α)
    (candidate : Bool) (backward : Bool) (p : Layout) (h : Valid p) (plan : Plan p)
    (before final : M) : Prop :=
  final = Physical.run ops (Tile.scalar algebra) backward p plan.order before ∧
  ∀ prior pid suffix, plan.order = prior ++ pid :: suffix →
    let entry := Physical.run ops (Tile.scalar algebra) backward p prior before
    ∃ t, run algebra (selectedSource backward candidate 128 p.count).surfaceBody
        (Tile.rowState pid (Source.state ops backward p entry)) = some t ∧
      ∀ role offset, offset < (p.allocation role).words →
        t.mem (Source.region backward role) offset = Cell.mk .real
          (ops.read (Physical.tile ops (Tile.scalar algebra) backward p pid entry)
            (p.key role offset))

theorem executes {M α β : Type} [Inhabited α] (ops : Ops M α β) (algebra : Algebra α)
    (candidate : Bool) (backward : Bool) (p : Layout) (h : Valid p) (plan : Plan p) (before : M) :
    Executes ops algebra candidate backward p h plan before
      (Physical.run ops (Tile.scalar algebra) backward p plan.order before) := by
  refine ⟨rfl,?_⟩
  intro prior pid suffix split
  obtain ⟨t,ran,linked⟩ := Source.certificates ops algebra backward p h
    plan.order before prior pid suffix split
  exact ⟨t,by rw [Rewrite.selected_run]; exact ran,linked⟩

/-- The native graph uses the same canonical source and actual physical reads.
Instruction semantics (including ternary FMA and libdevice tanh) remain opaque
model primitives. No native-to-software conformance is a numerical assumption. -/
def NativeExecutes {M β : Type} (ops : Ops M Native.Word β) (instructions : Native.Instructions)
    (candidate : Bool) (backward : Bool) (p : Layout) (h : Valid p) (plan : Plan p)
    (before final : M) : Prop :=
  final = Physical.run ops (Native.sourceScalar (Native.compiledPorts instructions)) backward
    p plan.order before ∧
  ∀ prior pid suffix, plan.order = prior ++ pid :: suffix →
    let entry := Physical.run ops (Native.sourceScalar (Native.compiledPorts instructions)) backward
      p prior before
    ∃ t, run VeriTile.Bench.TritonBenchG.SwigluBackward.Observed.graphAlgebra
        (selectedSource backward candidate 128 p.count).surfaceBody
        (Tile.rowState pid (Native.graphState ops backward p entry)) = some t ∧
      ∀ role offset, offset < (p.allocation role).words →
        Native.interpret (Native.compiledPorts instructions)
          ((t.mem (Source.region backward role) offset).read .real) =
          ops.read (Physical.tile ops (Native.sourceScalar (Native.compiledPorts instructions))
            backward p pid entry) (p.key role offset)

theorem native_executes {M β : Type} (ops : Ops M Native.Word β)
    (instructions : Native.Instructions) (candidate : Bool)
    (backward : Bool) (p : Layout) (h : Valid p) (plan : Plan p) (before : M) :
    NativeExecutes ops instructions candidate backward p h plan before
      (Physical.run ops (Native.sourceScalar (Native.compiledPorts instructions)) backward
        p plan.order before) := by
  refine ⟨rfl,?_⟩
  intro prior pid suffix split
  obtain ⟨t,ran,linked⟩ := Native.certificates ops instructions backward p h
    plan.order before prior pid suffix split
  exact ⟨t,by rw [Rewrite.selected_run]; exact ran,linked⟩

#axiomsClean run_spec
#axiomsClean executes
#axiomsClean native_executes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.GegluTanhTriton.RawKernel
