import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.Recurrence
import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.FPSymbolic

/-! Explicit structural compiler boundary for the selected FP32 specialization.
The source side is the exact canonical kernel, split at its actual start_n
loop. A certificate supplies the compiler's selected fragments and pins their
operand/control/initializer/iteration/cast/store structure. These are graph
equalities before numerical evaluation, not execution or output equations.
The public FP execution below consumes those selected fragments. This module
does not claim a register invariant for generic FP AST interpretation, or that
an arbitrary external compiler has emitted this certificate. -/
namespace VeriTile.Bench.Repairs.ContextAttnMistral.SourceGraph
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open Recurrence
noncomputable section

abbrev Word := BitVec 32
abbrev Expr := NativePhysical.Expr

def program (c : Host.Call) : ComputeKernel :=
  Program.kernel c.layout c.regions .fp32 Bool.false (Host.scale c.layout.D)

def parts (c : Host.Call) : Source.IntRange :=
  FPSymbolic.parts c.layout c.regions .fp32 (Host.scale c.layout.D)

theorem source_split (c : Host.Call) :
    (program c).surfaceBody=(parts c).before ++
      [.forRangeInt "start_n" (parts c).start (parts c).stop (parts c).stride (parts c).body] ++
      (parts c).after := FPSymbolic.source_split _ _ _ _

/-- The source syntax is an index of a selected fragment. A compiler binding
must supply the fragment for this exact source segment, not a hash or label. -/
structure Fragment (code : List ComputeStmt) (α : Type) where
  selected : α

structure Control (start stop : Op .int []) (stride : Op .nat []) where
  count : Nat → Nat → Nat

def operands (input : Inputs Word) : Inputs Expr :=
  ⟨fun r h d=>.leaf .fp32 (input.query r h d),
   fun r h d=>.leaf .fp32 (input.key r h d),
   fun r h d=>.leaf .fp32 (input.value r h d)⟩

def finish (plan : NativePhysical.Plan) (c : Host.Call)
    (state : Accumulator Expr c.layout.D) (row : Nat) (d : Fin c.layout.D) : Expr :=
  (NativePhysical.backend plan).scalar.cast precision .real .real
    (state.value (⟨row%128,Nat.mod_lt _ (by decide)⟩,d,()))

/-- Structural source-to-selected-graph certificate. Each equality is local:
raw input leaves, the initializer, one source iteration, integer trip count,
same-format output cast, or output address. There is no whole-loop result,
successful execution, final word, target, or mathematical result premise.

The single-iteration map retains the complete source operand/control graph:
masked Q/K/V coordinates and GQA, QK zero+dot then scale, both finite masks,
max/denominator guards, exp, ordered alpha/beta normalization, P cast and PV
with the scaled live accumulator. NativePhysical performs only the documented
selected staging (guard transport, contractions and physical owner routes). -/
structure Binding (c : Host.Call) (plan : NativePhysical.Plan) where
  input : Inputs Word → Inputs Expr
  input_map : input=operands
  prelude : Fragment (parts c).before (Accumulator Expr c.layout.D)
  prelude_map : prelude.selected=initial (NativePhysical.backend plan) c.layout.D
  iteration : Fragment (parts c).body
    (Inputs Expr → Nat → Nat → Nat → Nat → Accumulator Expr c.layout.D → Accumulator Expr c.layout.D)
  iteration_map : iteration.selected=update (NativePhysical.backend plan) c
  control : Control (parts c).start (parts c).stop (parts c).stride
  control_map : control.count=tiles c
  epilogue : Fragment (parts c).after
    (Accumulator Expr c.layout.D → Nat → Fin c.layout.D → Expr)
  epilogue_map : epilogue.selected=finish plan c
  store : Fragment (parts c).after (Nat → Nat → Nat → Nat → Nat)
  store_map : store.selected=c.output

/-- Selected imperative execution consumes the bound initialization and
single-iteration graph, not the independent accumulated recurrence. -/
def run (binding : Binding c plan) (input : Inputs Expr) (b h block start : Nat)
    (state : Accumulator Expr c.layout.D) : Nat → Accumulator Expr c.layout.D
  | 0 => state
  | n+1 => run binding input b h block (start+1)
      (binding.iteration.selected input b h block start state) n

def value (binding : Binding c plan) (input : Inputs Word) (b h row : Nat)
    (d : Fin c.layout.D) : Expr :=
  binding.epilogue.selected
    (run binding (binding.input input) b h (row/128) 0 binding.prelude.selected
      (binding.control.count b (row/128))) row d

theorem run_loop (binding : Binding c plan) (input : Inputs Expr) (b h block start : Nat)
    (state : Accumulator Expr c.layout.D) (count : Nat) :
    run binding input b h block start state count=
      Recurrence.loop (NativePhysical.backend plan) c input b h block start state count := by
  induction count generalizing start state with
  | zero => rfl
  | succ n ih => rw [run,Recurrence.loop,binding.iteration_map,ih]

/-- The source binding is consumed before applying the separately proved loop
invariant. The conclusion compares syntax graphs, without evaluating words. -/
theorem value_expected (binding : Binding c plan) (input : Inputs Word)
    (b h row : Nat) (d : Fin c.layout.D) :
    value binding input b h row d=
      expected (NativePhysical.backend plan) c (operands input) b h row d := by
  unfold value
  rw [binding.input_map,binding.prelude_map,binding.control_map,binding.epilogue_map,run_loop]
  exact source_expected (NativePhysical.backend plan) c (operands input) b h row d

theorem store_address (binding : Binding c plan) (b h row d : Nat) :
    binding.store.selected b h row d=c.output b h row d := by rw [binding.store_map]

#axiomsClean source_split
#axiomsClean value_expected
#axiomsClean store_address
#specNonCircular Binding avoiding [FP.Structural.exec,Recurrence.expected,Recurrence.sourceValue,value]
end
end VeriTile.Bench.Repairs.ContextAttnMistral.SourceGraph
