import bench.tritonbench_g.triton_attention.Proofs.NativeStaging
import bench.tritonbench_g.triton_attention.Proofs.Grid
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000

def mapValue {α β : Type} (f : α → β) : (d : TileDType) → Value α d → Value β d
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => f
  | .nat | .int | .bool | .ptr | .blockPtr => id
def mapCell {α β : Type} (f : α → β) : Cell α → Cell β
  | .mk d x => .mk d (mapValue f d x)
def inject (mem : Grid.Memory Word) : Grid.Memory Expr :=
  fun r o=>mapCell (fun w _=>Stage.ready (.word w)) (mem r o)

def storeOwner (c : Shape) (owners : TritonAttention.Whole001.Native.Owners)
    (r : RegionName) (o : Nat) : Nat :=
  if r="L" then (owners.l (o/c.rows) (o%c.rows)).val
  else if r="M" then (owners.m (o/c.rows) (o%c.rows)).val
  else if r="Delta" then owners.delta (o/c.rows) (o%c.rows)%(c.dim/4)
  else (Reduction.owner (o%c.dim)).val

def project (kind : Kind) (c : Shape) (owners : TritonAttention.Whole001.Native.Owners)
    (I : Prior.Instructions) (literal : ℝ → Word) (mem : Grid.Memory Expr) : Grid.Memory Word :=
  fun r o=>mapCell (fun e=>(lower kind (e (storeOwner c owners r o))).eval I literal) (mem r o)

theorem project_inject (kind : Kind) (c : Shape) (owners : TritonAttention.Whole001.Native.Owners)
    (I : Prior.Instructions) (literal : ℝ → Word) (mem : Grid.Memory Word) :
    project kind c owners I literal (inject mem)=mem := by
  funext r o
  simp only [project,inject]
  cases mem r o with
  | mk d x=>cases d <;> rfl

def gridFor (c : Shape) : Kind → Grid.Program
  | .forward=>Grid.forwardGrid c
  | .preprocess=>Grid.preprocessGrid c
  | .backward=>Grid.backwardGrid c

def kernelFor (c : Shape) (scale : ℝ) : Kind → ComputeKernel
  | .forward=>forwardKernel c scale
  | .preprocess=>preprocessKernel c
  | .backward=>backwardKernel c scale

/-- Every CTA reads materialized words from the preceding CTA/stage. Only
its local expressions retain reduction replicas and delayed compiler nodes. -/
def execute (kind : Kind) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Prior.Instructions)
    (literal : ℝ → Word) (pid : Grid.Program) (mem : Grid.Memory Word) : Option (Grid.Memory Word) :=
  (Grid.execute (algebra kind) (model kind) (kernelFor c scale kind) (gridFor c kind) pid (inject mem)).map
    (project kind c owners I literal)

/-- The already-proved input-derived store recurrence, evaluated and projected
at each CTA boundary. The independent native reference is connected separately. -/
def storeStep (kind : Kind) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Prior.Instructions)
    (literal : ℝ → Word) (pid : Grid.Program) (mem : Grid.Memory Word) : Grid.Memory Word :=
  project kind c owners I literal (match kind with
    | .forward=>Grid.forwardStep (model kind) c scale pid (inject mem)
    | .preprocess=>Grid.preprocessStep (model kind) c pid (inject mem)
    | .backward=>Grid.backwardStep (model kind) c scale pid (inject mem))

theorem execute_success (kind : Kind) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Prior.Instructions)
    (literal : ℝ → Word) (pid : Grid.Program) (mem : Grid.Memory Word) :
    execute kind c scale owners I literal pid mem=some (storeStep kind c scale owners I literal pid mem) := by
  cases kind <;> simp only [execute,storeStep,kernelFor,gridFor,Grid.forward_success,
    Grid.preprocess_success,Grid.backward_success,Option.map_some]

def wholeRun (c : Shape) (scale : ℝ) (owners : TritonAttention.Whole001.Native.Owners)
    (I : Prior.Instructions) (literal : ℝ → Word) (fw pw bw : List Grid.Program)
    (mem : Grid.Memory Word) : Option (Grid.Memory Word) := do
  let f ← Grid.run (execute .forward c scale owners I literal) fw mem
  let p ← Grid.run (execute .preprocess c scale owners I literal) pw f
  Grid.run (execute .backward c scale owners I literal) bw p

def storeReference (c : Shape) (scale : ℝ) (owners : TritonAttention.Whole001.Native.Owners)
    (I : Prior.Instructions) (literal : ℝ → Word) (fw pw bw : List Grid.Program)
    (mem : Grid.Memory Word) : Grid.Memory Word :=
  let f:=fw.foldl (fun m p=>storeStep .forward c scale owners I literal p m) mem
  let p:=pw.foldl (fun m p=>storeStep .preprocess c scale owners I literal p m) f
  bw.foldl (fun m p=>storeStep .backward c scale owners I literal p m) p

theorem whole_success (c : Shape) (scale : ℝ) (owners : TritonAttention.Whole001.Native.Owners)
    (I : Prior.Instructions) (literal : ℝ → Word) (fw pw bw : List Grid.Program)
    (mem : Grid.Memory Word) :
    wholeRun c scale owners I literal fw pw bw mem=some (storeReference c scale owners I literal fw pw bw mem) := by
  simp only [wholeRun,Grid.run_eq_fold _ _ (execute_success .forward c scale owners I literal),
    Grid.run_eq_fold _ _ (execute_success .preprocess c scale owners I literal),
    Grid.run_eq_fold _ _ (execute_success .backward c scale owners I literal),Option.bind_some,storeReference]
  rfl

#axiomsClean project_inject
#axiomsClean execute_success
#axiomsClean whole_success
#specNonCircular storeReference avoiding [wholeRun,execute,FP.Structural.exec,Grid.run]
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
