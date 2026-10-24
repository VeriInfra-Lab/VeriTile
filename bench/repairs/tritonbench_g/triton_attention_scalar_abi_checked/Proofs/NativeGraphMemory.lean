import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.NativeScalar
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.GridFrame
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native.Selected
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

instance : Inhabited Graph := ⟨.word 0⟩
def embed (mem : Grid.Memory Graph) : Grid.Memory Expr :=
  fun r o=>mapCell (fun g _=>Stage.ready g) (mem r o)

def finish (one : Bool) (kind : Kind) (c : Shape) (owners : TritonAttention.Whole001.Native.Owners)
    (mem : Grid.Memory Expr) : Grid.Memory Graph :=
  fun r o=>mapCell (fun e=>lowerScalar one kind (e (storeOwner c owners r o))) (mem r o)

theorem finish_embed (one : Bool) (kind : Kind) (c : Shape) (owners : TritonAttention.Whole001.Native.Owners)
    (mem : Grid.Memory Graph) : finish one kind c owners (embed mem)=mem := by
  funext r o
  simp only [finish,embed]
  cases mem r o with
  | mk d x=>cases d <;> rfl

/-- Compiler interpretation of one shared DSL CTA. Every store selects its
actual scalar replica and finishes the local instruction graph before another
CTA or stage can load that cell. Graphs denote native words through Graph.eval. -/
def execute (one : Bool) (kind : Kind) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program) (mem : Grid.Memory Graph) : Option (Grid.Memory Graph) :=
  (Grid.execute (algebraScalar one kind) (modelScalar one kind) (kernelFor c scale kind) (gridFor c kind) pid (embed mem)).map
    (finish one kind c owners)

def store (one : Bool) (kind : Kind) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program) (mem : Grid.Memory Graph) : Grid.Memory Graph :=
  finish one kind c owners (match kind with
    | .forward=>Grid.forwardStep (modelScalar one kind) c scale pid (embed mem)
    | .preprocess=>Grid.preprocessStep (modelScalar one kind) c pid (embed mem)
    | .backward=>Grid.backwardStep (modelScalar one kind) c scale pid (embed mem))

theorem execute_success (one : Bool) (kind : Kind) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program) (mem : Grid.Memory Graph) :
    execute one kind c scale owners pid mem=some (store one kind c scale owners pid mem) := by
  cases kind <;> simp only [execute,store,kernelFor,gridFor,Grid.forward_success,Grid.preprocess_success,Grid.backward_success,Option.map_some]

def run (one : Bool) (c : Shape) (scale : ℝ) (owners : TritonAttention.Whole001.Native.Owners)
    (fw pw bw : List Grid.Program) (mem : Grid.Memory Graph) : Option (Grid.Memory Graph) := do
  let f←Grid.run (execute one .forward c scale owners) fw mem
  let p←Grid.run (execute one .preprocess c scale owners) pw f
  Grid.run (execute one .backward c scale owners) bw p

def stores (one : Bool) (c : Shape) (scale : ℝ) (owners : TritonAttention.Whole001.Native.Owners)
    (fw pw bw : List Grid.Program) (mem : Grid.Memory Graph) : Grid.Memory Graph :=
  let f:=fw.foldl (fun m pid=>store one .forward c scale owners pid m) mem
  let p:=pw.foldl (fun m pid=>store one .preprocess c scale owners pid m) f
  bw.foldl (fun m pid=>store one .backward c scale owners pid m) p

theorem run_success (one : Bool) (c : Shape) (scale : ℝ) (owners : TritonAttention.Whole001.Native.Owners)
    (fw pw bw : List Grid.Program) (mem : Grid.Memory Graph) :
    run one c scale owners fw pw bw mem=some (stores one c scale owners fw pw bw mem) := by
  simp only [run,Grid.run_eq_fold _ _ (execute_success one .forward c scale owners),
    Grid.run_eq_fold _ _ (execute_success one .preprocess c scale owners),
    Grid.run_eq_fold _ _ (execute_success one .backward c scale owners),Option.bind_some,stores]
  rfl

def words (mem : Grid.Memory Word) : Grid.Memory Graph := fun r o=>mapCell Graph.word (mem r o)
def evaluate (I : Prior.Instructions) (literal : ℝ → Word) (mem : Grid.Memory Graph) : Grid.Memory Word :=
  fun r o=>mapCell (Graph.eval I literal) (mem r o)
def nativeRun (one : Bool) (c : Shape) (scale : ℝ) (owners : TritonAttention.Whole001.Native.Owners)
    (I : Prior.Instructions) (literal : ℝ → Word) (fw pw bw : List Grid.Program) (mem : Grid.Memory Word) : Option (Grid.Memory Word) :=
  (run one c scale owners fw pw bw (words mem)).map (evaluate I literal)

theorem evaluate_words (I : Prior.Instructions) (literal : ℝ → Word) (mem : Grid.Memory Word) :
    evaluate I literal (words mem)=mem := by
  funext r o
  simp only [evaluate,words]
  cases mem r o with
  | mk d x=>cases d <;> rfl

theorem store_frame (one : Bool) (kind : Kind) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program) (mem : Grid.Memory Graph)
    (hp : Grid.Legal (gridFor c kind) pid) (r : RegionName) (o : Nat) (hr : ¬Grid.Written c r o) :
    store one kind c scale owners pid mem r o=mem r o := by
  have back:=congrFun (congrFun (finish_embed one kind c owners mem) r) o
  cases kind
  · change mapCell _ (Grid.forwardStep _ c scale pid (embed mem) r o)=_
    rw [Grid.forwardStep_frame _ c scale pid _ hp r o hr]
    exact back
  · change mapCell _ (Grid.preprocessStep _ c pid (embed mem) r o)=_
    rw [Grid.preprocessStep_frame _ c pid _ hp r o hr]
    exact back
  · change mapCell _ (Grid.backwardStep _ c scale pid (embed mem) r o)=_
    rw [Grid.backwardStep_frame _ c scale pid _ hp r o hr]
    exact back

theorem stores_frame (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (fw pw bw : List Grid.Program) (mem : Grid.Memory Graph)
    (hf : ∀ pid∈fw,Grid.Legal (Grid.forwardGrid c) pid)
    (hp : ∀ pid∈pw,Grid.Legal (Grid.preprocessGrid c) pid)
    (hb : ∀ pid∈bw,Grid.Legal (Grid.backwardGrid c) pid)
    (r : RegionName) (o : Nat) (hr : ¬Grid.Written c r o) :
    stores one c scale owners fw pw bw mem r o=mem r o := by
  unfold stores
  rw [Grid.fold_frame _ bw _ r o (by intro pid hpid m;exact store_frame one .backward c scale owners pid m (hb pid hpid) r o hr)]
  rw [Grid.fold_frame _ pw _ r o (by intro pid hpid m;exact store_frame one .preprocess c scale owners pid m (hp pid hpid) r o hr)]
  exact Grid.fold_frame _ fw mem r o (by intro pid hpid m;exact store_frame one .forward c scale owners pid m (hf pid hpid) r o hr)

#axiomsClean execute_success
#axiomsClean run_success
#axiomsClean stores_frame
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native.Selected
