import bench.tritonbench_g.token_softmax_llama.Proofs.RowContract
namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Grid
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_mem)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false
abbrev Memory (α : Type) := RegionName→Nat→Cell α
abbrev PID := Nat×Nat
def state {α : Type} [Inhabited α] (m : Memory α) (pid : PID) : State α where
  mem := m
  regs := fun _ _ _ => none
  pids := fun a => if a=0 then pid.1 else if a=1 then pid.2 else 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

def launch {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (pid : PID) (m : Memory α) : Option (Memory α) :=
  (FP.Structural.exec M (kernel .fp32 p) (state m pid)).map State.mem
def run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) :
    List PID→Memory α→Option (Memory α)
  | [],m => some m
  | pid::rest,m => (launch M p pid m).bind (run M p rest)
def graph {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) :
    List PID→Memory α→Memory α
  | [],m => m
  | pid::rest,m => graph M p rest (reference M .fp32 p (state m pid)).mem

theorem launch_complete {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (positive : 0 < p.width) (pid : PID) (m : Memory α) :
    launch M p pid m=some (reference M .fp32 p (state m pid)).mem := by
  rw [launch,(execution M .fp32 p (state m pid) positive).property,Option.map_some,
    execution_memory]
theorem complete {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (positive : 0 < p.width) (work : List PID) (m : Memory α) :
    run M p work m=some (graph M p work m) := by
  induction work generalizing m with
  | nil => rfl
  | cons pid rest ih => simp only [run,launch_complete M p positive,Option.bind_some,ih,graph]

def Reads (p : Parameters) (r : RegionName) := r=p.input ∨ r=Region.cast p.start ∨ r=Region.cast p.length
def Separate (p : Parameters) := ∀ r,Reads p r → r≠p.output
def Agree {α : Type} (p : Parameters) (a b : Memory α) := ∀ r,Reads p r → a r=b r

theorem count_congr {α : Type} [Inhabited α] (p : Parameters) (a b : Memory α)
    (pid : PID) (h : Agree p a b) : count p (state a pid)=count p (state b pid) := by
  simp only [count,state,h (Region.cast p.length) (by simp [Reads])]
theorem start_congr {α : Type} [Inhabited α] (p : Parameters) (a b : Memory α)
    (pid : PID) (h : Agree p a b) : startIndex p (state a pid)=startIndex p (state b pid) := by
  simp only [startIndex,state,h (Region.cast p.start) (by simp [Reads])]
theorem values_congr {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (a b : Memory α) (pid : PID) (h : Agree p a b) :
    outputValues M .fp32 p (state a pid)=outputValues M .fp32 p (state b pid) := by
  have hr : inputRow M .fp32 p (state a pid)=inputRow M .fp32 p (state b pid) := by
    funext i
    simp only [inputRow]
    rw [count_congr p a b pid h,start_congr p a b pid h]
    simp only [state,h p.input (by simp [Reads])]
  simp only [outputValues,numerators,hr]

theorem block_agrees {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (pid : PID) (m : Memory α) (separate : Separate p) :
    Agree p (reference M .fp32 p (state m pid)).mem m := by
  intro r hr
  funext o
  exact reference_frame M .fp32 p (state m pid) r o (Or.inl (separate r hr))

def offset {α : Type} [Inhabited α] (p : Parameters) (base : Memory α) (pid : PID)
    (i : TileIndex [p.width]) := outputOffset p (state base pid) i
def active {α : Type} [Inhabited α] (p : Parameters) (base : Memory α) (pid : PID)
    (i : TileIndex [p.width]) := i.1.val < count p (state base pid)
instance {α : Type} [Inhabited α] (p : Parameters) (base : Memory α) (pid : PID)
    (i : TileIndex [p.width]) : Decidable (active p base pid i) := inferInstanceAs (Decidable (_ < _))
def scatterBlock {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (base : Memory α) (pid : PID) (m : Memory α) : Memory α :=
  (TileShape.allIndices [p.width]).foldl (fun m i => if active p base pid i then
    fun r o => if r=p.output ∧ o=offset p base pid i then
      .mk .fp32 (outputValues M .fp32 p (state base pid) i) else m r o else m) m

theorem block_snapshot {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (pid : PID) (base m : Memory α) (h : Agree p m base) :
    (reference M .fp32 p (state m pid)).mem=scatterBlock M p base pid m := by
  rw [reference,fold_writes_mem,values_congr M p m base pid h,count_congr p m base pid h]
  have ho : outputOffset p (state m pid)=outputOffset p (state base pid) := by
    funext i
    simp only [outputOffset]
    rw [start_congr p m base pid h]
    rfl
  rw [ho]
  rfl

def snapshot {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (base : Memory α) : List PID→Memory α→Memory α
  | [],m => m
  | pid::rest,m => snapshot M p base rest (scatterBlock M p base pid m)
theorem graph_snapshot {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (work : List PID) (base m : Memory α) (separate : Separate p) (agree : Agree p m base) :
    graph M p work m=snapshot M p base work m := by
  induction work generalizing m with
  | nil => rfl
  | cons pid rest ih =>
    have next : Agree p (reference M .fp32 p (state m pid)).mem base := by
      intro r hr
      exact (block_agrees M p pid m separate r hr).trans (agree r hr)
    rw [graph,ih _ next,block_snapshot M p pid base m agree]
    rfl
#axiomsClean complete
#axiomsClean graph_snapshot
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Grid
