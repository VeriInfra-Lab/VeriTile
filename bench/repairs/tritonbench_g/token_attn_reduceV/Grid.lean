import bench.repairs.tritonbench_g.token_attn_reduceV.Contract
namespace VeriTile.Bench.Repairs.TokenAttnReduceV.Grid
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false
abbrev Memory (α : Type) := RegionName → Nat → Cell α
abbrev PID := Nat × Nat

def state {α : Type} [Inhabited α] (m : Memory α) (pid : PID) : State α where
  mem := m
  regs := fun _ _ _ => none
  pids := fun a => if a=0 then pid.1 else if a=1 then pid.2 else 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

def launch {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (pid : PID) (m : Memory α) : Option (Memory α) :=
  (FP.Structural.exec (withSum M R) (kernel p) (state m pid)).map State.mem

def run {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) : List PID → Memory α → Option (Memory α)
  | [],m => some m
  | pid::rest,m => (launch M R p pid m).bind (run M R p rest)

def graph {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) : List PID → Memory α → Memory α
  | [],m => m
  | pid::rest,m => graph M R p rest (reference M R p (state m pid)).mem

theorem launch_complete {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (pid : PID) (m : Memory α) :
    launch M R p pid m=some (reference M R p (state m pid)).mem := by
  obtain ⟨t,run,mem⟩ := complete_run M R p (state m pid)
  simp [launch,run,mem]

theorem complete {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (work : List PID) (m : Memory α) :
    run M R p work m=some (graph M R p work m) := by
  induction work generalizing m with
  | nil => rfl
  | cons pid rest ih => simp only [run,launch_complete,Option.bind_some,ih,graph]

def Reads (p : Parameters) (r : RegionName) :=
  r=p.prob ∨ r=p.value ∨ r=Region.cast p.tokens ∨ r=Region.cast p.request ∨
    r=Region.cast p.start ∨ r=Region.cast p.length

def Separate (p : Parameters) := ∀ r, Reads p r → r≠p.output

def Agree {α : Type} (p : Parameters) (a b : Memory α) := ∀ r, Reads p r → a r=b r

theorem values_congr {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (a b : Memory α) (pid : PID) (h : Agree p a b) :
    outputValues M R p (state a pid)=outputValues M R p (state b pid) := by
  have hp := h p.prob (by simp [Reads])
  have hv := h p.value (by simp [Reads])
  have ht := h (Region.cast p.tokens) (by simp [Reads])
  have hr := h (Region.cast p.request) (by simp [Reads])
  have hs := h (Region.cast p.start) (by simp [Reads])
  have hl := h (Region.cast p.length) (by simp [Reads])
  have hproducts : ∀ n, products M p (state a pid) n=products M p (state b pid) n := by
    intro n
    funext i
    simp only [products,probability,value,gather,probOffset,gatherOffset,
      valueOffset,count,startIndex,requestIndex,state,hp,hv,ht,hr,hs,hl]
  have hu : update M R p (state a pid)=update M R p (state b pid) := by
    funext n acc i
    simp only [update,hproducts]
  have hc : count p (state a pid)=count p (state b pid) := by simp only [count,state,hl]
  funext i
  unfold outputValues accumulated
  rw [hu,hc]

theorem block_agrees {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (pid : PID) (m : Memory α) (separate : Separate p) :
    Agree p (reference M R p (state m pid)).mem m := by
  intro r hr
  funext o
  exact reference_frame M R p (state m pid) r o (Or.inl (separate r hr))

def offset (p : Parameters) (pid : PID) (i : TileIndex [p.width]) :=
  pid.1*p.outputBatchStride+pid.2*p.outputHeadStride+i.1.val*p.outputDimStride

def scatterBlock {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (base : Memory α) (pid : PID) (m : Memory α) : Memory α :=
  (TileShape.allIndices [p.width]).foldl
    (fun m i r o => if r=p.output ∧ o=offset p pid i then
      .mk .fp32 (outputValues M R p (state base pid) i) else m r o) m

theorem block_snapshot {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (pid : PID) (base m : Memory α) (h : Agree p m base) :
    (reference M R p (state m pid)).mem=scatterBlock M R p base pid m := by
  rw [reference,fold_write_memory,values_congr M R p m base pid h]
  simp only [outputOffset,state,↓reduceIte,show (1:Nat)≠0 by decide,offset,scatterBlock]

def snapshot {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (base : Memory α) : List PID → Memory α → Memory α
  | [],m => m
  | pid::rest,m => snapshot M R p base rest (scatterBlock M R p base pid m)

theorem graph_snapshot {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (work : List PID) (base m : Memory α) (separate : Separate p)
    (agree : Agree p m base) : graph M R p work m=snapshot M R p base work m := by
  induction work generalizing m with
  | nil => rfl
  | cons pid rest ih =>
    have next : Agree p (reference M R p (state m pid)).mem base := by
      intro r hr
      exact (block_agrees M R p pid m separate r hr).trans (agree r hr)
    rw [graph,ih _ next,block_snapshot M R p pid base m agree]
    rfl

#axiomsClean complete
#axiomsClean values_congr
#axiomsClean graph_snapshot
end
end VeriTile.Bench.Repairs.TokenAttnReduceV.Grid
