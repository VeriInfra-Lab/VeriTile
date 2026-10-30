import bench.tritonbench_g.log_softmax.Proofs.KernelMath
import bench.tritonbench_g.log_softmax.Proofs.Rewrite
import bench.tritonbench_g.log_softmax.Proofs.Grid
import bench.tritonbench_g.apply_penalty.Proofs.Scatter

namespace VeriTile.Bench.Repairs.LogSoftmax.KernelGrid
open VeriTile.Triton FP.Structural
open Address
open VeriTile.Bench.Repairs.ApplyPenalty
noncomputable section
open Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

abbrev Flat (A : Type) := RegionName × Nat → Cell A
def flatten (s : State A) : Flat A := fun q => s.mem q.1 q.2
def positioned (s : State A) (p k : Nat) : State A :=
  {s with pids := fun ax => if ax=0 then p else if ax=1 then k else 0}

def program (d : Direction) (candidate : Bool) (M N K BM C : Nat) : ComputeKernel :=
  match d,candidate with
  | .forward,Bool.false => Source.forward "output" "input" M N K BM C
  | .forward,Bool.true => Source.forwardCommuted "output" "input" M N K BM C
  | .backward,Bool.false => Source.backward "input" "gradient" "output" M N K BM C
  | .backward,Bool.true => Source.backwardCommuted "input" "gradient" "output" M N K BM C

def coord (M N K BM C : Nat) (p : Nat × Nat) (i : TileIndex [BM,C]) : Option (Coord M N K) :=
  if h : Kernel.active M N K BM p.1 p.2 i then
    some (⟨p.1*BM+i.1.val,h.1⟩,⟨i.2.1.val,h.2.1⟩,⟨p.2,h.2.2⟩)
  else none

def events (M N K BM C : Nat) (p : Nat × Nat) : List (Coord M N K) :=
  (TileShape.allIndices [BM,C]).filterMap (coord M N K BM C p)

def key (q : Coord M N K) : RegionName × Nat := ("output",linear q)

def result (d : Direction) (e : Kernel.Engine A C) (xs gs : Nat → A) (q : Coord M N K) : A :=
  let x := fun j => if j<N then xs (offset N K q.1.val j q.2.2.val) else e.ops.negativeInfinity
  let g := fun j => if j<N then gs (offset N K q.1.val j q.2.2.val) else e.ops.zero
  match d with
  | .forward =>
    let m := e.maximum (fun j => x j.val)
    let delta := fun j => e.ops.subtract (x j) m
    e.ops.subtract (delta q.2.1.val) (e.ops.log (e.summation (fun j => e.ops.exp (delta j.val))))
  | .backward =>
    e.ops.subtract (g q.2.1.val)
      (e.ops.multiply (e.ops.exp (x q.2.1.val)) (e.summation (fun j => g j.val)))

def cell [Inhabited A] (d : Direction) (e : Kernel.Engine A C) (s : State A) (q : Coord M N K) : Cell A :=
  .mk .real (result d e (fun i => (s.mem "input" i).read .real)
    (fun i => (s.mem "gradient" i).read .real) q)

def sourceValue [Inhabited A] (d : Direction) (e : Kernel.Engine A C)
    (M N K BM : Nat) (s : State A) : TileIndex [BM,C] → A :=
  match d with
  | .forward => Kernel.forwardValue e "input" M N K BM s
  | .backward => Kernel.backwardValue e "input" "gradient" M N K BM s

theorem source_value [Inhabited A] (d : Direction) (e : Kernel.Engine A C)
    (M N K BM : Nat) (s : State A) (p : Nat × Nat) (i : TileIndex [BM,C])
    (h : Kernel.active M N K BM p.1 p.2 i) :
    Cell.mk .real (sourceValue d e M N K BM (positioned s p.1 p.2) i) =
      cell d e s (⟨p.1*BM+i.1.val,h.1⟩,⟨i.2.1.val,h.2.1⟩,⟨p.2,h.2.2⟩) := by
  have hp0 : (positioned s p.1 p.2).pids 0=p.1 := by simp [positioned]
  have hp1 : (positioned s p.1 p.2).pids 1=p.2 := by simp [positioned]
  have loads (fallback : A) (X : RegionName) :
      (fun j : Fin C => Kernel.loaded fallback X M N K BM C (positioned s p.1 p.2) (i.1,j,PUnit.unit)) =
      (fun j : Fin C => if j.val<N then
        (s.mem X ((p.1*BM+i.1.val)*N*K+j.val*K+p.2)).read .real else fallback) := by
    funext j
    rw [KernelMath.loaded_row _ _ _ _ _ _ _ _ _ (by simpa only [hp0] using h.1)
      (by simpa only [hp1] using h.2.2) j]
    rfl
  have xi := congrFun (loads e.ops.negativeInfinity "input") i.2.1
  have gi := congrFun (loads e.ops.zero "gradient") i.2.1
  rcases i with ⟨a,b,u⟩
  cases u
  cases d <;> simp only [sourceValue,Kernel.forwardValue,Kernel.backwardValue,
    loads,xi,gi,cell,result,offset,h.2.1,if_true]
  simp only [show ∀ j : Fin C,
    Kernel.loaded e.ops.negativeInfinity "input" M N K BM C (positioned s p.1 p.2) (a,j,PUnit.unit) =
      (if j.val<N then (s.mem "input" ((p.1*BM+a.val)*N*K+j.val*K+p.2)).read .real
        else e.ops.negativeInfinity) from fun j => congrFun (loads e.ops.negativeInfinity "input") j]

theorem address_active (M N K BM p k : Nat) (i : TileIndex [BM,C])
    (h : Kernel.active M N K BM p k i) :
    Kernel.address M N K BM p k i = offset N K (p*BM+i.1.val) i.2.1.val k := by
  simp only [Kernel.address,offset,Nat.min_eq_left (show p*BM+i.1.val≤M-1 by exact Nat.le_pred_of_lt h.1),
    Nat.min_eq_left (show i.2.1.val≤N-1 by exact Nat.le_pred_of_lt h.2.1),
    Nat.min_eq_left (show k≤K-1 by exact Nat.le_pred_of_lt h.2.2)]

theorem flat_write (s : State A) (r : RegionName) (i : Nat) (v : Cell A) :
    flatten (s.write r i v)=Scatter.write (flatten s) (r,i) v := by
  funext q
  simp [flatten,State.write,Scatter.write,Prod.ext_iff]

theorem scatter_bridge [Inhabited A] (d : Direction) (e : Kernel.Engine A C)
    (M N K BM : Nat) (s t : State A) (p : Nat × Nat) (lanes : List (TileIndex [BM,C])) :
    flatten (lanes.foldl (fun acc i => if Kernel.active M N K BM p.1 p.2 i then
      acc.write "output" (Kernel.address M N K BM p.1 p.2 i)
        (.mk .real (sourceValue d e M N K BM (positioned s p.1 p.2) i)) else acc) t) =
      Scatter.run (lanes.filterMap (coord M N K BM C p)) key (cell d e s) (flatten t) := by
  induction lanes generalizing t with
  | nil => rfl
  | cons i rest ih =>
    by_cases h : Kernel.active M N K BM p.1 p.2 i
    · simp only [List.foldl_cons,if_pos h,List.filterMap_cons,coord,dif_pos h]
      rw [ih,flat_write,source_value _ _ _ _ _ _ _ _ _ h,address_active _ _ _ _ _ _ _ h]
      rfl
    · simp only [List.foldl_cons,if_neg h,List.filterMap_cons,coord,dif_neg h]
      exact ih t

theorem program_execution [Inhabited A] (d : Direction) (e : Kernel.Engine A C)
    (M N K BM : Nat) (hc : 0<C) (s : State A) (p : Nat × Nat) :
    ∃ t,FP.Structural.exec e.algebra (program d Bool.false M N K BM C)
        (positioned s p.1 p.2)=some t ∧
      flatten t=Scatter.run (events M N K BM C p) key (cell d e s) (flatten s) := by
  have raw : ∃ t,FP.Structural.exec e.algebra (program d Bool.false M N K BM C)
      (positioned s p.1 p.2)=some t ∧
      t.mem=(Kernel.stores "output" M N K BM C (positioned s p.1 p.2)
        (sourceValue d e M N K BM (positioned s p.1 p.2))).mem := by
    cases d
    · exact Kernel.forward_exec e "output" "input" M N K BM _ hc
    · exact Kernel.backward_exec e "input" "gradient" "output" M N K BM _
  obtain ⟨t,ran,memory⟩ := raw
  refine ⟨t,ran,?_⟩
  change (fun q : RegionName × Nat => t.mem q.1 q.2)=_
  rw [memory]
  simpa only [Kernel.stores,positioned,if_true,show (1 : Nat)≠0 by decide,if_false,
    events,flatten] using scatter_bridge d e M N K BM s (positioned s p.1 p.2) p
      (TileShape.allIndices [BM,C])

def run [Inhabited A] (d : Direction) (e : Kernel.Engine A C) (candidate : Bool)
    (M N K BM : Nat) : List (Nat × Nat) → State A → Option (State A)
  | [],s => some s
  | p::rest,s => do
    let t ← FP.Structural.exec e.algebra (program d candidate M N K BM C) (positioned s p.1 p.2)
    run d e candidate M N K BM rest t

theorem candidate_same_run [Inhabited A] (d : Direction) (e : Kernel.Engine A C)
    (M N K BM : Nat) (order : List (Nat × Nat)) (s : State A) :
    run d e Bool.false M N K BM order s=run d e Bool.true M N K BM order s := by
  induction order generalizing s with
  | nil => rfl
  | cons p rest ih =>
    have same : FP.Structural.exec e.algebra (program d Bool.false M N K BM C) (positioned s p.1 p.2)=
        FP.Structural.exec e.algebra (program d Bool.true M N K BM C) (positioned s p.1 p.2) := by
      cases d
      · exact Rewrite.forward_same_run e "output" "input" M N K BM _
      · exact Rewrite.backward_same_run e "input" "gradient" "output" M N K BM _
    simp only [run,same]
    cases FP.Structural.exec e.algebra (program d Bool.true M N K BM C) (positioned s p.1 p.2)
    · rfl
    · exact ih _

theorem chain [Inhabited A] (d : Direction) (e : Kernel.Engine A C)
    (M N K BM : Nat) (hc : 0<C) (order : List (Nat × Nat)) (s : State A) :
    ∃ t,run d e Bool.false M N K BM order s=some t ∧
      flatten t=Scatter.run (order.flatMap (events M N K BM C)) key (cell d e s) (flatten s) := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons p rest ih =>
    obtain ⟨mid,hm,hmem⟩ := program_execution d e M N K BM hc s p
    obtain ⟨t,ht,htmem⟩ := ih mid
    have inputs (r : RegionName) (hr : r≠"output") (i : Nat) : mid.mem r i=s.mem r i := by
      change flatten mid (r,i)=flatten s (r,i)
      rw [hmem]
      exact Scatter.frame _ _ _ _ _ (fun q _ same => hr (congrArg Prod.fst same))
    have input : (fun i => (mid.mem "input" i).read .real)=(fun i => (s.mem "input" i).read .real) :=
      funext (fun i => congrArg (Cell.read .real) (inputs "input" (by decide) i))
    have grad : (fun i => (mid.mem "gradient" i).read .real)=(fun i => (s.mem "gradient" i).read .real) :=
      funext (fun i => congrArg (Cell.read .real) (inputs "gradient" (by decide) i))
    have values : cell (M:=M) (N:=N) (K:=K) d e mid=cell d e s := by
      funext q
      simp only [cell,input,grad]
    refine ⟨t,by simp only [run,hm]; exact ht,?_⟩
    rw [htmem,values,hmem]
    simp only [List.flatMap_cons,Scatter.run,List.foldl_append]

def programs (M N K BM : Nat) : List (Nat × Nat) :=
  if M*N*K=0 then [] else
    (List.range (Grid.groups M BM)).flatMap fun p => (List.range K).map fun k => (p,k)

theorem coverage (q : Coord M N K) (hb : 0<BM) (hc : N≤C) :
    q∈(programs M N K BM).flatMap (events M N K BM C) := by
  let p : Nat × Nat := (q.1.val/BM,q.2.2.val)
  let i : TileIndex [BM,C] := (⟨q.1.val%BM,Nat.mod_lt _ hb⟩,
    ⟨q.2.1.val,lt_of_lt_of_le q.2.1.isLt hc⟩,PUnit.unit)
  have row : q.1.val/BM*BM+q.1.val%BM=q.1.val := by
    simpa only [Nat.mul_comm] using Nat.div_add_mod q.1.val BM
  apply List.mem_flatMap.mpr
  refine ⟨p,?_,?_⟩
  · have nonempty : M*N*K≠0 := by
      have := linear_bound q
      omega
    simp only [programs,if_neg nonempty]
    apply List.mem_flatMap.mpr
    exact ⟨p.1,List.mem_range.mpr (Grid.group_cover q.1.val q.1.isLt hb),
      List.mem_map.mpr ⟨p.2,List.mem_range.mpr q.2.2.isLt,rfl⟩⟩
  · apply List.mem_filterMap.mpr
    refine ⟨i,TileShape.mem_allIndices _ i,?_⟩
    have active : Kernel.active M N K BM p.1 p.2 i := by
      exact ⟨by simpa only [p,i,row] using q.1.isLt,q.2.1.isLt,q.2.2.isLt⟩
    simp only [coord,dif_pos active,p,i,row]

theorem key_injective : Function.Injective (@key M N K) := by
  intro a b same
  exact linear_injective (congrArg Prod.snd same)

/-- Whole source grid, with constructive success, every tensor cell and full
typed-memory frame. Empty dimensions have no active writes. -/
theorem whole [Inhabited A] (d : Direction) (e : Kernel.Engine A C)
    (M N K BM : Nat) (hb : 0<BM) (hc : 0<C) (covers : N≤C) (s : State A) :
    ∃ t,run d e Bool.false M N K BM (programs M N K BM) s=some t ∧
      (∀ q : Coord M N K,t.mem "output" (linear q)=cell d e s q) ∧
      (∀ (r : RegionName) i,r≠"output" ∨ M*N*K≤ i → t.mem r i=s.mem r i) := by
  obtain ⟨t,ran,memory⟩ := chain d e M N K BM hc (programs M N K BM) s
  refine ⟨t,ran,?_,?_⟩
  · intro q
    change flatten t (key q)=_
    rw [memory]
    exact Scatter.readback_injective _ key (cell d e s) (flatten s) q (coverage q hb covers) key_injective
  · intro r i outside
    change flatten t (r,i)=flatten s (r,i)
    rw [memory]
    apply Scatter.frame
    intro q _ same
    have rr := congrArg Prod.fst same
    have ri := congrArg Prod.snd same
    rcases outside with outside | outside
    · exact outside rr
    · have bound := linear_bound q
      simp only [key] at ri
      omega

/-- The executed forward graph is linked to the retained independent
mathematical proof, without importing a host-return theorem. -/
theorem forward_graph (ops : Ops A) (plan : Plan N C) (xs gs : Nat → A)
    (q : Coord M N K) :
    result .forward (KernelMath.forwardEngine ops plan) xs gs q =
      forward ops (KernelMath.uniform plan)
        (fun j => xs (offset N K q.1.val j q.2.2.val)) q.2.1.val := by
  simp [result,KernelMath.forwardEngine,forward,cast32,rowMax,KernelMath.uniform,
    denominator,exponentials,shifted,forwardLoad,q.2.1.isLt]
  rfl

theorem backward_graph (ops : Ops A) (plan : Plan N C) (ys gs : Nat → A)
    (q : Coord M N K) :
    result .backward (KernelMath.backwardEngine ops plan) ys gs q =
      backward ops (KernelMath.uniform plan)
        (fun j => ys (offset N K q.1.val j q.2.2.val))
        (fun j => gs (offset N K q.1.val j q.2.2.val)) q.2.1.val := by
  simp [result,KernelMath.backwardEngine,backward,cast32,scale,KernelMath.uniform,
    gradientLoad,q.2.1.isLt]
  rfl

#axiomsClean program_execution
#axiomsClean candidate_same_run
#axiomsClean whole
end
end VeriTile.Bench.Repairs.LogSoftmax.KernelGrid
