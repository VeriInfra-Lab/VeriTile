import bench.tritonbench_g.dequantize_matmul.Proofs.Events

namespace VeriTile.Bench.Repairs.DequantizeMatmul
open CrossEntropy1
noncomputable section
set_option maxHeartbeats 4000000

def pointTile : Tile := ⟨1,1,by decide,by decide⟩
def unitRect (w : Window) : Rect := ⟨w,1,1⟩

theorem Window.decode_inside (w : Window) (p : Address) (h : w.inside p) :
    w.row p<w.rows ∧ w.col p<w.columns ∧ w.addr (w.row p) (w.col p)=p := by
  have positive : 0<w.columns := by
    by_contra hz
    have : w.columns=0 := by omega
    simp only [Window.inside,this,Nat.mul_zero,Nat.add_zero] at h
    omega
  have lower := h.2.1
  have offsetBound : p.2-w.offset<w.rows*w.columns := by
    have := h.2.2
    omega
  refine ⟨(Nat.div_lt_iff_lt_mul positive).mpr offsetBound,Nat.mod_lt _ positive,?_⟩
  apply Prod.ext
  · exact h.1.symm
  · have split : (p.2-w.offset)/w.columns*w.columns+(p.2-w.offset)%w.columns=p.2-w.offset := by
      simpa only [Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div (p.2-w.offset) w.columns
    simp only [addr,row,col]
    omega

theorem unit_writes (w : Window) (q : Task) (hq : q.1<w.rows ∧ q.2<w.columns) (p : Address) :
    (unitRect w).writes q p ↔ p=w.addr q.1 q.2 := by
  constructor
  · intro h
    have dec := w.decode_inside p h.1
    have coord : (w.row p,w.col p)=q := by simpa [Rect.owner,unitRect] using h.2
    have hrow := congrArg Prod.fst coord
    have hcol := congrArg Prod.snd coord
    simp only [Prod.fst,Prod.snd] at hrow hcol
    simpa only [hrow,hcol] using dec.2.2.symm
  · intro h
    subst p
    refine ⟨w.address_inside _ _ hq.1 hq.2,?_⟩
    have coord := w.coordinates q.1 q.2 hq.2
    simpa [Rect.owner,unitRect,coord.1,coord.2]

theorem unit_step (a : Arithmetic α) (w : Window)
    (value : Memory (Cell α) → Nat → Nat → Cell α)
    (typed : ∀ m r c,∃ x,value m r c=.fp32 x)
    (q : Task) (hq : q.1<w.rows ∧ q.2<w.columns) (m : Memory (Cell α)) :
    (unitRect w).step value q m=
      writeWord m (w.addr q.1 q.2) ((value m q.1 q.2).float a.zero) := by
  funext p
  simp only [Rect.step,Schedule.put,writeWord]
  simp only [unit_writes w q hq p]
  by_cases h : p=w.addr q.1 q.2
  · rw [if_pos h,if_pos h,h]
    have coord := w.coordinates q.1 q.2 hq.2
    obtain ⟨x,hx⟩ := typed m q.1 q.2
    simp only [unitRect,coord.1,coord.2,hx,Cell.float]
  · rw [if_neg h,if_neg h]

def sequenceCode (code : Task → Code α Unit) : List Task → Code α Unit
  | [] => pure ()
  | q::tail => do
      let _ ← code q
      sequenceCode code tail

theorem eval_sequence (a : Arithmetic α) (w : Window)
    (code : Task → Code α Unit) (value : Memory (Cell α) → Nat → Nat → Cell α)
    (typed : ∀ m r c,∃ x,value m r c=.fp32 x)
    (localRule : ∀ q m,(code q).eval a m=
      ((),writeWord m (w.addr q.1 q.2) ((value m q.1 q.2).float a.zero)))
    (tasks : List Task) (valid : ∀ q∈tasks,q.1<w.rows ∧ q.2<w.columns)
    (m : Memory (Cell α)) :
    (sequenceCode code tasks).eval a m=((),Schedule.run ((unitRect w).step value) tasks m) := by
  induction tasks generalizing m with
  | nil => rfl
  | cons q tail ih =>
    simp only [sequenceCode,bind,eval_bind,localRule,Prod.fst,Prod.snd]
    rw [ih (by intro p hp; exact valid p (by simp [hp]))]
    rw [←unit_step a w value typed q (valid q (by simp))]
    rfl

theorem unit_tasks_valid (w : Window) (q : Task) (h : q∈(unitRect w).tasks) :
    q.1<w.rows ∧ q.2<w.columns := by
  apply (Grid.task_mem _ _ _).mp
  simpa only [Rect.tasks,unitRect,ceilDiv,Nat.div_one,Nat.mod_one,ite_true,Nat.add_zero] using h

def dequantGridCode (s : Shape) (b : Buffers) : Code α Unit :=
  sequenceCode (fun q => dequantCode b s q.1 q.2) (unitRect (b.fpb.window s.K s.N)).tasks
def mmGridCode (s : Shape) (b : Buffers) (plan : LibraryPlan s) : Code α Unit :=
  sequenceCode (fun q => outputCode b s plan q.1 q.2) (outputRect s b).tasks
def pipelineCode (s : Shape) (b : Buffers) (plan : LibraryPlan s) : Code α Unit := do
  let _ ← dequantGridCode s b
  mmGridCode s b plan

@[simp] theorem eval_dequantGridCode (a : Arithmetic α) (s : Shape) (b : Buffers)
    (m : Memory (Cell α)) :
    (dequantGridCode s b).eval a m=((),dequantRun a s b pointTile m) := by
  apply eval_sequence a _ _ _ (by intro m r c; exact ⟨_,rfl⟩)
    (by intro q m; simpa only [Location.window,Location.addr,Window.addr,Nat.add_assoc] using eval_dequantCode a b s q.1 q.2 m) _ (unit_tasks_valid _)

@[simp] theorem eval_mmGridCode (a : Arithmetic α) (s : Shape) (b : Buffers)
    (plan : LibraryPlan s) (m : Memory (Cell α)) :
    (mmGridCode s b plan).eval a m=((),mmRun a s b plan m) := by
  apply eval_sequence a _ _ _ (by intro m r c; exact ⟨_,rfl⟩)
    (by intro q m; simpa only [Location.window,Location.addr,Window.addr,Nat.add_assoc] using eval_outputCode a b s plan q.1 q.2 m) _ (unit_tasks_valid _)

@[simp] theorem eval_pipelineCode (a : Arithmetic α) (s : Shape) (b : Buffers)
    (plan : LibraryPlan s) (m : Memory (Cell α)) :
    (pipelineCode s b plan).eval a m=((),run a s b pointTile plan m) := by
  simp only [pipelineCode,bind,eval_bind,eval_dequantGridCode,Prod.fst,Prod.snd,eval_mmGridCode]
  rfl

theorem dequant_tile_independent (a : Arithmetic α) (s : Shape) (b : Buffers)
    (first second : Tile) (sep : Separate b) (m : Memory (Cell α)) :
    dequantRun a s b first m=dequantRun a s b second m := by
  funext p
  by_cases h : (b.fpb.window s.K s.N).inside p
  · have dec := (b.fpb.window s.K s.N).decode_inside p h
    rw [←dec.2.2]
    have h1 := dequant_output a s b first sep m _ _ dec.1 dec.2.1
    have h2 := dequant_output a s b second sep m _ _ dec.1 dec.2.1
    simpa only [Location.window,Location.addr,Window.addr,Nat.add_assoc] using h1.trans h2.symm
  · exact ((dequantRect s b first).frame _ (dequant_stable a s b first sep) m p h).trans
      ((dequantRect s b second).frame _ (dequant_stable a s b second sep) m p h).symm

theorem eval_actual_tile (a : Arithmetic α) (s : Shape) (b : Buffers)
    (tile : Tile) (plan : LibraryPlan s) (sep : Separate b) (m : Memory (Cell α)) :
    (pipelineCode s b plan).eval a m=((),run a s b tile plan m) := by
  rw [eval_pipelineCode,run,run,dequant_tile_independent a s b pointTile tile sep m]

/-- Complete source execution and whole memory equality are consequences of
local event rules, including every store's local frame. -/
theorem native_pipeline (abi : LocalEventABI α σ a) (s : Shape) (b : Buffers)
    (tile : Tile) (plan : LibraryPlan s) (sep : Separate b) (state : σ) :
    abi.observe ((pipelineCode s b plan).native abi state).2=
      run a s b tile plan (abi.observe state) := by
  have h := Code.native_refines abi (pipelineCode s b plan) state
  rw [eval_actual_tile a s b tile plan sep] at h
  exact congrArg Prod.snd h

#axiomsClean unit_step
#axiomsClean eval_sequence
#axiomsClean eval_actual_tile
#axiomsClean native_pipeline
end
end VeriTile.Bench.Repairs.DequantizeMatmul
