import bench.repairs.tritonbench_g.context_attn_llama.Proofs.WrapperSetup

namespace VeriTile.Bench.Repairs.ContextAttnLlama.Wrapper
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
set_option maxHeartbeats 4000000

abbrev Coordinate (c : Host.Call) := FPHost.Coordinate c
def active (c : Host.Call) (i : Coordinate c) := i.2.2.1 < FPHost.copyRows c i.1

def batchView (c : Host.Call) (o : FPViews.Output (Coordinate c)) (b : Fin c.batches) (mem : Memory) :=
  {FPHost.capture c o with initial := mem, active := fun i => active c i ∧ i.1 = b}
def batchCopy (c : Host.Call) (o : FPViews.Output (Coordinate c))
    (desired : Coordinate c → BitVec 32) (b : Fin c.batches) (mem : Memory) : Memory :=
  (batchView c o b mem).copyBack desired

def batchWrites (c : Host.Call) (o : FPViews.Output (Coordinate c)) (b : Fin c.batches) (cell : Key) : Prop :=
  ∃ i, active c i ∧ i.1 = b ∧ o.address i = cell

theorem batch_frame (c : Host.Call) (o : FPViews.Output (Coordinate c))
    (desired : Coordinate c → BitVec 32) (b : Fin c.batches) (mem : Memory)
    (cell : Key) (h : ¬ batchWrites c o b cell) : batchCopy c o desired b mem cell = mem cell := by
  apply (batchView c o b mem).copyBack_frame
  simpa only [batchView,FPHost.capture,batchWrites,and_assoc] using h

theorem batch_read (c : Host.Call) (o : FPViews.Output (Coordinate c))
    (desired : Coordinate c → BitVec 32) (b : Fin c.batches) (mem : Memory)
    (unique : ∀ i j, active c i → active c j → o.address i = o.address j → i = j)
    (i : Coordinate c) (hi : active c i) (hb : i.1 = b) :
    o.logical (batchCopy c o desired b mem) i = desired i := by
  exact (batchView c o b mem).copyBack_read desired (fun i j hi hj => unique i j hi.1 hj.1) i ⟨hi,hb⟩

inductive CopySite where | destinationSlice | sourceSlice | copy deriving DecidableEq

/-- The ordinary copy contract allows an error to leave a partial write in
the attempted slice. It cannot change other caller cells. Successful copy is
the coordinate-preserving sign-bit write; slicing itself does not write. -/
structure CopyAPI (c : Host.Call) (o : FPViews.Output (Coordinate c)) (desired : Coordinate c → BitVec 32) where
  destinationSlice : Fin c.batches → Except String Unit
  sourceSlice : Fin c.batches → Except String Unit
  copy : Fin c.batches → Memory → StepResult Memory
  success : ∀ b mem after, copy b mem = .ok after → after = batchCopy c o desired b mem
  frame : ∀ b mem cell, ¬ batchWrites c o b cell → (copy b mem).state cell = mem cell

inductive CopyResult (c : Host.Call) where
  | ok (memory : Memory)
  | failed (batch : Fin c.batches) (site : CopySite) (error : String) (memory : Memory)

def CopyResult.memory {c : Host.Call} : CopyResult c → Memory
  | .ok mem | .failed _ _ _ mem => mem

def attempt (api : CopyAPI c o desired) (b : Fin c.batches) (mem : Memory) : CopyResult c :=
  if FPHost.copyRows c b = 0 then .ok mem else
  match api.destinationSlice b with
  | .error e => .failed b .destinationSlice e mem
  | .ok _ => match api.sourceSlice b with
    | .error e => .failed b .sourceSlice e mem
    | .ok _ => match api.copy b mem with
      | .failed e after => .failed b .copy e after
      | .ok after => .ok after

def copies (api : CopyAPI c o desired) : List (Fin c.batches) → Memory → CopyResult c
  | [],mem => .ok mem
  | b :: rest,mem => match attempt api b mem with
    | .failed b site e after => .failed b site e after
    | .ok after => copies api rest after

inductive CopyFailedAt (api : CopyAPI c o desired) :
    List (Fin c.batches) → Memory → Fin c.batches → CopySite → String → Memory → Prop
  | here {b rest mem failed site e after} (h : attempt api b mem = .failed failed site e after) :
      CopyFailedAt api (b :: rest) mem failed site e after
  | later {b rest mem next failed site e after} (h : attempt api b mem = .ok next)
      (tail : CopyFailedAt api rest next failed site e after) :
      CopyFailedAt api (b :: rest) mem failed site e after

theorem copies_failure (api : CopyAPI c o desired) (order : List (Fin c.batches))
    (mem : Memory) (b : Fin c.batches) (site : CopySite) (e : String) (after : Memory)
    (h : copies api order mem = .failed b site e after) : CopyFailedAt api order mem b site e after := by
  induction order generalizing mem with
  | nil => simp [copies] at h
  | cons next rest ih =>
    cases hs : attempt api next mem with
    | failed batch st err state =>
      simp only [copies,hs,CopyResult.failed.injEq] at h
      rcases h with ⟨rfl,rfl,rfl,rfl⟩
      exact .here hs
    | ok state => exact .later hs (ih state (by simpa [copies,hs] using h))

theorem attempt_frame (api : CopyAPI c o desired) (b : Fin c.batches) (mem : Memory)
    (cell : Key) (h : ¬ batchWrites c o b cell) : (attempt api b mem).memory cell = mem cell := by
  unfold attempt
  split
  · rfl
  · cases hd : api.destinationSlice b with
    | error e => rfl
    | ok u =>
      cases hs : api.sourceSlice b with
      | error e => rfl
      | ok u =>
        have hf := api.frame b mem cell h
        cases hc : api.copy b mem <;> simpa [hc,CopyResult.memory,StepResult.state] using hf

theorem attempt_read (api : CopyAPI c o desired) (b : Fin c.batches) (mem after : Memory)
    (unique : ∀ i j, active c i → active c j → o.address i = o.address j → i = j)
    (run : attempt api b mem = .ok after) (i : Coordinate c) (hi : active c i) (hb : i.1 = b) :
    o.logical after i = desired i := by
  have hn : FPHost.copyRows c b ≠ 0 := by
    have := hi; unfold active at this; rw [hb] at this; omega
  cases hd : api.destinationSlice b with
  | error e => simp [attempt,hn,hd] at run
  | ok u =>
    cases hs : api.sourceSlice b with
    | error e => simp [attempt,hn,hd,hs] at run
    | ok u =>
      cases hc : api.copy b mem with
      | failed e state => simp [attempt,hn,hd,hs,hc] at run
      | ok state =>
        have he : state = after := by simpa [attempt,hn,hd,hs,hc] using run
        rw [← he,api.success b mem state hc]
        exact batch_read c o desired b mem unique i hi hb

theorem copies_frame (api : CopyAPI c o desired) (order : List (Fin c.batches)) (mem : Memory)
    (cell : Key) (h : ¬∃ i, active c i ∧ o.address i = cell) :
    (copies api order mem).memory cell = mem cell := by
  induction order generalizing mem with
  | nil => rfl
  | cons b rest ih =>
    have hb : ¬ batchWrites c o b cell := by
      rintro ⟨i,hi,hb,he⟩; exact h ⟨i,hi,he⟩
    have hf := attempt_frame api b mem cell hb
    cases he : attempt api b mem with
    | failed batch site e after => simpa [copies,he] using hf
    | ok after => simpa [copies,he] using (ih after).trans (by simpa [he] using hf)

theorem copies_reads (api : CopyAPI c o desired) (order : List (Fin c.batches)) (mem after : Memory)
    (unique : ∀ i j, active c i → active c j → o.address i = o.address j → i = j)
    (run : copies api order mem = .ok after) :
    ∀ i, active c i → (i.1 ∈ order ∨ o.logical mem i = desired i) → o.logical after i = desired i := by
  induction order generalizing mem with
  | nil =>
    cases run
    intro i hi h
    exact h.resolve_left (by simp)
  | cons b rest ih =>
    cases he : attempt api b mem with
    | failed batch site err state => simp [copies,he] at run
    | ok state =>
      have hr : copies api rest state = .ok after := by simpa [copies,he] using run
      intro i hi h
      apply ih state hr i hi
      by_cases hb : i.1 = b
      · exact Or.inr (attempt_read api b mem state unique he i hi hb)
      · rcases h with hmem | hold
        · exact Or.inl ((List.mem_cons.mp hmem).resolve_left hb)
        · apply Or.inr
          have hmiss : ¬ batchWrites c o b (o.address i) := by
            rintro ⟨j,hj,hb',heq⟩
            have hij : j=i := unique j i hj hi heq
            exact hb (hij ▸ hb')
          have hf := attempt_frame api b mem (o.address i) hmiss
          simp only [he,CopyResult.memory] at hf
          simpa only [FPViews.Output.logical,hf] using hold

theorem complete_copy (api : CopyAPI c o desired) (mem after : Memory)
    (unique : ∀ i j, active c i → active c j → o.address i = o.address j → i = j)
    (run : copies api (List.finRange c.batches) mem = .ok after) :
    (∀ i, active c i → o.logical after i = desired i) ∧
    (∀ cell, (¬∃ i, active c i ∧ o.address i = cell) → after cell = mem cell) := by
  constructor
  · intro i hi
    exact copies_reads api _ mem after unique run i hi (Or.inl (List.mem_finRange i.1))
  · intro cell h
    simpa [run] using copies_frame api (List.finRange c.batches) mem cell h

#print axioms complete_copy
#print axioms copies_failure
#print axioms copies_frame
end
end VeriTile.Bench.Repairs.ContextAttnLlama.Wrapper
