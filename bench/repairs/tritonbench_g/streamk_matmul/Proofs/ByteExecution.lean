import bench.repairs.tritonbench_g.streamk_matmul.Proofs.ByteMemory
import bench.repairs.tritonbench_g.streamk_matmul.Whole001.Native

namespace VeriTile.Bench.Repairs.StreamK.Checked.ByteExecution
open TensorBinding ByteMemory Whole001 Source
noncomputable section
set_option maxHeartbeats 4000000

structure Layout (p : Parameters) where
  outputBase : Nat
  scratchBase : Nat
  scratchActive : Bool
  scratch_eq : scratchActive=decide (0<(Plan.select p.h).streamTiles)

def cView (v : Layout p) : View := ⟨v.outputBase,p.h.m*p.h.n⟩
def sView (v : Layout p) : View := ⟨v.scratchBase,if v.scratchActive then p.h.programs*p.h.bm*p.h.bn else 0⟩
structure Laws (v : Layout p) : Prop where
  separate : Separate (cView v) (sView v)
  cWindow : v.outputBase+4*(cView v).extent≤2^64
  sWindow : v.scratchActive=Bool.true → v.scratchBase+4*(sView v).extent≤2^64

def address (base index : Nat) : Nat :=
  (BitVec.ofNat 64 base+BitVec.ofNat 64 index*BitVec.ofNat 64 4).toNat

theorem address_eq (base index extent : Nat) (hi : index<extent) (h : base+4*extent≤2^64) :
    address base index=base+4*index := by
  simp only [address,BitVec.toNat_add,BitVec.toNat_mul,BitVec.toNat_ofNat]
  norm_num
  rw [Nat.mod_eq_of_lt (by omega : base+index*4<18446744073709551616)]
  omega

def rendered (read : Read) (v : Layout p) (s : State Word p exterior) : Read :=
  render (render read (sView v) (extend s.scratch)) (cView v) (extend s.output)

def Allowed (v : Layout p) : Event α p → Prop
  | .storeOutput _ _ => True
  | .storeScratch _ _ => v.scratchActive=Bool.true
  | .atomic _ _ => v.scratchActive=Bool.true

abbrev Machine (p : Parameters) (exterior : Nat) := State Word p exterior × Read
/-- Store addresses use the actual 64-bit pointer expression. Atomics read
both current physical regions before writing the scalar instruction result. -/
def step (v : Layout p) (add : Word → Word → Word) (e : Event Word p)
    (t : Machine p exterior) : Machine p exterior :=
  match e with
  | .storeOutput i f =>
      (Memory.step add e t.1,store t.2 (address v.outputBase i.val) (f t.1.inputA t.1.inputB).bits)
  | .storeScratch i f =>
      (Memory.step add e t.1,store t.2 (address v.scratchBase i.val) (f t.1.inputA t.1.inputB).bits)
  | .atomic i j =>
      let old : Word := ⟨read32 t.2 (address v.outputBase i.val)⟩
      let part : Word := ⟨read32 t.2 (address v.scratchBase j.val)⟩
      (Memory.step add e t.1,store t.2 (address v.outputBase i.val) (add old part).bits)

theorem step_render (v : Layout p) (laws : Laws v) (add : Word → Word → Word)
    (e : Event Word p) (allowed : Allowed v e) (read : Read) (s : State Word p exterior) :
    step v add e (s,rendered read v s)=
      (Memory.step add e s,rendered read v (Memory.step add e s)) := by
  cases e with
  | storeOutput i f =>
    have addr := address_eq v.outputBase i.val (cView v).extent i.isLt laws.cWindow
    simp only [step,addr,Memory.step,rendered,extend_update]
    rw [render_write _ _ _ _ i.isLt]
    rfl
  | storeScratch i f =>
    have active : v.scratchActive=Bool.true := allowed
    have size : (sView v).extent=p.h.programs*p.h.bm*p.h.bn := by simp [sView,active]
    have hi : i.val<(sView v).extent := by rw [size]; exact i.isLt
    have addr := address_eq v.scratchBase i.val (sView v).extent hi (laws.sWindow active)
    simp only [step,addr,Memory.step,rendered,extend_update]
    rw [render_write _ _ _ _ hi,render_store_other _ _ _ _ _ hi _ laws.separate]
    rfl
  | atomic i j =>
    have active : v.scratchActive=Bool.true := allowed
    have size : (sView v).extent=p.h.programs*p.h.bm*p.h.bn := by simp [sView,active]
    have hj : j.val<(sView v).extent := by rw [size]; exact j.isLt
    have ca := address_eq v.outputBase i.val (cView v).extent i.isLt laws.cWindow
    have sa := address_eq v.scratchBase j.val (sView v).extent hj (laws.sWindow active)
    have cr : read32 (rendered read v s) (v.outputBase+4*i.val)=(s.output i).bits := by
      exact (render_read _ _ _ _ i.isLt).trans (congrArg (fun x : Word => x.bits) (extend_at s.output i))
    have sr : read32 (rendered read v s) (v.scratchBase+4*j.val)=(s.scratch j).bits := by
      unfold rendered
      change read32 (render (render read (sView v) (extend s.scratch)) (cView v) (extend s.output)) ((sView v).base+4*j.val)=_
      rw [render_read_other _ (cView v) (sView v) _ _ hj laws.separate]
      exact (render_read _ _ _ _ hj).trans (congrArg (fun x : Word => x.bits) (extend_at s.scratch j))
    simp only [step,ca,sa]
    change (Memory.step add (.atomic i j) s,store (rendered read v s) (v.outputBase+4*i.val) (add ⟨read32 (rendered read v s) (v.outputBase+4*i.val)⟩ ⟨read32 (rendered read v s) (v.scratchBase+4*j.val)⟩).bits)=_
    rw [cr,sr]
    simp only [Memory.step,rendered,extend_update]
    rw [render_write _ _ _ _ i.isLt]
    rfl

def run (v : Layout p) (add : Word → Word → Word) (events : List (Event Word p)) :
    Machine p exterior → Machine p exterior := events.foldl (fun t e => step v add e t)

theorem run_render (v : Layout p) (laws : Laws v) (add : Word → Word → Word)
    (events : List (Event Word p)) (allowed : ∀ e∈events,Allowed v e)
    (read : Read) (s : State Word p exterior) :
    run v add events (s,rendered read v s)=
      (Memory.run add events s,rendered read v (Memory.run add events s)) := by
  induction events generalizing s with
  | nil => rfl
  | cons e tail ih =>
    simp only [run,List.foldl_cons,step_render v laws add e (allowed e (by simp))]
    exact ih (fun x hx => allowed x (by simp [hx])) _

theorem first_allowed (v : Layout p) (ops : Instructions α) (e : Event α p)
    (he : e∈firstWave p ops) : Allowed v e := by
  by_cases active : 0<(Plan.select p.h).streamTiles
  · have va : v.scratchActive=Bool.true := by simp [v.scratch_eq,active]
    cases e <;> simp [Allowed,va]
  · simp [firstWave,active] at he

theorem full_allowed (v : Layout p) (ops : Instructions α) (e : Event α p)
    (he : e∈fullWave p ops) : Allowed v e := by
  unfold fullWave at he
  obtain ⟨pid,hpid,he⟩ := List.mem_flatMap.mp he
  unfold storeTile at he
  obtain ⟨lane,hl,he⟩ := List.mem_flatMap.mp he
  split at he
  · simp only [List.mem_singleton] at he
    subst e
    trivial
  · simp at he

theorem reduce_allowed (v : Layout p) (schedule : Schedule.CTAInterleaving p)
    (e : Event α p) (he : e∈(schedule.events.map Prod.snd).map fun pair => .atomic pair.1 pair.2) :
    Allowed v e := by
  obtain ⟨pair,hp,rfl⟩ := List.mem_map.mp he
  obtain ⟨tag,ht,rfl⟩ := List.mem_map.mp hp
  have htag := (Schedule.shuffle_perm _ _ _ schedule.order).mem_iff.mpr ht
  have origin := Schedule.tagged_origin p tag htag
  have active : 0<(Plan.select p.h).streamTiles := by omega
  simp [Allowed,v.scratch_eq,active]

def call (v : Layout p) (ops : Instructions Word)
    (first : Schedule.Order (firstWave p ops).length) (reduce : Schedule.CTAInterleaving p)
    (last : Schedule.Order (fullWave p ops).length) (initial : Machine p exterior) : Machine p exterior :=
  if p.h.m=0 ∨ p.h.n=0 then initial else
    run v ops.atomicAdd (Schedule.reorder (fullWave p ops) last)
      (run v ops.atomicAdd ((reduce.events.map Prod.snd).map fun pair => .atomic pair.1 pair.2)
        (run v ops.atomicAdd (Schedule.reorder (firstWave p ops) first) initial))

theorem call_render (v : Layout p) (laws : Laws v) (ops : Instructions Word)
    (first : Schedule.Order (firstWave p ops).length) (reduce : Schedule.CTAInterleaving p)
    (last : Schedule.Order (fullWave p ops).length) (read : Read) (initial : State Word p exterior) :
    call v ops first reduce last (initial,rendered read v initial)=
      let final := Schedule.call p ops first reduce last initial
      (final,rendered read v final) := by
  unfold call Schedule.call
  split
  · rfl
  · rw [run_render v laws _ _ (fun e he => first_allowed v ops e ((Schedule.reorder_perm _ first).mem_iff.mp he)),
      run_render v laws _ _ (reduce_allowed v reduce),
      run_render v laws _ _ (fun e he => full_allowed v ops e ((Schedule.reorder_perm _ last).mem_iff.mp he))]

#axiomsClean address_eq
#axiomsClean step_render
#axiomsClean run_render
#axiomsClean call_render
end
end VeriTile.Bench.Repairs.StreamK.Checked.ByteExecution
