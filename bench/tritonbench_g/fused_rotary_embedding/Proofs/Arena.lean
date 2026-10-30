import bench.tritonbench_g.fused_rotary_embedding.Proofs.Contract

namespace VeriTile.Bench.Repairs.FusedRotaryEmbedding.Arena
open Checked
open Classical
noncomputable section
abbrev Address := Fin (2^64)
abbrev Memory := Address → Physical.Byte

def address (c : Call) (r : Region) (b : Nat) : Address :=
  ⟨((c.tensor r).base+b)%(2^64),Nat.mod_lt _ (by decide)⟩

def mirror (c : Call) (m : Memory) : Physical.Bytes :=
  fun r b => if b<4*extent c r then m (address c r b) else 0

/-- Literal aligned four-byte store in a finite uint64 address space. -/
def store (c : Call) (m : Memory) (r : Region) (a : Nat) (v : Native.Word) : Memory :=
  fun k => if h : (c.tensor r).base+4*a≤k.val ∧ k.val<(c.tensor r).base+4*a+4 then
    Physical.encode v ⟨k.val-((c.tensor r).base+4*a),by omega⟩ else m k

theorem region_of_byte (c : Call) (r : Region) (b : Nat) (h : b<4*extent c r) : r∈regions := by
  by_contra hn
  simp [extent,hn] at h

theorem byte_bound (c : Call) (m : Physical.Bytes) (checked : check c m=true)
    (live : 0<c.shape.tokens) (r : Region) (b : Nat) (hb : b<4*extent c r) :
    (c.tensor r).base+b<2^64 ∧ (address c r b).val=(c.tensor r).base+b := by
  have hr := region_of_byte c r b hb
  have hp := (live_checked c m checked live).2.1 r hr
  have size : 4*extent c r≤(c.tensor r).storageBytes := by
    simpa [extent,hr] using hp.2.2.2.2.1
  have bound := hp.2.2.2.2.2
  have hn : (c.tensor r).base+b<2^64 := by omega
  exact ⟨hn,Nat.mod_eq_of_lt hn⟩

theorem mirror_store (c : Call) (initial : Physical.Bytes) (checked : check c initial=true)
    (live : 0<c.shape.tokens) (m : Memory) (r : Region) (a : Nat) (ha : a<extent c r)
    (v : Native.Word) :
    mirror c (store c m r a v)=Physical.store (mirror c m) r a v := by
  have hr : r∈regions := region_of_byte c r (4*a) (by omega)
  have rp := (live_checked c initial checked live).2.1 r hr
  have rs : 4*extent c r≤(c.tensor r).storageBytes := by
    simpa [extent,hr] using rp.2.2.2.2.1
  funext q b
  by_cases hb : b<4*extent c q
  · have hq := region_of_byte c q b hb
    have bp := (byte_bound c initial checked live q b hb).2
    have qp := (live_checked c initial checked live).2.1 q hq
    have qs : 4*extent c q≤(c.tensor q).storageBytes := by
      simpa [extent,hq] using qp.2.2.2.2.1
    by_cases same : q=r
    · subst q
      have he : ((c.tensor r).base+4*a≤(c.tensor r).base+b ∧
          (c.tensor r).base+b<(c.tensor r).base+4*a+4) ↔ b/4=a := by omega
      simp only [mirror,hb,ite_true,store,bp,he,Physical.store,true_and]
      split_ifs with h
      · congr 1
        apply Fin.ext
        simp only
        omega
      · rfl
    · have sep := (live_checked c initial checked live).2.2 q hq r hr same
      have outside : ¬((c.tensor r).base+4*a≤(c.tensor q).base+b ∧
          (c.tensor q).base+b<(c.tensor r).base+4*a+4) := by
        rcases sep with h|h <;> omega
      simp [mirror,hb,store,bp,outside,Physical.store,same]
  · have outside : ¬(q=r ∧ b/4=a) := by
      rintro ⟨rfl,h⟩
      omega
    simp [mirror,hb,Physical.store,outside]

def step (c : Call) (p : Native.Instructions) (w : Work) (m : Memory) : Memory :=
  let bytes := mirror c m
  let config := Physical.configuration c.shape bytes
  let words := Physical.load bytes
  let low := Native.pairWord c.shape p w words false
  let high := Native.pairWord c.shape p w words true
  store c (store c m w.kind.output (config.output w false) low)
    w.kind.output (config.output w true) high

theorem output_extent (c : Call) (m : Physical.Bytes) (h : check c m=true)
    (w : Work) (hw : c.shape.Active w) (o : Bool) :
    (host c m).config.output w o<extent c w.kind.output := by
  have b := output_bound (host c m).config (config_valid c m h) w hw o
  have le := source_extent_le c m h w.kind.output
  cases kw : w.kind <;> simp_all [sourceExtent,Kind.output,host,Physical.configuration] <;> omega

theorem mirror_step (c : Call) (p : Native.Instructions) (w : Work) (m : Memory)
    (checked : check c (mirror c m)=true) (active : c.shape.Active w) :
    mirror c (step c p w m)=Physical.dynamicStep c.shape p w (mirror c m) := by
  have live : 0<c.shape.tokens := by have := active.1; omega
  simp only [step,Physical.dynamicStep,Physical.byteStep]
  have lo := output_extent c (mirror c m) checked w active false
  have hi := output_extent c (mirror c m) checked w active true
  change (Physical.configuration c.shape (mirror c m)).output w false<extent c w.kind.output at lo
  change (Physical.configuration c.shape (mirror c m)).output w true<extent c w.kind.output at hi
  rw [mirror_store c (mirror c m) checked live _ _ _ hi,
      mirror_store c (mirror c m) checked live _ _ _ lo]
  rfl

/-- Admission depends only on descriptors and metadata, which each source
store preserves. No output-valued runtime or destination callback is assumed. -/
theorem step_checked (c : Call) (p : Native.Instructions) (w : Work) (m : Memory)
    (checked : check c (mirror c m)=true) (active : c.shape.Active w) :
    check c (mirror c (step c p w m))=true := by
  rw [mirror_step c p w m checked active]
  have hm : Physical.metadata (Physical.dynamicStep c.shape p w (mirror c m))=
      Physical.metadata (mirror c m) := Physical.byteStep_metadata _ p w _
  have config : Physical.configuration c.shape (Physical.dynamicStep c.shape p w (mirror c m))=
      Physical.configuration c.shape (mirror c m) := by simp [Physical.configuration,hm]
  simpa only [check,metadataCheck,config] using checked

def execute (c : Call) (p : Native.Instructions) : List Work → Memory → Memory
  | [],m => m
  | w::ws,m => execute c p ws (step c p w m)

theorem mirror_execute (c : Call) (p : Native.Instructions) (ws : List Work) (m : Memory)
    (checked : check c (mirror c m)=true) (active : ∀ w∈ws,c.shape.Active w) :
    mirror c (execute c p ws m)=Physical.dynamicExecute c.shape p ws (mirror c m) := by
  induction ws generalizing m with
  | nil => rfl
  | cons w ws ih =>
    simp only [execute,Physical.dynamicExecute]
    rw [ih _ (step_checked c p w m checked (active w (by simp))) (fun x hx => active x (by simp [hx])),
      mirror_step c p w m checked (active w (by simp))]

def run (c : Call) (p : Native.Instructions) (m : Memory) := execute c p (schedule c.shape) m

theorem mirror_run (c : Call) (p : Native.Instructions) (m : Memory)
    (checked : check c (mirror c m)=true) :
    mirror c (run c p m)=Physical.dynamic c.shape p (mirror c m) :=
  mirror_execute c p _ m checked (schedule_active c.shape)

def Outside (c : Call) (m : Physical.Bytes) (a : Address) : Prop :=
  ∀ kind t h d,t<c.shape.tokens → h<c.shape.heads kind → d<c.shape.width →
    ¬((c.tensor kind.output).base+4*logicalOutput (host c m).config kind t h d≤a.val ∧
      a.val<(c.tensor kind.output).base+4*logicalOutput (host c m).config kind t h d+4)

theorem configuration_step (c : Call) (p : Native.Instructions) (w : Work) (m : Memory)
    (checked : check c (mirror c m)=true) (active : c.shape.Active w) :
    (host c (mirror c (step c p w m))).config=(host c (mirror c m)).config := by
  unfold host
  rw [mirror_step c p w m checked active]
  have hm := Physical.byteStep_metadata (Physical.configuration c.shape (mirror c m)) p w (mirror c m)
  simp only [Physical.dynamicStep] at *
  simp only [Physical.configuration] at hm ⊢
  rw [hm]

theorem step_frame (c : Call) (p : Native.Instructions) (w : Work) (m : Memory)
    (checked : check c (mirror c m)=true) (active : c.shape.Active w)
    (a : Address) (outside : Outside c (mirror c m) a) : step c p w m a=m a := by
  have hv := config_valid c (mirror c m) checked
  have lo := outside w.kind w.token w.head (c.shape.dim w false) active.1 active.2.1
    (dim_bound c.shape hv.shape w active false)
  have hi := outside w.kind w.token w.head (c.shape.dim w true) active.1 active.2.1
    (dim_bound c.shape hv.shape w active true)
  have ol := output_logical (host c (mirror c m)).config w false
  have oh := output_logical (host c (mirror c m)).config w true
  change (host c (mirror c m)).config.output w false=logicalOutput (host c (mirror c m)).config w.kind w.token w.head (c.shape.dim w false) at ol
  change (host c (mirror c m)).config.output w true=logicalOutput (host c (mirror c m)).config w.kind w.token w.head (c.shape.dim w true) at oh
  rw [←ol] at lo
  rw [←oh] at hi
  simp only [host] at lo hi
  simp only [step,store,lo,hi,↓reduceDIte]

theorem execute_frame (c : Call) (p : Native.Instructions) (ws : List Work) (m : Memory)
    (checked : check c (mirror c m)=true) (active : ∀ w∈ws,c.shape.Active w)
    (a : Address) (outside : Outside c (mirror c m) a) : execute c p ws m a=m a := by
  induction ws generalizing m with
  | nil => rfl
  | cons w ws ih =>
    have hw := active w (by simp)
    have next := step_checked c p w m checked hw
    have same := configuration_step c p w m checked hw
    have out : Outside c (mirror c (step c p w m)) a := by
      simpa only [Outside,same] using outside
    change execute c p ws (step c p w m) a=m a
    rw [ih _ next (fun x hx => active x (by simp [hx])) out,step_frame c p w m checked hw a outside]

theorem run_frame (c : Call) (p : Native.Instructions) (m : Memory)
    (checked : check c (mirror c m)=true) (a : Address) (outside : Outside c (mirror c m) a) :
    run c p m a=m a := execute_frame c p _ m checked (schedule_active c.shape) a outside

theorem empty_run (c : Call) (p : Native.Instructions) (m : Memory)
    (empty : c.shape.tokens=0) : run c p m=m := by
  rw [run,Host.empty_schedule c.shape (by simp [Host.empty,empty])]
  rfl

#axiomsClean mirror_store
#axiomsClean mirror_run
#axiomsClean run_frame
#axiomsClean empty_run
end
end VeriTile.Bench.Repairs.FusedRotaryEmbedding.Arena
