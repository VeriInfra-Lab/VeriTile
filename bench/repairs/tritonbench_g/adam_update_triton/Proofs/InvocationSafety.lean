import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Compilation
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.InvocationSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton.Physical (Role)
open VeriTile.Bench.Repairs.SwigluTriton.Index (signed_ofNat)
open VeriTile.Bench.TritonBenchG.AdamUpdateTriton.HostAutotune (Config)
open Host
noncomputable section

/-- The native word key uses the actual private allocation owner and the
signed-wide offset added to the actual sliced pointer. -/
def key (p : Private.Layout) (chunk : Index.Chunk) (config : Config)
    (role : Role) (pid lane : Nat) : Key :=
  ⟨(p.allocation role).owner,
    (Index.pointer (p.allocation role).baseByte chunk.first pid config.block lane).toNat⟩

/-- Certificates quantify every reached PID and every lane, including masked
tails. Backing is required only for active word accesses, never for masked
pointer formation. All bounds are consequences of the existing chunk plan. -/
structure Coordinates (p : Private.Layout) (chunk : Index.Chunk) (config : Config)
    (schedule : List Nat) : Prop where
  member : chunk ∈ Index.chunks 0 p.total
  positive : 0 < chunk.count
  countBound : chunk.count ≤ 8388480
  endpoint : chunk.first+chunk.count ≤ p.total
  complete : schedule.Perm (List.range (Index.grid chunk.count config))
  gridPositive : 0 < Index.grid chunk.count config
  gridBound : Index.grid chunk.count config ≤ 65535
  launcherProduct : Index.grid chunk.count config*1*1 < 2147483648
  effectiveX : Index.grid chunk.count config*1 < 2147483648
  countSigned : (BitVec.ofNat 32 chunk.count).toInt=(chunk.count:Int)
  totalSigned : (BitVec.ofNat 64 p.total).toInt=(p.total:Int)
  integers : ∀ pid∈schedule,∀ lane,lane<config.block →
    (BitVec.ofNat 32 pid).toInt=(pid:Int) ∧
    (BitVec.ofNat 32 config.block).toInt=(config.block:Int) ∧
    (BitVec.ofNat 32 lane).toInt=(lane:Int) ∧
    (BitVec.ofNat 32 pid*BitVec.ofNat 32 config.block).toInt=(pid*config.block:Nat) ∧
    (Index.offset pid config.block lane).toInt=(pid*config.block+lane:Nat)
  mask : ∀ pid∈schedule,∀ lane,lane<config.block →
    (Index.offset pid config.block lane).slt (BitVec.ofNat 32 chunk.count)=
      decide (pid*config.block+lane<chunk.count)
  modular : ∀ role pid,pid∈schedule → ∀ lane,lane<config.block →
    (key p chunk config role pid lane).address=
      ((p.allocation role).baseByte+4*(chunk.first+pid*config.block+lane))%2^64
  active : ∀ role pid,pid∈schedule → ∀ lane,lane<config.block →
    pid*config.block+lane<chunk.count →
    key p chunk config role pid lane=(p.allocation role).key (chunk.first+pid*config.block+lane) ∧
    (p.allocation role).contains (key p chunk config role pid lane)

/-- Derive all integer/launcher/address facts from prepared capacities and
membership in the actual chunk list; no native-safe input premise is added. -/
theorem coordinates (p : Private.Layout) (valid : Private.Valid p)
    (chunk : Index.Chunk) (member : chunk∈Index.chunks 0 p.total)
    (config : Config) (schedule : List Nat)
    (complete : schedule.Perm (List.range (Index.grid chunk.count config))) :
    Coordinates p chunk config schedule := by
  obtain ⟨_,positive,bounded,endpoint⟩ := Index.chunks_bounds 0 p.total chunk member
  have capacity (role : Role) : p.total ≤ (p.allocation role).words := by
    simpa [Private.Layout.flat] using valid.capacity role
  have reached (pid : Nat) (hp : pid∈schedule) : pid<Index.grid chunk.count config :=
    List.mem_range.mp (complete.mem_iff.mp hp)
  refine ⟨member,positive,bounded,by omega,complete,?_,
    Index.grid_bound _ _ bounded,Compilation.launcher_product _ _ bounded,?_,
    signed_ofNat 32 chunk.count (by decide) (by omega),
    Index.total_i64 (p.allocation .a) p.total (capacity .a),?_,?_,?_,?_⟩
  · cases config <;> simp only [Index.grid,Config.block] <;> omega
  · have := Index.grid_bound chunk.count config bounded
    omega
  · intro pid hp lane hl
    exact Index.i32_exact _ _ _ _ bounded (reached pid hp) hl
  · intro pid hp lane hl
    exact Index.mask_exact _ _ _ _ bounded (reached pid hp) hl
  · intro role pid hp lane hl
    exact Index.pointer_modular _ _ _ _ _ _ bounded (reached pid hp) hl
  · intro role pid hp lane hl active
    obtain ⟨address,contained⟩ := Index.active_pointer (p.allocation role) p.total chunk.first chunk.count
      config pid lane (capacity role) (by omega) bounded (reached pid hp) hl active
    have exactKey : key p chunk config role pid lane=
        (p.allocation role).key (chunk.first+pid*config.block+lane) := by
      unfold key Allocation.key
      simp only [Key.mk.injEq]
      exact ⟨trivial,address⟩
    exact ⟨exactKey,exactKey.symm ▸ contained⟩

/-- The address certificate is attached to this very source step, including
its actual input memory, finite source execution and output memory. It is
independent of the numerical carrier; Real and raw FP32 use the same proof. -/
structure Step {M α β τ : Type} [Inhabited α] [Inhabited τ]
    (ops : Ops M α β) (model : Machine.Model α τ) (p : Private.Layout)
    (chunk : Index.Chunk) (config : Config) (schedule : List Nat) (before after : M)
    (_source : Machine.SourceStep ops model p chunk config schedule before after) : Prop where
  coordinates : Coordinates p chunk config schedule
  loads : ∀ role pid,pid∈schedule → ∀ lane,lane<config.block →
    pid*config.block+lane<chunk.count →
    model.observe (((Machine.state ops model p chunk.first before).mem (Private.region role)
      (pid*config.block+lane)).read .real)=ops.read before (key p chunk config role pid lane)
  execution : ∃ final,
    Grid.runPrograms model.semantics model.arguments p.total chunk.count config.block schedule
      (Machine.state ops model p chunk.first before)=some final ∧
    ∀ role pid,pid∈schedule → ∀ lane,lane<config.block → pid*config.block+lane<chunk.count →
      model.observe ((final.mem (Private.region role) (pid*config.block+lane)).read .real)=
        ops.read after (key p chunk config role pid lane)

theorem step {M α β τ : Type} [Inhabited α] [Inhabited τ]
    (ops : Ops M α β) (model : Machine.Model α τ) (p : Private.Layout) (valid : Private.Valid p)
    (chunk : Index.Chunk) (member : chunk∈Index.chunks 0 p.total)
    (config : Config) (schedule : List Nat) (before after : M)
    (source : Machine.SourceStep ops model p chunk config schedule before after) :
    Step ops model p chunk config schedule before after source := by
  have cert := coordinates p valid chunk member config schedule source.schedule_complete
  refine ⟨cert,?_,?_⟩
  · intro role pid hp lane hl active
    rw [(cert.active role pid hp lane hl active).1,Machine.state_read]
    congr 2
    omega
  · obtain ⟨final,ran,readback⟩ := source.source
    refine ⟨final,ran,?_⟩
    intro role pid hp lane hl active
    rw [(cert.active role pid hp lane hl active).1]
    have inside : chunk.first+(pid*config.block+lane)<p.total := by have := cert.endpoint; omega
    simpa only [Nat.add_assoc] using readback role (pid*config.block+lane) inside

/-- Additional local compiler evidence, when requested, is carried by the
same constructor and actual source-entry state as the address certificate. -/
inductive Trials {M α β τ : Type} [Inhabited α] [Inhabited τ]
    (ops : Ops M α β) (model : Machine.Model α τ) (p : Private.Layout)
    (extra : Index.Chunk → Config → M → Prop) (chunk : Index.Chunk) (caller : Nat → Nat → Prop) :
    {history : List Config} → {before after : M} →
      Autotune.Trials ops model p chunk caller history before after → Prop
  | nil (state) : Trials ops model p extra chunk caller (.nil state)
  | cons {config rest schedule before prepared trial restored after} {prepare source restore tail}
      (head : Step ops model p chunk config schedule prepared trial source)
      (bound : extra chunk config prepared)
      (next : Trials ops model p extra chunk caller tail) :
      Trials ops model p extra chunk caller
        (.cons config rest schedule before prepared trial restored after prepare source restore tail)
  | administrative {history before prepared after} {effect tail}
      (next : Trials ops model p extra chunk caller tail) :
      Trials ops model p extra chunk caller (.administrative history before prepared after effect tail)

def Launch {M α β τ : Type} [Inhabited α] [Inhabited τ]
    (ops : Ops M α β) (model : Machine.Model α τ) (p : Private.Layout)
    (extra : Index.Chunk → Config → M → Prop) (chunk : Index.Chunk) (caller : Nat → Nat → Prop)
    (before after : M) (launch : Autotune.Launched ops model p chunk caller before after) : Prop :=
  Trials ops model p extra chunk caller launch.trials ∧
  Step ops model p chunk launch.selected launch.schedule launch.restored after launch.selectedSource ∧
  extra chunk launch.selected launch.restored

theorem trials {M α β τ : Type} [Inhabited α] [Inhabited τ]
    (ops : Ops M α β) (model : Machine.Model α τ) (p : Private.Layout) (valid : Private.Valid p)
    (extra : Index.Chunk → Config → M → Prop) (chunk : Index.Chunk)
    (member : chunk∈Index.chunks 0 p.total) (bound : ∀ config state,extra chunk config state)
    (caller : Nat → Nat → Prop) (history : List Config) (before after : M)
    (run : Autotune.Trials ops model p chunk caller history before after) :
    Trials ops model p extra chunk caller run := by
  induction run with
  | nil state => exact .nil state
  | cons config rest schedule before prepared trial restored after prepare source restore tail ih =>
    exact .cons (prepare:=prepare) (restore:=restore)
      (step ops model p valid chunk member config schedule prepared trial source) (bound config prepared) ih
  | administrative history before prepared after effect tail ih => exact .administrative (effect:=effect) ih

inductive Launches {M α β τ : Type} [Inhabited α] [Inhabited τ]
    (ops : Ops M α β) (model : Machine.Model α τ) (p : Private.Layout)
    (extra : Index.Chunk → Config → M → Prop) (caller : Nat → Nat → Prop) :
    {parts : List Index.Chunk} → {before after : M} →
      Autotune.Launches ops model p caller parts before after → Prop
  | nil (state) : Launches ops model p extra caller (.nil state)
  | cons {chunk rest before middle after} {head tail}
      (first : Launch ops model p extra chunk caller before middle head)
      (next : Launches ops model p extra caller tail) :
      Launches ops model p extra caller (.cons chunk rest before middle after head tail)

theorem launches {M α β τ : Type} [Inhabited α] [Inhabited τ]
    (ops : Ops M α β) (model : Machine.Model α τ) (p : Private.Layout) (valid : Private.Valid p)
    (extra : Index.Chunk → Config → M → Prop)
    (bound : ∀ chunk∈Index.chunks 0 p.total,∀ config state,extra chunk config state)
    (caller : Nat → Nat → Prop) (parts : List Index.Chunk)
    (members : ∀ chunk∈parts,chunk∈Index.chunks 0 p.total) (before after : M)
    (run : Autotune.Launches ops model p caller parts before after) :
    Launches ops model p extra caller run := by
  induction run with
  | nil state => exact .nil state
  | cons chunk rest before middle after head tail ih =>
    have member := members chunk (by simp)
    exact .cons ⟨trials ops model p valid extra chunk member (bound chunk member)
        caller head.history before head.restored head.trials,
      step ops model p valid chunk member head.selected head.schedule head.restored middle head.selectedSource,
      bound chunk member head.selected head.restored⟩
      (ih (fun c hc => members c (by simp [hc])))

#axiomsClean coordinates
#axiomsClean step
#axiomsClean trials
#axiomsClean launches
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.InvocationSafety
