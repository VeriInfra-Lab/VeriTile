import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeMachineLoads

/-! The actual word-addressed input-stream graph. Every raw/scale/seed operand
comes from the guarded physical load implementation, under the selected launch
and ordinary tensor allocation contracts. Inactive scratch fields need not
match: only source-enabled operations produce registers. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeMachine
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow Address PhysicalDomain NativeABI
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000

structure Config (α : Type) (block : Nat) where
  ops : Ops α
  reductions : Nat → Whole.Reductions block
  plan : AffinePlan
  plan1 : AffinePlan
  flags : Flags
  regions : Regions
  strides : Strides
  M : Nat
  N : Nat
  eps : α
  p : α
  allocations : RegionName → Allocation
  accepted : Accepted M N block
  storage : PhysicalDomain.Domain flags regions strides M N allocations

abbrev Memory {α : Type} (c : Config α block) := TypedPhysical.Physical α (layout c.allocations)

def view {α : Type} (c : Config α block) (physical : Memory c) (template : State α) : State α :=
  TypedPhysical.pull (layout c.allocations) physical template

def inputRole (second : Bool) : Role := if second then .x1 else .x
def scaleRole (second : Bool) : Role := if second then .rowscale1 else .rowscale
def seedRole (second : Bool) : Role := if second then .seeds1 else .seeds
def maskRole (second : Bool) : Role := if second then .mask1 else .mask
def inputBase (stride : Strides) (row : Nat) (second : Bool) : Nat :=
  row*(if second then stride.x1 else stride.x)
def inputIndex (M row : Nat) (second : Bool) : Nat := if second then M+row else row

def stream {α : Type} [Inhabited α] (c : Config α block) (row : Fin c.M) (second : Bool)
    (physical : Memory c) : NativeSchedule.StreamRegisters α block :=
  let x := fun j : Fin block => NativeMachineMemory.raw c.allocations c.regions c.strides
    c.M c.N row.val j.val c.flags (inputRole second) physical
  let scale := NativeMachineMemory.scalar c.allocations c.regions c.strides
    c.M c.N row.val c.flags (scaleRole second) physical
  let seed := (NativeMachineMemory.loaded c.allocations c.regions c.strides
    c.M c.N row.val 0 c.flags (seedRole second) physical).read .int
  let random := fun j : Fin block => Expr.random seed j.val
  { raw := x
    scale := scale
    value := fun j =>
      let scaled := if c.flags.rowscale then Expr.mul (x j) scale else x j
      if c.flags.dropout then .selectGT (random j) (.input c.p)
        (.div scaled (.sub (.literal 1) (.input c.p))) (.literal 0) else scaled
    keep := fun j => c.ops.greater ((random j).eval c.ops) ((Expr.input c.p).eval c.ops) }

structure StreamMatch {α : Type} (f : Flags) (word logical : NativeSchedule.StreamRegisters α block) : Prop where
  raw : word.raw=logical.raw
  scale : f.rowscale=Bool.true → word.scale=logical.scale
  value : word.value=logical.value
  keep : f.dropout=Bool.true → word.keep=logical.keep

theorem stream_correspondence {α : Type} [Inhabited α] (c : Config α block) (row : Fin c.M)
    (second : Bool) (hx : second=Bool.true → c.flags.x1=Bool.true)
    (physical : Memory c) (template : State α) :
    StreamMatch c.flags (stream c row second physical)
      (NativeSchedule.stream c.ops c.flags c.regions (region c.regions (inputRole second))
        (inputBase c.strides row.val second) (inputIndex c.M row.val second) c.N c.p block (view c physical template)) := by
  have hactive : active c.flags (inputRole second) := by
    cases second <;> simp_all [inputRole,active]
  have hvector : scalarRole (inputRole second)=Bool.false := by cases second <;> rfl
  have hoffset : ∀ col, offset c.strides c.M c.N row.val col (inputRole second)=inputBase c.strides row.val second+col := by
    cases second <;> intro col <;> rfl
  have hraw : ∀ j : Fin block,
      NativeMachineMemory.raw c.allocations c.regions c.strides c.M c.N row.val j.val c.flags (inputRole second) physical=
      NativeSchedule.raw (view c physical template) (region c.regions (inputRole second))
        (inputBase c.strides row.val second) c.N j.val := by
    intro j
    exact NativeMachineMemory.raw_correspondence c.flags c.regions c.strides c.M c.N block
      c.allocations c.accepted c.storage row.val j.val row.isLt j.isLt (inputRole second) hactive hvector
      (inputBase c.strides row.val second) (hoffset j.val) physical template
  have hscale : c.flags.rowscale=Bool.true →
      NativeMachineMemory.scalar c.allocations c.regions c.strides c.M c.N row.val c.flags (scaleRole second) physical=
      NativeSchedule.scalar (view c physical template) c.regions.rowscale (inputIndex c.M row.val second) := by
    intro hs
    have ha : active c.flags (scaleRole second) := by cases second <;> simp_all [scaleRole,active]
    have hscalar : scalarRole (scaleRole second)=Bool.true := by cases second <;> rfl
    have he := NativeMachineMemory.scalar_correspondence c.flags c.regions c.strides c.M c.N block
      c.allocations c.accepted c.storage row.val row.isLt (scaleRole second) ha hscalar physical template
    cases second <;> simpa only [scaleRole,region,offset,inputIndex] using he
  have hseed : c.flags.dropout=Bool.true →
      NativeMachineMemory.loaded c.allocations c.regions c.strides c.M c.N row.val 0 c.flags (seedRole second) physical=
      (view c physical template).mem c.regions.seeds (inputIndex c.M row.val second) := by
    intro hd
    have ha : active c.flags (seedRole second) := by cases second <;> simp_all [seedRole,active]
    have hscalar : scalarRole (seedRole second)=Bool.true := by cases second <;> rfl
    have he := NativeMachineMemory.loaded_eq_tagged c.flags c.regions c.strides c.M c.N block
      c.allocations c.accepted c.storage row.val 0 row.isLt c.accepted.blockPositive (seedRole second) ha
      (NativeMachineMemory.scalar_lane_active _ hscalar c.N 0) physical template
    cases second <;> simpa only [seedRole,region,offset,inputIndex] using he
  refine ⟨funext hraw,?_,?_,?_⟩
  · exact hscale
  · funext j
    cases hs : c.flags.rowscale <;> cases hd : c.flags.dropout
    all_goals simp [stream,NativeSchedule.stream,hs,hd,hraw,hscale,hseed]
  · intro hd
    funext j
    simp only [stream,NativeSchedule.stream,hseed hd]

def storeMask {α : Type} (c : Config α block) (row : Fin c.M) (second : Bool)
    (values : NativeSchedule.StreamRegisters α block) (physical : Memory c) : Memory c :=
  if c.flags.dropout && c.flags.dropoutMask then
    NativeMachineMemory.scatter c.allocations c.regions c.strides c.M c.N row.val block c.flags
      (maskRole second) (fun j => .mk .bool (values.keep j)) physical
  else physical

theorem mask_correspondence {α : Type} (c : Config α block) (row : Fin c.M) (second : Bool)
    (hx : second=Bool.true → c.flags.x1=Bool.true) (values : NativeSchedule.StreamRegisters α block)
    (physical : Memory c) (template : State α) :
    view c (storeMask c row second values physical) template=
      NativeSchedule.storeMask c.flags c.regions (inputIndex c.M row.val second) c.N block values (view c physical template) := by
  cases hd : c.flags.dropout <;> cases hm : c.flags.dropoutMask
  all_goals simp only [storeMask,NativeSchedule.storeMask,hd,hm,Bool.false_and,Bool.true_and,
    Bool.false_eq_true,ite_false,ite_true]
  all_goals first | rfl | skip
  have ha : active c.flags (maskRole second) := by cases second <;> simp_all [maskRole,active]
  have hw : NativeMachineMemory.writeRole (maskRole second) := by cases second <;> trivial
  have hvector : scalarRole (maskRole second)=Bool.false := by cases second <;> rfl
  have hoffset : ∀ col, offset c.strides c.M c.N row.val col (maskRole second)=
      (inputIndex c.M row.val second)*c.N+col := by cases second <;> intro col <;> rfl
  have he := NativeMachineMemory.scatter_commutes c.flags c.regions c.strides c.M c.N block
    c.allocations c.accepted c.storage row.val row.isLt (maskRole second) hw ha hvector
    ((inputIndex c.M row.val second)*c.N) hoffset (fun j => Cell.mk .bool (values.keep j)) physical template
  cases second <;> simpa only [view,maskRole,region,inputIndex] using he

#axiomsClean stream_correspondence
#axiomsClean mask_correspondence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeMachine
