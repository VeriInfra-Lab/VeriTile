import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.CompilerStores

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.StorageCompiler
open VeriTile.Bench.Repairs.LayerNormFwd
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper Compiler
noncomputable section
set_option maxHeartbeats 3000000

structure Profile where
  y : DType
  residual : DType

def Profile.dtype (profile : Profile) : Destination → DType
  | .y _ | .y1 _ => profile.y
  | .residualOut _ => profile.residual
  | .mean | .rstd => .fp32
  | .mask _ _ => .bool

/-- Total local store conversions on a unified opaque scalar-storage carrier.
The carrier α must represent every accepted destination encoding; it is not a
32-bit-word type. The legacy Cell.real tag selects this carrier, not an FP32
storage width. Thus accepted FP64/integer/bool outputs can retain their actual
encodings; descriptors determine widths. Seed/mask roles use their own tags.
No Real interpretation or finite-only side condition is used; nonfinite and
signed-zero inputs stay in the total callback domain. FP32 stores are identities.
Every other destination names its actual requested DType. -/
structure StoreCasts (α : Type) where
  convertFP32 : DType → α → α
  fp32 : ∀ value, convertFP32 .fp32 value=value

inductive Operand where
  | convertFP32 (dtype : DType) (value : Code)
  | boolGreater (left right : Code)

def Operand.erase : Operand → StoreOperand
  | .convertFP32 _ code => .real32 code
  | .boolGreater a b => .boolGreater a b

def Operand.ofBase (dtype : DType) : StoreOperand → Operand
  | .real32 code => .convertFP32 dtype code
  | .boolGreater a b => .boolGreater a b

def Operand.eval (ops : NativeRow.Ops α) (casts : StoreCasts α) (x : NativeRow.Inputs α)
    (M row : Nat) (eps p : α) : Operand → Cell α
  | .convertFP32 dtype code => .mk .real (casts.convertFP32 dtype ((code.expression x M row eps p).eval ops))
  | .boolGreater a b => .mk .bool (ops.greater
      ((a.expression x M row eps p).eval ops) ((b.expression x M row eps p).eval ops))

def sourceStore (profile : Profile) (f : Flags) (N : Nat) (code : Root → Code) (destination : Destination) : Option Operand :=
  (Compiler.sourceStore f N code destination).map (Operand.ofBase (profile.dtype destination))

/-- Actual emitted typed stores, with their local conversion opcode and source
mask/target/register bindings. The existing graph artifact is reused; no
new arithmetic equation is assumed. -/
structure Artifact (profile : Profile) (f : Flags) (N block : Nat) where
  graph : Compiler.Artifact f N block
  stores : Destination → Option Operand
  storeOperands : ∀ destination, stores destination=sourceStore profile f N graph.code destination

theorem Artifact.erased (artifact : Artifact profile f N block) (destination : Destination) :
    (artifact.stores destination).map Operand.erase=artifact.graph.stores destination := by
  rw [artifact.storeOperands,artifact.graph.storeOperands]
  cases h : Compiler.sourceStore f N artifact.graph.code destination with
  | none => simp [sourceStore,h]
  | some operand => cases operand <;> simp [sourceStore,h,Operand.ofBase,Operand.erase]

/-- Input-only reference for explicit storage conversions. Execution below
applies these conversions at each store, before any readback. -/
def Artifact.observe {α : Type} [Inhabited α] (artifact : Artifact profile f N block)
    (ops : NativeRow.Ops α) (casts : StoreCasts α) (r : Regions) (stride : Strides)
    (M : Nat) (eps p : α) (s : State α) : TypedSource.Result α M N (HostReturn.seedCount f M) :=
  let x := NativeSchedule.snapshot r stride s
  let value := fun row root => ((artifact.graph.code root).expression x M row eps p).eval ops
  let keep := fun row second col => ops.greater (value row (.random second col)) (value row .dropoutP)
  { y := fun i j => .mk .real (casts.convertFP32 profile.y (value i.val (.y j.val)))
    y1 := if f.w1 then some (fun i j => .mk .real (casts.convertFP32 profile.y (value i.val (.y1 j.val)))) else none
    mean := if f.rms then none else some (fun i => .mk .real (value i.val .mean))
    rstd := fun i => .mk .real (value i.val .rstd)
    residual := if f.residualOut then
      (fun i j => .mk .real (casts.convertFP32 profile.residual (value i.val (.residual j.val))))
      else (fun i j => s.mem r.x (i.val*stride.x+j.val))
    seeds := if f.dropout then some (fun i => s.mem r.seeds i.val) else none
    mask := if f.dropout && f.dropoutMask then some (fun i j => .mk .bool (keep i.val Bool.false j.val)) else none
    mask1 := if f.dropout && f.dropoutMask && f.x1 then some (fun i j => .mk .bool (keep i.val Bool.true j.val)) else none }

theorem Artifact.observe_store {α : Type} [Inhabited α] (artifact : Artifact profile f N block)
    (ops : NativeRow.Ops α) (casts : StoreCasts α) (r : Regions) (stride : Strides)
    (M : Nat) (eps p : α) (s : State α) (row : Fin M) (destination : Destination) :
    (artifact.stores destination).map
      (Operand.eval ops casts (NativeSchedule.snapshot r stride s) M row.val eps p)=
      Compiler.storeCoordinate f (artifact.observe ops casts r stride M eps p s) row destination := by
  rw [artifact.storeOperands]
  cases destination with
  | y col => by_cases h : col<N <;> simp [sourceStore,Compiler.sourceStore,storeCoordinate,Artifact.observe,h,Operand.ofBase,Operand.eval,Profile.dtype]
  | y1 col =>
    cases hw : f.w1 <;> by_cases h : col<N
    all_goals simp [sourceStore,Compiler.sourceStore,storeCoordinate,Artifact.observe,hw,h,Operand.ofBase,Operand.eval,Profile.dtype]
  | residualOut col =>
    cases hr : f.residualOut <;> by_cases h : col<N
    all_goals simp [sourceStore,Compiler.sourceStore,storeCoordinate,Artifact.observe,hr,h,Operand.ofBase,Operand.eval,Profile.dtype]
  | mean => cases hr : f.rms <;> simp [sourceStore,Compiler.sourceStore,storeCoordinate,Artifact.observe,hr,Operand.ofBase,Operand.eval,Profile.dtype,casts.fp32]
  | rstd => simp [sourceStore,Compiler.sourceStore,storeCoordinate,Artifact.observe,Operand.ofBase,Operand.eval,Profile.dtype,casts.fp32]
  | mask second col =>
    cases second <;> cases hd : f.dropout <;> cases hm : f.dropoutMask <;> cases hx : f.x1 <;> by_cases h : col<N
    all_goals simp [sourceStore,Compiler.sourceStore,storeCoordinate,Artifact.observe,hd,hm,hx,h,Operand.ofBase,Operand.eval,Profile.dtype]

#axiomsClean Artifact.erased
#axiomsClean Artifact.observe_store
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.StorageCompiler
