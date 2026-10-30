import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.TypedNative
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.SetupCalls
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.HostBlock

/-! Source-level wrapper contract. Ordinary successful tensor-library returns
supply coordinate-preserving FP32 resolve/pack views and fresh allocation
metadata. Kernel execution, output readback, native addresses and frames are
conclusions. Allocation/compiler failure remains an ordered API outcome. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.TypedHost
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.LayerNormFwd
open TypedPhysical (Physical pull)
open Address (Allocation signed64)
open Verified.PhysicalCells (Layout Role width FreshBytes)
open Verified.NativeOffsets
open Verified.HostSetup (Call calls beforeEmpty nonemptyChecks)
open TypedExecution
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000

open HostShape (blockD blockD_covers OutputDescriptor)

structure Request (α : Type) where
  batches : Nat
  heads : Nat
  context : Nat
  headDim : Nat
  /-- Initial logical FP32 Q/K/V and packed-flat scale coordinates. -/
  value : Role → Nat → Nat → Nat → Nat → α

def Request.active {α : Type} (r : Request α) (z h n d : Nat) : Role → Prop
  | .queryScale => z=0 ∧ h=0 ∧ d=0 ∧ n<r.batches*r.heads*((r.context+127)/128)
  | .keyScale => z=0 ∧ h=0 ∧ d=0 ∧ n<r.batches*r.heads*((r.context+63)/64)
  | _ => z<r.batches ∧ h<r.heads ∧ n<r.context ∧ d<r.headDim

/-- Ordinary input tensor views of the original caller memory, including lazy
negative views. `negate` is the tensor backend's logical sign operation. -/
structure CallerInputs {α : Type} [Inhabited α] (request : Request α) (a : Layout)
    (caller : Physical α a.toLayout) where
  owned : Fin a.capacity → Prop
  cell : Role → Nat → Nat → Nat → Nat → Fin a.capacity
  negative : Role → Bool
  negate : α → α
  ownedInputs : ∀ z h n d r, r≠Role.output → request.active z h n d r → owned (cell r z h n d)
  fp32 : ∀ z h n d r, r≠Role.output → request.active z h n d r →
    ∃ x, caller (cell r z h n d)=Cell.mk .fp32 x
  logical : ∀ z h n d r, r≠Role.output → request.active z h n d r →
    request.value r z h n d =
      (if negative r then negate ((caller (cell r z h n d)).read .fp32)
       else (caller (cell r z h n d)).read .fp32)

def outputDescriptor (p : Config) (base : Role → Nat) : OutputDescriptor :=
  ⟨p.Out,base .output,p.batches,p.heads,p.context,p.headDim,
    p.outputBatch,p.outputHead,p.outputRow,p.outputDim,.bf16⟩

/-- Metadata returned by ordinary successful resolve_neg, scale packing and
fresh output allocation. Preparation writes only its fresh allocation set;
all original caller cells outside that set retain their exact typed values. -/
structure Prepared {α : Type} [Inhabited α] (request : Request α) (a : Layout) (caller : Physical α a.toLayout) where
  original : CallerInputs request a caller
  p : Config
  shape : p.batches=request.batches ∧ p.heads=request.heads ∧
    p.context=request.context ∧ p.headDim=request.headDim
  stage : p.stage=3
  base : Role → Nat
  views : Views a p base
  returnedOutput : OutputDescriptor
  outputBinding : returnedOutput=outputDescriptor p base
  alloc : Role → Allocation
  allocated : Allocated p base alloc
  nonoverlap : NonOverlapping p
  fresh : FreshBytes a (fun k => Window p k.1 k.2)
  initial : Physical α a.toLayout
  preparationWrites : Fin a.capacity → Prop
  preparationFrame : ∀ cell, ¬preparationWrites cell → initial cell=caller cell
  preparationFresh : ∀ cell, original.owned cell → ¬preparationWrites cell
  outputFresh : ∀ cell, original.owned cell → ∀ k, Window p k.1 k.2 → a.address k≠cell.val
  inputs : ∀ z h n d r, r≠Role.output → roleActive p z h n d r →
    TypedPhysical.read a.toLayout initial (roleKey p z h n d r)=
      Cell.mk .fp32 (request.value r z h n d)

inductive Outcome {α : Type} [Inhabited α] {request : Request α} {a : Layout} {caller : Physical α a.toLayout}
    (ret : Prepared request a caller) where
  | failed (call : Call)
  | empty (returned : OutputDescriptor) (same : returned=ret.returnedOutput)
      (h : ret.p.batches*ret.p.heads*ret.p.context*ret.p.headDim=0)
  | launch (returned : OutputDescriptor) (same : returned=ret.returnedOutput)
      (nonempty : ret.p.batches*ret.p.heads*ret.p.context*ret.p.headDim≠0)
      (feasible : NativeLaunch.CudaLaunchable (launchShape ret.p) (blockD ret.p.headDim))

def setup {α : Type} [Inhabited α] {request : Request α} {a : Layout} {caller : Physical α a.toLayout}
    (ret : Prepared request a caller) (ok : Call → Bool) : Outcome ret :=
  match calls ok beforeEmpty with
  | .error c => .failed c
  | .ok _ =>
    if he : ret.p.batches*ret.p.heads*ret.p.context*ret.p.headDim=0 then .empty ret.returnedOutput rfl he else
      match calls ok nonemptyChecks with
      | .error c => .failed c
      | .ok _ =>
        if hf : NativeLaunch.CudaLaunchable (launchShape ret.p) (blockD ret.p.headDim) then
          if ok .compileAndLaunch then .launch ret.returnedOutput rfl he hf else .failed .compileAndLaunch
        else .failed .compileAndLaunch

/-- Every successful launch uses the same actual allocation metadata as the
source views and returned output. All helper and scale paths are included. -/
def NativeCertificate {α : Type} [Inhabited α] {request : Request α} {a : Layout} {caller : Physical α a.toLayout}
    (ret : Prepared request a caller) : Prop :=
  (∀ z h n d r, roleActive ret.p z h n d r →
    machineOffset ret.p z h n d r=roleOffset ret.p z h n d r ∧
    signed64 (machineOffset ret.p z h n d r)=(roleOffset ret.p z h n d r : Int) ∧
    bytePointer (ret.alloc r).base (ret.alloc r).width (machineOffset ret.p z h n d r)=
      a.address (roleKey ret.p z h n d r)) ∧
  (∀ z h lo it lane d,
    (roleActive ret.p z h (lo+it*64+lane) d .key →
      bytePointer (ret.alloc .key).base (ret.alloc .key).width
        (keyValue ret.p.keyBatch ret.p.keyHead ret.p.keyRow ret.p.keyDim z h lane d lo it)=
        a.address (roleKey ret.p z h (lo+it*64+lane) d .key)) ∧
    (roleActive ret.p z h (lo+it*64+lane) d .value →
      bytePointer (ret.alloc .value).base (ret.alloc .value).width
        (keyValue ret.p.valueBatch ret.p.valueHead ret.p.valueRow ret.p.valueDim z h lane d lo it)=
        a.address (roleKey ret.p z h (lo+it*64+lane) d .value))) ∧
  (∀ head block,
    (head*((ret.p.context+127)/128)+block<ret.p.batches*ret.p.heads*((ret.p.context+127)/128) →
      bytePointer (ret.alloc .queryScale).base (ret.alloc .queryScale).width
        (packedScale head ((ret.p.context+127)/128) block)=
        a.address (roleKey ret.p 0 0 (head*((ret.p.context+127)/128)+block) 0 .queryScale)) ∧
    (head*((ret.p.context+63)/64)+block<ret.p.batches*ret.p.heads*((ret.p.context+63)/64) →
      bytePointer (ret.alloc .keyScale).base (ret.alloc .keyScale).width
        (packedScale head ((ret.p.context+63)/64) block)=
        a.address (roleKey ret.p 0 0 (head*((ret.p.context+63)/64)+block) 0 .keyScale))) ∧
  (∀ head lower iterations,
    head*((ret.p.context+63)/64)+(lower+iterations*64)/64<ret.p.batches*ret.p.heads*((ret.p.context+63)/64) →
    bytePointer (ret.alloc .keyScale).base (ret.alloc .keyScale).width
      (advance (packedScale head ((ret.p.context+63)/64) (lower/64)) (Address.wrap64 1) iterations)=
      a.address (roleKey ret.p 0 0 (head*((ret.p.context+63)/64)+(lower+iterations*64)/64) 0 .keyScale)) ∧
  (NativeLaunch.CudaLaunchable (launchShape ret.p) (blockD ret.p.headDim) →
    ControlCertificate ret.p (blockD ret.p.headDim))

theorem native_certificate {α : Type} [Inhabited α] {request : Request α} {a : Layout} {caller : Physical α a.toLayout}
    (ret : Prepared request a caller) : NativeCertificate ret :=
  ⟨native_address ret.p a ret.base ret.views ret.alloc ret.allocated,
    native_helper ret.p a ret.base ret.views ret.alloc ret.allocated,
    native_packed ret.p a ret.base ret.views ret.alloc ret.allocated,
    native_key_scale_helper ret.p a ret.base ret.views ret.alloc ret.allocated,
    control_certificate ret.p (blockD ret.p.headDim)⟩

/-- Complete successful public return, including ordinary setup order, actual
source grids, initial logical-input reference, native pointers, and caller frame. -/
theorem successful_return {α : Type} [Inhabited α] (R : FPEquivProofs.Rules)
    (A : Algebra α) (D : FP.Guarded.Domain α) (hM : FP.Guarded.Models R.assumptions A D)
    (request : Request α) (a : Layout) (caller : Physical α a.toLayout)
    (ret : Prepared request a caller) (template : State α) (ok : Call → Bool)
    (hbefore : calls ok beforeEmpty=.ok ()) (hscales : calls ok nonemptyChecks=.ok ())
    (hcompile : ok .compileAndLaunch=Bool.true)
    (hne : ret.p.batches*ret.p.heads*ret.p.context*ret.p.headDim≠0)
    (hlaunch : NativeLaunch.CudaLaunchable (launchShape ret.p) (blockD ret.p.headDim))
    (support : DotSupport A (blockD ret.p.headDim))
    (hd : ∀ idx : ProgramIndex ret.p, ∀ site∈(program ret.p (blockD ret.p.headDim) Bool.false).domain,
      site.Holds A D (atProgram ret.p (pull a.toLayout ret.initial template) idx)) :
    setup ret ok=.launch ret.returnedOutput rfl hne hlaunch ∧ NativeCertificate ret ∧
    (ret.p.context≤274877906816 ∧ ret.p.batches*ret.p.heads≤65535 ∧ blockD ret.p.headDim≤1048576) ∧
    ∃ final, ∃ physical : Physical α a.toLayout,
      Nonempty (GridRun A ret.p (blockD ret.p.headDim) Bool.false (pull a.toLayout ret.initial template) final) ∧
      Nonempty (GridRun A ret.p (blockD ret.p.headDim) Bool.true (pull a.toLayout ret.initial template) final) ∧
      pull a.toLayout physical final=final ∧
      (∀ c : Coordinate ret.p, physical (outputCell a ret.p ret.base ret.views c)=
        reference A ret.p (blockD ret.p.headDim) support (pull a.toLayout ret.initial template) c) ∧
      (∀ cell : Fin a.capacity, ¬ret.preparationWrites cell →
        (∀ k, Window ret.p k.1 k.2 → a.address k≠cell.val) → physical cell=caller cell) ∧
      (∀ cell : Fin a.capacity, ret.original.owned cell → physical cell=caller cell) := by
  refine ⟨by simp [setup,hbefore,hscales,hcompile,hne,hlaunch],native_certificate ret,
    ⟨NativeLaunch.context _ _ hlaunch,hlaunch.gridY,hlaunch.arange⟩,?_⟩
  obtain ⟨final,physical,ho,hr,hpull,hread,hframe⟩ := physical_grid_equivalence R A D hM ret.p
    (blockD ret.p.headDim) support a ret.base ret.views ret.initial template (blockD_covers _)
    ret.nonoverlap ret.fresh hd
  refine ⟨final,physical,ho,hr,hpull,hread,fun cell hp hw =>
    (hframe cell hw).trans (ret.preparationFrame cell hp),?_⟩
  intro cell hc
  exact (hframe cell (ret.outputFresh cell hc)).trans
    (ret.preparationFrame cell (ret.preparationFresh cell hc))

/-- Preparation's actual source readback equals the caller's initial logical
view coordinates, including lazy sign and packed scale order. -/
theorem initial_coordinates {α : Type} [Inhabited α] {request : Request α} {a : Layout}
    {caller : Physical α a.toLayout} (ret : Prepared request a caller)
    (z h n d : Nat) (r : Role) (hne : r≠Role.output) (hr : roleActive ret.p z h n d r) :
    TypedPhysical.read a.toLayout ret.initial (roleKey ret.p z h n d r)=
      Cell.mk .fp32 (if ret.original.negative r then
        ret.original.negate ((caller (ret.original.cell r z h n d)).read .fp32)
        else (caller (ret.original.cell r z h n d)).read .fp32) := by
  rw [ret.inputs z h n d r hne hr]
  have ha : request.active z h n d r := by
    cases r <;> simpa only [Request.active,roleActive,ret.shape.1,ret.shape.2.1,
      ret.shape.2.2.1,ret.shape.2.2.2] using hr
  rw [ret.original.logical z h n d r hne ha]

theorem returned_output {α : Type} [Inhabited α] {request : Request α} {a : Layout}
    {caller : Physical α a.toLayout} (ret : Prepared request a caller) :
    ret.returnedOutput=outputDescriptor ret.p ret.base := ret.outputBinding

theorem empty_return {α : Type} [Inhabited α] {request : Request α} {a : Layout}
    {caller : Physical α a.toLayout} (ret : Prepared request a caller) (ok : Call → Bool)
    (hbefore : calls ok beforeEmpty=.ok ())
    (he : ret.p.batches*ret.p.heads*ret.p.context*ret.p.headDim=0) :
    setup ret ok=.empty ret.returnedOutput rfl he ∧ IsEmpty (Coordinate ret.p) ∧
      (∀ cell, ret.original.owned cell → ret.initial cell=caller cell) := by
  refine ⟨by simp [setup,hbefore,he],⟨?_⟩,fun cell hc =>
    ret.preparationFrame cell (ret.preparationFresh cell hc)⟩
  intro c
  have hB := c.1.isLt
  have hH := c.2.1.isLt
  have hN := c.2.2.1.isLt
  have hD := c.2.2.2.isLt
  have hp := Nat.mul_pos (Nat.mul_pos (Nat.mul_pos (by omega : 0<ret.p.batches)
    (by omega : 0<ret.p.heads)) (by omega : 0<ret.p.context)) (by omega : 0<ret.p.headDim)
  omega

theorem infeasible_failure {α : Type} [Inhabited α] {request : Request α} {a : Layout}
    {caller : Physical α a.toLayout} (ret : Prepared request a caller) (ok : Call → Bool)
    (hbefore : calls ok beforeEmpty=.ok ()) (hscales : calls ok nonemptyChecks=.ok ())
    (hne : ret.p.batches*ret.p.heads*ret.p.context*ret.p.headDim≠0)
    (hf : ¬NativeLaunch.CudaLaunchable (launchShape ret.p) (blockD ret.p.headDim)) :
    setup ret ok=.failed .compileAndLaunch := by simp [setup,hbefore,hscales,hne,hf]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.TypedHost
