import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.ForwardValues

namespace VeriTile.Bench.Repairs.LayerNormTriton.Host
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage Preparation Layout ReceiptAction
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {inputs : List Storage.Tensor}
  {Metadata : Type} {platform : Compiler.Platform}

theorem config_state (N : Nat) (s : State Store Metadata) : ((config (api:=api) (inputs:=inputs) N).run s).state=s := by
  unfold config
  split <;> rfl

structure Call (lib : APIHost api neg inputs Metadata) (service : ForwardService platform api neg inputs Metadata)
    (identity : Nat) (x w b : Storage.Tensor) (normalized : List Nat) (eps : Compiler.Shared.HostScalar)
    (initial : State Store Metadata) where
  shape : Geometry.Shape x.view.shape w.view.shape b.view.shape
  validated : validate x w b normalized=.ok shape
  B : Nat
  settings : Compiler.settings shape.columns=.ok (B,Geometry.warps B)
  entered : ((lib.enter x.view.device).call.run initial).2=.ok ()
  body : BodyRoute lib service identity x w b shape B settings eps ((lib.enter x.view.device).call.run initial).1
  left : ((lib.leave initial.owner).call.run body.final).2=.ok ()

variable {lib : APIHost api neg inputs Metadata} {service : ForwardService platform api neg inputs Metadata}
  {identity : Nat} {x w b : Storage.Tensor} {normalized : List Nat} {eps : Compiler.Shared.HostScalar}
  {initial : State Store Metadata}

def Call.final (r : Call lib service identity x w b normalized eps initial) :=
  ((lib.leave initial.owner).call.run r.body.final).1

def Call.realFinal (r : Call lib service identity x w b normalized eps initial)
    (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) :=
  ((lib.leave initial.owner).transition r.body.final).shadow
    (r.body.realFinal (((lib.enter x.view.device).transition initial).shadow m) epsilon)

theorem public_route (out : Dense)
    (ok : ((forward lib service identity x w b normalized eps).run initial).result=.ok out) :
    ∃ r : Call lib service identity x w b normalized eps initial,
      r.body.output=out ∧ ((forward lib service identity x w b normalized eps).run initial).state=r.final := by
  obtain ⟨shape,hshape,rest,hs0,_⟩ := bind_success _ _ _ _ ok
  have hv : validate x w b normalized=.ok shape := hshape
  change (((config shape.columns).bind _).run initial).result = .ok out at rest
  obtain ⟨block,hblock,rest2,hs1,_⟩ := bind_success _ _ _ _ rest
  rw [config_state] at rest2 hs1
  obtain ⟨snapshot,hsnapshot,hdevice,hs2,_⟩ := bind_success _ _ _ _ rest2
  have heq : snapshot=initial := by simpa [getState] using hsnapshot.symm
  subst snapshot
  obtain ⟨he,hbody,hl,hstate⟩ := device_success _ _ _ initial out hdevice
  obtain ⟨br,hout,hbodyState⟩ := body_route out hbody
  rw [hbodyState] at hl hstate
  let r : Call lib service identity x w b normalized eps initial := ⟨shape,hv,block.val,block.property,he,br,hl⟩
  refine ⟨r,hout,?_⟩
  exact hs0.trans (hs1.trans (hs2.trans hstate))

theorem returned_identity (r : Call lib service identity x w b normalized eps initial) :
    r.body.output.tensor.backing=r.body.preparation.data.y.tensor.backing ∧
    r.body.output.tensor.view.shape=x.view.shape ∧ r.final.owner=initial.owner :=
  ⟨(body_storage r.body).1,(body_storage r.body).2,lib.leaveOwner initial.owner r.body.final r.left⟩

theorem saved_context (r : Call lib service identity x w b normalized eps initial) :
    lib.context r.final.metadata identity=some r.body.context := by
  rw [Call.final,lib.leaveContext]
  exact body_saved_context r.body

theorem returned_native (r : Call lib service identity x w b normalized eps initial)
    (i : Fin r.body.preparation.data.y.count) :
    Storage.logical api neg r.final.memory r.body.output.tensor i.val=r.body.launch.nativeValue i.val := by
  have hv := lib.viewed r.body.preparation.data.y x.view.shape r.body.sSave r.body.output r.body.viewed
  have hc : i.val<r.body.output.tensor.view.shape.prod := by rw [←r.body.output.countEq,hv.count]; exact i.isLt
  exact ((lib.leave initial.owner).transition r.body.final |>.values _ hv.present i.val hc).trans (body_native r.body i)

theorem returned_real (r : Call lib service identity x w b normalized eps initial)
    (xPresent : initial.live x) (wPresent : initial.live w) (bPresent : initial.live b)
    (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) (i : Fin r.body.preparation.data.y.count) :
    Storage.realLogical (r.realFinal m epsilon) r.body.output.tensor i.val=
      RealProof.Spec.output r.shape.columns (logicalRow m x r.shape.columns (i.val/r.shape.columns))
        (Storage.realLogical m w) (Storage.realLogical m b) epsilon (i.val%r.shape.columns) := by
  let tr := (lib.enter x.view.device).transition initial
  have hv := lib.viewed r.body.preparation.data.y x.view.shape r.body.sSave r.body.output r.body.viewed
  have hc : i.val<r.body.output.tensor.view.shape.prod := by rw [←r.body.output.countEq,hv.count]; exact i.isLt
  have ht := (lib.leave initial.owner).transition r.body.final |>.realValues
    (r.body.realFinal (tr.shadow m) epsilon) _ hv.present i.val hc
  change Storage.realLogical (r.realFinal m epsilon) r.body.output.tensor i.val=_ at ht
  rw [ht,body_original_real r.body (tr.live w wPresent) (tr.live b bPresent) (tr.shadow m) epsilon i]
  have hi : i.val<r.shape.rows*r.shape.columns := by simpa only [←r.body.layout.yCount] using i.isLt
  have hn : 0<r.shape.columns := by
    have hp : 0<r.shape.rows*r.shape.columns := lt_of_le_of_lt (Nat.zero_le _) hi
    exact Nat.pos_of_mul_pos_left hp
  have hcol := Nat.mod_lt i.val hn
  have hrow := (Nat.div_lt_iff_lt_mul hn).mpr hi
  apply VeriTile.Bench.Repairs.FastLayernorm.RealProof.output_congr _ _ _ _ _ _ _ _ _ hcol
  · intro col hcol
    have hoff : i.val/r.shape.columns*r.shape.columns+col<x.view.shape.prod := by
      rw [(Geometry.shape_product r.shape).1]
      have hmul := Nat.mul_le_mul_right r.shape.columns (show i.val/r.shape.columns+1≤r.shape.rows by omega)
      nlinarith
    exact tr.realValues m x xPresent _ hoff
  · exact tr.realValues m w wPresent _ (by simpa only [(Geometry.shape_product r.shape).2.1] using hcol)
  · exact tr.realValues m b bPresent _ (by simpa only [(Geometry.shape_product r.shape).2.2] using hcol)

#axiomsClean public_route
#axiomsClean returned_identity
#axiomsClean returned_native
#axiomsClean returned_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.Host
