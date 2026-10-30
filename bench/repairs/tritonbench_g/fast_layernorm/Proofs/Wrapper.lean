import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Runtime

/-! Host shape requests and the source checks. Shape success is the ordinary
Torch reshape(-1,last_dimension) contract; rank-zero and ambiguous zero-column
requests can fail without entering the device context or launching a kernel. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.Wrapper
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Preparation Layout
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical

structure ShapeInfo (x : Storage.Tensor) where
  prefixShape : List Nat
  columns : Nat
  shape : x.view.shape=prefixShape++[columns]
  positive : 0<columns

def ShapeInfo.rows (s : ShapeInfo x) := s.prefixShape.prod

theorem shape_count (s : ShapeInfo x) : x.view.shape.prod=s.rows*s.columns := by
  simp [s.shape,ShapeInfo.rows,List.prod_append]

structure MetadataAPI (api : Memory.API UInt32 Store UInt8) (neg : UInt32 → UInt32)
    (inputs : List Storage.Tensor) (Metadata : Type) where
  reshapeRows : (x : Storage.Tensor) → Effect api neg inputs Metadata (ShapeInfo x)
  reshapeOwner : ∀ x s,((reshapeRows x).call.run s).1.owner=s.owner

structure Module where
  affine : Bool
  weight : Storage.Tensor
  bias : Option Storage.Tensor
  epsilon : Compiler.Shared.HostScalar
  varianceEpsilon : Option Compiler.Shared.HostScalar

def Module.selectedEpsilon (m : Module) : Compiler.Shared.HostScalar :=
  m.varianceEpsilon.getD m.epsilon

variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {Metadata : Type}

/-- Width/count checks after admission, with the original width threshold and
all four warp-count branches. The complete tensor/scalar admission predicate
and its guard equivalence are stated in Contract and Proofs/Contract. -/
def guards (x w : Storage.Tensor) (bias : Option Storage.Tensor) (shape : ShapeInfo x) :
    Action (Host.world (Metadata:=Metadata) api inputs) (Nat × Nat) :=
  (checked .width ((Compiler.settings shape.columns).mapError (fun _ => Failure.width))).bind fun config =>
  (check (.shape .weight) (shape.columns≤w.view.shape.prod ∧
    ∀ b,bias=some b → shape.columns≤b.view.shape.prod) (.assertion (.shape .weight))).bind fun _ =>
  Action.pure config

structure Checks (x w : Storage.Tensor) (bias : Option Storage.Tensor) (shape : ShapeInfo x)
    (block warps : Nat) : Prop where
  settings : Compiler.settings shape.columns=.ok (block,warps)
  weight : shape.columns≤w.view.shape.prod
  bias : ∀ b,bias=some b → shape.columns≤b.view.shape.prod

theorem guards_success (x w : Storage.Tensor) (bias : Option Storage.Tensor) (shape : ShapeInfo x)
    (s : State Store Metadata) (block warps : Nat)
    (ok : ((guards (api:=api) (inputs:=inputs) x w bias shape).run s).result=.ok (block,warps)) :
    Checks x w bias shape block warps := by
  obtain ⟨config,hconfig,rest,_,_⟩ := bind_success _ _ _ _ ok
  obtain ⟨unit,hcheck,hret,_,_⟩ := bind_success _ _ _ _ rest
  have he : config=(block,warps) := by simpa [Action.pure] using hret
  subst config
  have hc : Compiler.settings shape.columns=.ok (block,warps) := by
    cases h : Compiler.settings shape.columns <;> simp [checked,h,Except.mapError] at hconfig ⊢
    assumption
  have ha : shape.columns≤w.view.shape.prod ∧ ∀ b,bias=some b → shape.columns≤b.view.shape.prod := by
    by_contra h
    simp [check,h] at hcheck
  exact ⟨hc,ha.1,ha.2⟩

#axiomsClean shape_count
#axiomsClean guards_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.Wrapper
