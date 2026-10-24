import bench.repairs.tritonbench_g.cross_entropy1.Proofs.InstructionBinding

namespace VeriTile.Bench.Repairs.CrossEntropy1.Physical
open VeriTile.Triton Source Host Floating
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- Reviewable object/code/address binding for one returned coordinate.
No arithmetic event execution or output-value equality is supplied here. -/
structure LinkedReturn {σ : Type} {selection : Selection σ}
    (layout : Layout σ) (observe : σ → Nat → Nat → Byte)
    (events : ByteEvents σ) (registry : InstructionRegistry σ)
    (readWord : σ → Tensor → Nat → Floating.Word)
    (expression : Expr) (state : σ) (tensor : Tensor) (index : Nat) where
  plan : SelectedPlan selection expression
  code : LinkedCode plan registry readWord
  site : Occurrence σ
  stored : σ
  active : registry.active site
  store : registry.opcode site=some (.store (code.nativeGraph.output registry) stored tensor index)
  stable : layout.buffer state tensor.region=layout.buffer stored tensor.region
  writes : List ByteAddress
  tail : ByteTrace events stored state writes
  untouched : ∀ byte : Fin 4,
    ((layout.buffer stored tensor.region).device,
      (layout.buffer stored tensor.region).base+4*tensor.index index+byte.val)∉writes

def LinkedReturn.executed {σ : Type} {selection : Selection σ}
    {layout : Layout σ} {observe events registry trace readWord expression state tensor index}
    (binding : LinkedReturn (selection:=selection) layout observe events registry readWord expression state tensor index)
    (api : InstructionExecution registry trace) :
    SelectedReturn (selection:=selection) layout observe events trace readWord expression state tensor index :=
  ⟨binding.plan,binding.code.executed api,binding.site,binding.stored,
    api.store _ _ _ _ _ binding.active binding.store,binding.stable,binding.writes,binding.tail,binding.untouched⟩

#axiomsClean LinkedReturn.executed
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Physical
