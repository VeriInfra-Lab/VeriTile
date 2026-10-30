import bench.repairs.tritonbench_g.rotary_emb.Proofs.TypedResult
import bench.repairs.tritonbench_g.rotary_emb.Proofs.Native

namespace VeriTile.Bench.Repairs.RotaryEmb.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RotaryEmbNopad
noncomputable section

instance : Inhabited Native.Node := ⟨Native.atom 0⟩

/-- Independent selected PTX evaluation, on unrestricted 32-bit words.
The source/compiler interpretation uses the same local primitive semantics. -/
def graphValue (p : Native.Instructions) (shape : Shape) (s : State Native.Node)
    (key odd : Bool) (i : Lane (layout shape)) : Cell Native.Node :=
  let l := layout shape
  let a := ((s.mem (region l key) (offset l s key Bool.false i)).read .real).value
  let b := ((s.mem (region l key) (offset l s key Bool.true i)).read .real).value
  let c0 := ((s.mem l.Cos (coefficient l s Bool.false Bool.false i)).read .real).value
  let z0 := ((s.mem l.Sin (coefficient l s Bool.true Bool.false i)).read .real).value
  let c1 := ((s.mem l.Cos (coefficient l s Bool.false Bool.true i)).read .real).value
  let z1 := ((s.mem l.Sin (coefficient l s Bool.true Bool.true i)).read .real).value
  .mk .real (Native.atom (Native.graph p a b c0 z0 c1 z1 odd))

theorem value_graph (p : Native.Instructions) (shape : Shape) (s : State Native.Node)
    (key odd : Bool) (i : Lane (layout shape)) :
    value (Native.ops p) (layout shape) s key odd i=graphValue p shape s key odd i := by
  cases odd <;> rfl

def graphBranch (p : Native.Instructions) (shape : Shape) (s : State Native.Node)
    (key : Bool) (entry : State Native.Node) : State Native.Node :=
  Pair.written (region (layout shape) key) (active (layout shape) s key)
    (offset (layout shape) s key) (graphValue p shape s key) entry

def graphTarget (p : Native.Instructions) (shape : Shape) (s : State Native.Node) : State Native.Node :=
  graphBranch p shape s Bool.true (graphBranch p shape s Bool.false s)

theorem target_graph (p : Native.Instructions) (shape : Shape) (s : State Native.Node) :
    target (Native.ops p) shape s=graphTarget p shape s := by
  have hv (key : Bool) : value (Native.ops p) (layout shape) s key=graphValue p shape s key :=
    funext fun odd => funext (value_graph p shape s key odd)
  simp only [target,branch,graphTarget,graphBranch,hv]

theorem native_successful (rest : Algebra Native.Node) (p : Native.Instructions)
    (shape : Shape) (valid : shape.Valid) (s : State Native.Node) :
    ∃ t,exec (algebra rest (Native.ops p)) (Kernels.kernel (layout shape)) s=some t ∧
      t.mem=(graphTarget p shape s).mem ∧
      (∀ key odd i,active (layout shape) s key i →
        t.mem (region (layout shape) key) (offset (layout shape) s key odd i)=
          graphValue p shape s key odd i) ∧
      (∀ r o,Outside shape s r o → t.mem r o=s.mem r o) := by
  simpa only [target_graph,value_graph] using successful rest (Native.ops p) shape valid s

#specNonCircular graphTarget avoiding [Kernels.kernel,execution,reference,target]
#axiomsClean native_successful
end
end VeriTile.Bench.Repairs.RotaryEmb.Typed
