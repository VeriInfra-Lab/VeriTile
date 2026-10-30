import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.Grid
import bench.repairs.tritonbench_g.fused_rotary_embedding.Proofs.NativeHost

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.NativeSource
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

/-- Direct product of the source-selected scalar profiles. The Q and K
coordinates preserve their distinct multiplication/FMA operand orders. -/
abbrev Payload := Kind → Native.Node
instance : Inhabited Native.Node := ⟨Native.atom 0⟩
def ops (p : Native.Instructions) : Ops Payload where
  mul:=fun a b kind=>(Native.ops p kind).mul (a kind) (b kind)
  sub:=fun a b kind=>(Native.ops p kind).sub (a kind) (b kind)
  add:=fun a b kind=>(Native.ops p kind).add (a kind) (b kind)
def initial (m : Memory Native.Word) : Memory Payload := fun r a _=>Native.atom (m r a)
def coordinate : Region → Kind | .kcache=>.k | .vcache=>.v | _=>.q
def project (m : Memory Payload) : Memory Native.Word := fun r a=>(m r a (coordinate r)).value
@[simp] theorem coordinate_output (kind : Kind) : coordinate kind.output=kind := by cases kind <;> rfl
@[simp] theorem project_initial (m : Memory Native.Word) : project (initial m)=m := rfl

theorem pair_projection (p : Native.Instructions) (s : Shape) (w : Work)
    (m : Memory Native.Word) (high : Bool) :
    (component (pairValues s (fun _=>ops p) w (initial m)) high w.kind).value=
      Native.pairWord s p w m high := by
  cases kw : w.kind <;> cases high <;>
    simp [pairValues,Native.pairWord,kw,component,sourcePair,ops,Native.ops,Native.multiply,
      Native.subtract,Native.add,Native.graph,initial,Native.atom,Kind.input]

theorem source_selected (p : Native.Instructions) (c : Config) (hc : c.Valid)
    (m : Memory Native.Word) :
    project (actualSource c (fun _=>ops p) (initial m))=Native.selected c p m := by
  rw [actualSource_eq_run c hc]
  funext r a
  by_cases touched : ∃ w,c.shape.Active w ∧ Writes c w r a
  · obtain ⟨w,hw,hr,ha|ha⟩:=touched
    all_goals
      subst r a
      simp only [project,coordinate_output]
    · rw [run_readback c hc (fun _=>ops p) (initial m) w hw Bool.false,
        Native.selected_readback c hc p m w hw Bool.false]
      exact pair_projection p c.shape w m Bool.false
    · rw [run_readback c hc (fun _=>ops p) (initial m) w hw Bool.true,
        Native.selected_readback c hc p m w hw Bool.true]
      exact pair_projection p c.shape w m Bool.true
  · have outside : ∀ w,c.shape.Active w → ¬Writes c w r a := by simpa using touched
    simp only [project,run_frame c (fun _=>ops p) (initial m) r a outside,
      Native.selected_frame c p m r a outside]
    rfl

/-- Full native execution, observed at each output's selected profile. -/
theorem complete (rest : Algebra Payload) (p : Native.Instructions) (c : Config) (hc : c.Valid)
    (m : Memory Native.Word) :
    ∃ final,Source.grid rest (ops p) (Source.parameters c.shape) (programs c.shape)
        (Source.lift c (initial m))=some final ∧
      (∀ kind a,((final (Source.region kind.output) a).read .real kind).value=
        Native.selected c p m kind.output a) := by
  refine ⟨Source.lift c (actualSource c (fun _=>ops p) (initial m)),
    Source.complete rest (ops p) c hc (initial m),?_⟩
  intro kind a
  rw [Source.lift_output]
  have h:=congrFun (congrFun (source_selected p c hc m) kind.output) a
  simpa [project] using h

/-- The actual quoted stores and existing finite byte execution have the
same output words. This connects the byte/frame theorem to source execution. -/
theorem complete_bytes (rest : Algebra Payload) (c : Checked.Call) (callerDevice : Nat)
    (p : Native.Instructions) (m : Arena.Memory)
    (admitted : inputChecks c (Arena.mirror c m)=Bool.true) :
    ∃ final result,
      Source.grid rest (ops p) (Source.parameters c.shape) (programs c.shape)
        (Source.lift (Checked.host c (Arena.mirror c m)).config
          (initial (Physical.load (Arena.mirror c m))))=some final ∧
      Binding.invoke c callerDevice p m=some result ∧
      Binding.NativeResult c callerDevice p p m result ∧
      (∀ kind a,((final (Source.region kind.output) a).read .real kind).value=
        Physical.load (Arena.mirror c result.memory) kind.output a) := by
  have checked:=original_checks c (Arena.mirror c m) admitted
  have valid:=Checked.config_valid c (Arena.mirror c m) checked
  obtain ⟨final,executed,words⟩:=complete rest p (Checked.host c (Arena.mirror c m)).config valid
    (Physical.load (Arena.mirror c m))
  obtain ⟨result,invoked,properties⟩:=Binding.complete c callerDevice p m checked
  refine ⟨final,result,executed,invoked,properties,?_⟩
  intro kind a
  rw [words,properties.source,Native.source_selected _ valid]

#axiomsClean source_selected
#axiomsClean complete_bytes
#specNonCircular Native.graph avoiding [Source.kernel,Source.grid,ops,actualSource,Native.selected]
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.NativeSource
