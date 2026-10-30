import bench.tritonbench_g.fused_rotary_embedding.Proofs.Connection
import bench.tritonbench_g.fused_rotary_embedding.Proofs.Real

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
noncomputable section
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

/-- Execute every actual program through the shared quoted statement semantics,
threading its resulting memory into the next program. -/
def grid (rest : Algebra α) (ops : Ops α) (p : Parameters) :
    List Program → Mem α → Option (Mem α)
  | [],m=>some m
  | pid::ps,m=>(FP.Structural.exec (algebra rest ops) (kernel p) (state pid m)).bind
      (fun final=>grid rest ops p ps final.mem)

theorem grid_memory (rest : Algebra α) (ops : Ops α) (c : Config) (hc : c.Valid)
    (ps : List Program) (inside : ∀ pid∈ps,pid∈programs c.shape) (m : Memory α) :
    grid rest ops (parameters c.shape) ps (lift c m)=
      some (lift c (actualPrograms c (fun _=>ops) ps m)) := by
  induction ps generalizing m with
  | nil=>rfl
  | cons pid ps ih=>
    rw [grid,(execution rest ops (parameters c.shape) (state pid (lift c m))).property]
    simp only [Option.bind_some,program_memory rest ops c hc pid (inside pid (by simp))]
    exact ih (fun p hp=>inside p (by simp [hp])) _

theorem complete (rest : Algebra α) (ops : Ops α) (c : Config) (hc : c.Valid) (m : Memory α) :
    grid rest ops (parameters c.shape) (programs c.shape) (lift c m)=
      some (lift c (actualSource c (fun _=>ops) m)) :=
  grid_memory rest ops c hc _ (fun _ h=>h) m

@[simp] theorem lift_output (c : Config) (m : Memory α) (kind : Kind) (a : Nat) :
    lift c m (region kind.output) a=.mk .real (m kind.output a) := by
  cases kind <;> simp [lift,region,Kind.output,decoded]

theorem complete_real (rest : Algebra ℝ) (c : Config) (hc : c.Valid) (m : Memory ℝ) :
    ∃ final,grid rest realOps (parameters c.shape) (programs c.shape) (lift c m)=some final ∧
      (∀ kind t h d,t<c.shape.tokens → h<c.shape.heads kind → d<c.shape.width → 
        (final (region kind.output) (logicalOutput c kind t h d)).read .real=
          formula c.shape m kind t h d) ∧
      (∀ r a,¬Footprint c (decoded r) a → final r a=lift c m r a) := by
  refine ⟨lift c (actualSource c (fun _=>realOps) m),complete rest realOps c hc m,?_,?_⟩
  · intro kind t h d ht hh hd
    rw [lift_output]
    exact actual_formula c hc m kind t h d ht hh hd
  · intro r a outside
    simp only [lift]
    split_ifs <;> try rfl
    rw [(real_post c hc m).2 (decoded r) a outside]

#axiomsClean complete
#axiomsClean complete_real
#specNonCircular formula avoiding [grid,kernel,written,naturalProgram,actualSource]
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
