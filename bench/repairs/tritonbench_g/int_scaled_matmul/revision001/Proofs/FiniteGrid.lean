import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.FiniteSource

namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.FiniteGrid
open VeriTile.Triton FP.Structural
open FiniteExecution FiniteIndex GridExecution
noncomputable section
set_option maxHeartbeats 4000000
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def intOne (x : Launch) (pid : Nat) (s : State Word) := intProgram x (atProgram s pid)
def intGrid (x : Launch) (s : State Word) := sequence (intOne x) (List.range (count x.M x.N x.config)) s
def scaledOne (cfg : FP.Config) (u : Nat → Word → Int) (x : Launch) (scale : RegionName) (pid : Nat) (s : State Word) :=
  scaledProgram cfg u x scale (atProgram s pid)
def scaledGrid (cfg : FP.Config) (u : Nat → Word → Int) (x : Launch) (scale : RegionName) (s : State Word) :=
  sequence (scaledOne cfg u x scale) (List.range (count x.M x.N x.config)) s

theorem int_single (rest : Algebra Word) (x : Launch)
    (legal : Contract.ConfigContract x.config) (env : Contract.IndexEnvelope x.M x.N x.K x.config)
    (geometry : Geometry.Valid (BlockWord.geometry (blockParameters x)))
    (ha : x.a≠x.c) (hb : x.b≠x.c)
    (initial current : State Word) (outside : Outside x.c initial current)
    (pid : Nat) (hp : pid < count x.M x.N x.config) :
    ∃ final,intOne x pid current=some final ∧
      Post x.c (Geometry.address (BlockWord.geometry (blockParameters x)))
        (fun i : BlockWord.LogicalIndex (blockParameters x) =>
          Cell.mk .int (BlockWord.logicalDot (blockParameters x) initial i.1.val i.2.val))
        (Geometry.Owns (BlockWord.geometry (blockParameters x)) pid) current final := by
  obtain ⟨actual,typed,he,ht,hm⟩ := FiniteSource.unscaled_program_source rest x legal env (atProgram current pid) hp
  obtain ⟨other,ho,post⟩ := BlockWord.single_logical rest (blockParameters x) geometry
    (Contract.positive_tiles x.config legal).2.2.1 ha hb initial current outside pid
  have same : typed=other := Option.some.inj (ht.symm.trans ho)
  subst other
  refine ⟨actual,he,?_⟩
  simpa only [Post,hm] using post

theorem int_correct (rest : Algebra Word) (x : Launch)
    (legal : Contract.ConfigContract x.config) (env : Contract.IndexEnvelope x.M x.N x.K x.config)
    (geometry : Geometry.Valid (BlockWord.geometry (blockParameters x)))
    (ha : x.a≠x.c) (hb : x.b≠x.c) (initial : State Word) :
    ∃ final,intGrid x initial=some final ∧
      Post x.c (Geometry.address (BlockWord.geometry (blockParameters x)))
        (fun i : BlockWord.LogicalIndex (blockParameters x) =>
          Cell.mk .int (BlockWord.logicalDot (blockParameters x) initial i.1.val i.2.val))
        (fun _ => True) initial final := by
  let g := BlockWord.geometry (blockParameters x)
  obtain ⟨final,he,values,frame⟩ := compose x.c (Geometry.address g) geometry.writable
    (fun i : Geometry.Cell g => Cell.mk .int (BlockWord.logicalDot (blockParameters x) initial i.1.val i.2.val))
    (Geometry.Owns g) (intOne x) initial (count x.M x.N x.config)
    (fun pid hp current hout => int_single rest x legal env geometry ha hb initial current hout pid hp)
    (List.range (count x.M x.N x.config)) (fun _ hp => List.mem_range.mp hp) initial (fun _ _ _ => rfl)
  refine ⟨final,he,?_,?_⟩
  · intro i _
    obtain ⟨pid,hp,owns⟩ := Geometry.covers g geometry i
    exact values i ⟨pid,List.mem_range.mpr hp,owns⟩
  · intro r o outside
    exact frame r o (outside.imp id (fun h i _ => h i trivial))

theorem even_exact (x : Launch) (scale : RegionName) (legal : Contract.ConfigContract x.config) :
    (scaledParameters x scale).even=Bool.true →
      (scaledParameters x scale).inner=(scaledParameters x scale).steps*(scaledParameters x scale).blockInner := by
  intro he
  change decide (x.K%x.config.blockK=0)=Bool.true at he
  exact LogicalSum.ceil_exact _ _ (Contract.positive_tiles x.config legal).2.2.1 (of_decide_eq_true he)

theorem scaled_single (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (x : Launch) (scale : RegionName) (legal : Contract.ConfigContract x.config)
    (env : Contract.IndexEnvelope x.M x.N x.K x.config)
    (geometry : Geometry.Valid (ScaledWord.geometry (scaledParameters x scale)))
    (ha : x.a≠x.c) (hb : x.b≠x.c) (hs : scale≠x.c)
    (initial current : State Word) (outside : Outside x.c initial current)
    (pid : Nat) (hp : pid < count x.M x.N x.config) :
    ∃ final,scaledOne cfg u x scale pid current=some final ∧
      Post x.c (Geometry.address (ScaledWord.geometry (scaledParameters x scale)))
        (fun i : ScaledWord.LogicalIndex (scaledParameters x scale) =>
          Cell.mk .int (TotalScaled.logicalResult cfg u (scaledParameters x scale) initial i.1.val i.2.val))
        (Geometry.Owns (ScaledWord.geometry (scaledParameters x scale)) pid) current final := by
  obtain ⟨actual,typed,he,ht,hm⟩ := FiniteSource.scaled_program_source rest cfg u x scale legal env (atProgram current pid) hp
  obtain ⟨other,ho,post⟩ := TotalScaled.single_logical rest cfg u (scaledParameters x scale) geometry
    (even_exact x scale legal) (LogicalSum.ceil_cover x.K x.config.blockK (Contract.positive_tiles x.config legal).2.2.1)
    ha hb hs initial current outside pid
  have same : typed=other := Option.some.inj (ht.symm.trans ho)
  subst other
  refine ⟨actual,he,?_⟩
  simpa only [Post,hm] using post

theorem scaled_correct (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (x : Launch) (scale : RegionName) (legal : Contract.ConfigContract x.config)
    (env : Contract.IndexEnvelope x.M x.N x.K x.config)
    (geometry : Geometry.Valid (ScaledWord.geometry (scaledParameters x scale)))
    (ha : x.a≠x.c) (hb : x.b≠x.c) (hs : scale≠x.c) (initial : State Word) :
    ∃ final,scaledGrid cfg u x scale initial=some final ∧
      Post x.c (Geometry.address (ScaledWord.geometry (scaledParameters x scale)))
        (fun i : ScaledWord.LogicalIndex (scaledParameters x scale) =>
          Cell.mk .int (TotalScaled.logicalResult cfg u (scaledParameters x scale) initial i.1.val i.2.val))
        (fun _ => True) initial final := by
  let g := ScaledWord.geometry (scaledParameters x scale)
  obtain ⟨final,he,values,frame⟩ := compose x.c (Geometry.address g) geometry.writable
    (fun i : Geometry.Cell g => Cell.mk .int (TotalScaled.logicalResult cfg u (scaledParameters x scale) initial i.1.val i.2.val))
    (Geometry.Owns g) (scaledOne cfg u x scale) initial (count x.M x.N x.config)
    (fun pid hp current hout => scaled_single rest cfg u x scale legal env geometry ha hb hs initial current hout pid hp)
    (List.range (count x.M x.N x.config)) (fun _ hp => List.mem_range.mp hp) initial (fun _ _ _ => rfl)
  refine ⟨final,he,?_,?_⟩
  · intro i _
    obtain ⟨pid,hp,owns⟩ := Geometry.covers g geometry i
    exact values i ⟨pid,List.mem_range.mpr hp,owns⟩
  · intro r o outside
    exact frame r o (outside.imp id (fun h i _ => h i trivial))

#axiomsClean int_correct
#axiomsClean scaled_correct
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.FiniteGrid
