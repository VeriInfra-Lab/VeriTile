import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.RawProgram
import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.Grid

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def rawPrograms (c : Config) (ops : Ops α) : List Program → Mem α → Mem α
  | [],m => m
  | pid::ps,m => rawPrograms c ops ps (rawProgram c ops pid m)

theorem rawPrograms_values (c : Config) (hc : c.Valid) (ops : Ops α)
    (ps : List Program) (inside : ∀ pid ∈ ps, pid ∈ programs c.shape) (m : Mem α) :
    rawValues (rawPrograms c ops ps m) = actualPrograms c (fun _=>ops) ps (rawValues m) := by
  induction ps generalizing m with
  | nil => rfl
  | cons pid ps ih =>
    simp only [rawPrograms,actualPrograms]
    rw [ih (fun p hp=>inside p (by simp [hp])),rawProgram_values,
      ←actualProgram_natural c hc ops pid (inside pid (by simp))]

theorem rawPrograms_metadata (c : Config) (ops : Ops α) (ps : List Program)
    (m : Mem α) (hm : MetadataBinding c m) : MetadataBinding c (rawPrograms c ops ps m) := by
  induction ps generalizing m with
  | nil => exact hm
  | cons pid ps ih => exact ih _ (rawProgram_metadata c ops pid m hm)

theorem schedule_memory (rest : Algebra α) (ops : Ops α) (c : Config) (hc : c.Valid)
    (ps : List Program) (inside : ∀ pid ∈ ps, pid ∈ programs c.shape)
    (s : State α) (hm : MetadataBinding c s.mem) :
    ∃ final, runSchedule (ps.map (invocation rest ops c)) s = some final ∧
      final.mem = rawPrograms c ops ps s.mem := by
  induction ps generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons pid ps ih =>
    have ht := List.mem_range.mp (List.mem_product.mp (inside pid (by simp))).2
    obtain ⟨next,executed,memory⟩ := invocation_memory rest ops c hc pid ht s hm
    have metadataOk : MetadataBinding c next.mem := by
      rw [memory]
      exact rawProgram_metadata c ops pid s.mem hm
    obtain ⟨final,tail,hmemory⟩ := ih (fun p hp=>inside p (by simp [hp])) next metadataOk
    refine ⟨final,?_,?_⟩
    · simp only [List.map_cons,runSchedule,executed,Option.bind_some,tail]
    · simpa only [rawPrograms,memory] using hmemory

theorem rotationWrites_footprint (c : Config) (hc : c.Valid) (ops : Ops α)
    (pid : Program) (hp : pid ∈ programs c.shape) (key high : Bool) (m : Memory α)
    (w : Stores.Write α) (hw : w ∈ rotationWrites c ops pid key high m) :
    Footprint c w.1.1 w.1.2 := by
  rcases List.mem_map.mp hw with ⟨i,hi,rfl⟩
  have ht := List.mem_range.mp (List.mem_product.mp hp).2
  have hh := program_head_bound c.shape hc.shape pid hp (if key then .k else .q)
  have activeWork : c.shape.Active (work c.shape pid key i) := by
    cases key <;> simpa [work,Shape.Active,programHead,ht,List.mem_range.mp hi] using hh
  exact writes_footprint c hc _ activeWork _ _ ⟨rfl,by cases high <;> simp⟩

theorem vWrites_footprint (c : Config) (hc : c.Valid) (pid : Program)
    (hp : pid ∈ programs c.shape) (m : Memory α) (w : Stores.Write α)
    (hw : w ∈ vWrites c pid m) : Footprint c w.1.1 w.1.2 := by
  rcases List.mem_map.mp hw with ⟨i,hi,rfl⟩
  have ht := List.mem_range.mp (List.mem_product.mp hp).2
  have hh := program_head_bound c.shape hc.shape pid hp .v
  refine ⟨.v,pid.2,pid.1/c.shape.groups,i,ht,?_,List.mem_range.mp hi,rfl,rfl⟩
  simpa [programHead] using hh

theorem programWrites_footprint (c : Config) (hc : c.Valid) (ops : Ops α)
    (pid : Program) (hp : pid ∈ programs c.shape) (m : Memory α)
    (w : Stores.Write α) (hw : w ∈ programWrites c ops pid m) : Footprint c w.1.1 w.1.2 := by
  simp only [programWrites,List.mem_append] at hw
  rcases hw with (hw|hw)|hw
  · exact rotationWrites_footprint c hc ops pid hp _ _ m w hw
  · exact rotationWrites_footprint c hc ops pid hp _ _ m w hw
  · split_ifs at hw with leader
    · rcases List.mem_append.mp hw with hw|hw
      · rcases List.mem_append.mp hw with hw|hw <;>
          exact rotationWrites_footprint c hc ops pid hp _ _ m w hw
      · exact vWrites_footprint c hc pid hp m w hw
    · simp at hw

theorem decoded_output (kind : Kind) : decoded (region kind.output) = kind.output := by
  cases kind <;> simp [decoded,region,Kind.output]

theorem rawProgram_frame (c : Config) (hc : c.Valid) (ops : Ops α)
    (pid : Program) (hp : pid ∈ programs c.shape) (m : Mem α) (r : RegionName) (a : Nat)
    (outside : ¬Footprint c (decoded r) a) : rawProgram c ops pid m r a = m r a := by
  apply rawCommit_frame
  intro w hw ⟨hr,ha⟩
  obtain ⟨kind,hk⟩ := programWrites_outputs c ops pid (rawValues m) w hw
  have footprint := programWrites_footprint c hc ops pid hp (rawValues m) w hw
  apply outside
  simpa only [hr,ha,hk,decoded_output] using footprint

theorem rawPrograms_frame (c : Config) (hc : c.Valid) (ops : Ops α)
    (ps : List Program) (inside : ∀ pid ∈ ps, pid ∈ programs c.shape) (m : Mem α)
    (r : RegionName) (a : Nat) (outside : ¬Footprint c (decoded r) a) :
    rawPrograms c ops ps m r a = m r a := by
  induction ps generalizing m with
  | nil => rfl
  | cons pid ps ih =>
    change rawPrograms c ops ps (rawProgram c ops pid m) r a = m r a
    rw [ih (fun p hp=>inside p (by simp [hp])),rawProgram_frame c hc ops pid (inside pid (by simp)) m r a outside]

theorem rawPrograms_real (c : Config) (ops : Ops α) (ps : List Program) (m : Mem α)
    (r : RegionName) (a : Nat) (already : ∃ x, m r a = .mk .real x) :
    ∃ x, rawPrograms c ops ps m r a = .mk .real x := by
  induction ps generalizing m with
  | nil => exact already
  | cons pid ps ih => exact ih _ (rawCommit_real _ m r a already)

def producer (s : Shape) (kind : Kind) (t h : Nat) : Program :=
  (if kind=.q then h else h*s.groups,t)

theorem producer_member (s : Shape) (hs : s.Valid) (kind : Kind) (t h : Nat)
    (ht : t<s.tokens) (hh : h<s.heads kind) : producer s kind t h ∈ programs s := by
  apply List.mem_product.mpr
  refine ⟨List.mem_range.mpr ?_,List.mem_range.mpr ht⟩
  by_cases hq : kind=.q
  · simpa [producer,hq,Shape.heads] using hh
  · have hk : h<s.headsK := by simpa [Shape.heads,hq] using hh
    have result := Nat.mul_lt_mul_of_pos_right hk (groups_positive s hs)
    simpa only [producer,hq,ite_false,Nat.mul_comm s.headsK s.groups,groups_mul s hs] using result

theorem rotation_written (c : Config) (ops : Ops α) (pid : Program)
    (key high : Bool) (i : Nat) (hi : i<c.shape.half) (m : Memory α)
    (guard : key=Bool.false ∨ pid.1%c.shape.groups=0) :
    ∃ w ∈ programWrites c ops pid m,
      region (work c.shape pid key i).kind.output = region w.1.1 ∧
        c.output (work c.shape pid key i) high = w.1.2 := by
  let w := work c.shape pid key i
  refine ⟨((w.kind.output,c.output w high),component (pairValues c.shape (fun _=>ops) w m) high),?_,rfl,rfl⟩
  have member : ((w.kind.output,c.output w high),component (pairValues c.shape (fun _=>ops) w m) high) ∈
      rotationWrites c ops pid key high m := List.mem_map.mpr ⟨i,List.mem_range.mpr hi,rfl⟩
  cases key
  · cases high <;> simp only [programWrites,List.mem_append] <;> tauto
  · have leader : pid.1%c.shape.groups=0 := guard.resolve_left (by decide)
    cases high <;> simp only [programWrites,leader,ite_true,List.mem_append] <;> tauto

theorem producer_written (c : Config) (hc : c.Valid) (ops : Ops α) (kind : Kind)
    (t h d : Nat) (hd : d<c.shape.width) (m : Memory α) :
    ∃ w ∈ programWrites c ops (producer c.shape kind t h) m,
      region kind.output = region w.1.1 ∧ logicalOutput c kind t h d = w.1.2 := by
  have gp := groups_positive c.shape hc.shape
  have gh : h*c.shape.groups/c.shape.groups=h := Nat.mul_div_cancel h gp
  have wh := width_halves c.shape hc.shape
  cases kind
  · by_cases low : d<c.shape.half
    · simpa [producer,work,Config.output,logicalOutput,Shape.input,Shape.dim] using
        rotation_written c ops (producer c.shape .q t h) Bool.false Bool.false d low m (Or.inl rfl)
    · have half : d-c.shape.half<c.shape.half := by omega
      simpa [producer,work,Config.output,logicalOutput,Shape.input,Shape.dim,
        Nat.sub_add_cancel (by omega : c.shape.half≤d)] using
        rotation_written c ops (producer c.shape .q t h) Bool.false Bool.true (d-c.shape.half) half m (Or.inl rfl)
  · have leader : (producer c.shape .k t h).1%c.shape.groups=0 := by simp [producer]
    by_cases low : d<c.shape.half
    · simpa [producer,work,Config.output,logicalOutput,Shape.dim,gh] using
        rotation_written c ops (producer c.shape .k t h) Bool.true Bool.false d low m (Or.inr leader)
    · have half : d-c.shape.half<c.shape.half := by omega
      simpa [producer,work,Config.output,logicalOutput,Shape.dim,gh,
        Nat.sub_add_cancel (by omega : c.shape.half≤d)] using
        rotation_written c ops (producer c.shape .k t h) Bool.true Bool.true (d-c.shape.half) half m (Or.inr leader)
  · let pid := producer c.shape .v t h
    let a := c.shape.cache Bool.false (c.block pid.2) (pid.1/c.shape.groups) (c.slot pid.2) d
    let v := m .v (c.shape.data .v pid.2 (pid.1/c.shape.groups) d)
    refine ⟨((.vcache,a),v),?_,rfl,?_⟩
    · have member : ((.vcache,a),v) ∈ vWrites c pid m := List.mem_map.mpr ⟨d,List.mem_range.mpr hd,rfl⟩
      have leader : pid.1%c.shape.groups=0 := by simp [pid,producer]
      change _ ∈ programWrites c ops pid m
      simp only [programWrites,leader,ite_true,List.mem_append]
      exact Or.inr (Or.inr member)
    · simp [a,pid,producer,logicalOutput,gh]

theorem rawPrograms_written (c : Config) (hc : c.Valid) (ops : Ops α) (kind : Kind)
    (t h d : Nat) (hd : d<c.shape.width) (ps : List Program)
    (member : producer c.shape kind t h ∈ ps) (m : Mem α) :
    ∃ x, rawPrograms c ops ps m (region kind.output) (logicalOutput c kind t h d) = .mk .real x := by
  induction ps generalizing m with
  | nil => simp at member
  | cons pid ps ih =>
    rcases List.mem_cons.mp member with eq|member
    · subst pid
      exact rawPrograms_real c ops ps _ _ _
        (rawCommit_written _ m _ _ (producer_written c hc ops kind t h d hd (rawValues m)))
    · exact ih member _

theorem complete_raw_output (rest : Algebra α) (ops : Ops α) (c : Config) (hc : c.Valid)
    (s : State α) (hm : MetadataBinding c s.mem) :
    ∃ final, runSchedule ((programs c.shape).map (invocation rest ops c)) s = some final ∧
      (∀ kind t h d, t<c.shape.tokens → h<c.shape.heads kind → d<c.shape.width →
        final.mem (region kind.output) (logicalOutput c kind t h d) =
          .mk .real (actualSource c (fun _=>ops) (rawValues s.mem) kind.output (logicalOutput c kind t h d))) ∧
      (∀ r a, ¬Footprint c (decoded r) a → final.mem r a = s.mem r a) := by
  obtain ⟨final,executed,memory⟩ := schedule_memory rest ops c hc (programs c.shape) (fun _ h=>h) s hm
  refine ⟨final,executed,?_,?_⟩
  · intro kind t h d ht hh hd
    obtain ⟨x,hx⟩ := rawPrograms_written c hc ops kind t h d hd (programs c.shape)
      (producer_member c.shape hc.shape kind t h ht hh) s.mem
    have value := congrFun (congrFun (rawPrograms_values c hc ops (programs c.shape) (fun _ h=>h) s.mem)
      kind.output) (logicalOutput c kind t h d)
    change (rawPrograms c ops (programs c.shape) s.mem (region kind.output) (logicalOutput c kind t h d)).read .real = actualSource c (fun _=>ops) (rawValues s.mem) kind.output (logicalOutput c kind t h d) at value
    rw [hx] at value
    change x = _ at value
    rw [memory,hx,value]
  · intro r a outside
    rw [memory]
    exact rawPrograms_frame c hc ops _ (fun _ h=>h) s.mem r a outside

#axiomsClean schedule_memory
#axiomsClean complete_raw_output
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
