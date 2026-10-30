import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.Real
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.RealHost
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_readback_active)
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
/-- The ideal-real interpretation retains signed packed32 storage and all
caller views. It reads the current memory of each launched program. -/
def input (c : Call) (s : State ℝ) : Real.Input where
  A := fun r k => (s.mem "A" (c.A.addr r k)).read .fp16
  Q := fun k n => BitVec.ofInt 32 ((s.mem "Q" (c.Q.addr k n)).read .int)
  S := fun g n => (s.mem "S" (c.S.addr g n)).read .fp16
  Z := fun g n => BitVec.ofInt 32 ((s.mem "Z" (c.Z.addr g n)).read .int)

def stored (c : Call) (i : Fin 17) (pid : Nat) (s : State ℝ) : State ℝ :=
  let g := Grid.geometry (params c i)
  (TileShape.allIndices [g.blockRows,g.blockColumns]).foldl
    (fun out lane => if Geometry.Active g pid lane then out.write "C" (Geometry.offset g pid lane)
      (.mk .fp16 (Real.value c i (input c s) (Geometry.outRow g pid lane) (Geometry.outColumn g pid lane)))
      else out) s

def one (c : Call) (i : Fin 17) (pid : Nat) (s : State ℝ) : Option (State ℝ) :=
  some (stored c i pid (GridExecution.atProgram s pid))
def grid (c : Call) (i : Fin 17) (s : State ℝ) :=
  GridExecution.sequence (one c i) (List.range (Geometry.count (Grid.geometry (params c i)))) s

theorem inputs_preserved (c : Call) (s t : State ℝ) (h : GridExecution.Outside "C" s t) :
    input c t=input c s := by
  have ha : ("A":RegionName)≠"C" := by decide
  have hq : ("Q":RegionName)≠"C" := by decide
  have hs : ("S":RegionName)≠"C" := by decide
  have hz : ("Z":RegionName)≠"C" := by decide
  simp only [input,h _ _ ha,h _ _ hq,h _ _ hs,h _ _ hz]

theorem single (c : Call) (h : c.Valid) (i : Fin 17) (initial current : State ℝ)
    (hout : GridExecution.Outside "C" initial current) (pid : Nat) :
    ∃ final,one c i pid current=some final ∧
      GridExecution.Post "C" (Geometry.address (Grid.geometry (params c i)))
        (fun x : Grid.Index (params c i) => .mk .fp16 (Real.mathematical c (input c initial) x.1.val x.2.val))
        (Geometry.Owns (Grid.geometry (params c i)) pid) current final := by
  let g := Grid.geometry (params c i)
  let s := GridExecution.atProgram current pid
  refine ⟨stored c i pid s,rfl,?_,?_⟩
  · intro x hx
    obtain ⟨lane,hr,hc⟩ := hx
    change Geometry.outRow g pid lane=x.1.val at hr
    change Geometry.outColumn g pid lane=x.2.val at hc
    have hi : Geometry.Active g pid lane := Geometry.owns_active g pid x lane hr hc
    have ho : Geometry.offset g pid lane=Geometry.address g x := by simp only [Geometry.offset,Geometry.address,hr,hc]
    rw [←ho]
    have read := masked_readback_active (α:=ℝ) "C" (Geometry.offset g pid)
      (fun lane => Cell.mk .fp16 (Real.value c i (input c s) (Geometry.outRow g pid lane) (Geometry.outColumn g pid lane)))
      (Geometry.Active g pid) s lane hi (Geometry.offset_injective g (geometry_valid c h i) pid)
    change (stored c i pid s).mem "C" (Geometry.offset g pid lane)=_ at read
    rw [read]
    simp only [hr,hc,Real.exact_schedule c h i,inputs_preserved c initial s hout]
  · intro r a hf
    apply State.masked_scatter_frame (α:=ℝ) "C" (Geometry.offset g pid)
      (fun lane => Cell.mk .fp16 (Real.value c i (input c s) (Geometry.outRow g pid lane) (Geometry.outColumn g pid lane)))
      (Geometry.Active g pid) _ r a _ s
    rcases hf with hf|hf
    · exact Or.inl hf
    · right
      intro lane _ hl
      let x := Geometry.index g pid lane hl
      exact hf x ⟨lane,rfl,rfl⟩

theorem whole_grid (c : Call) (h : c.Valid) (i : Fin 17) (initial : State ℝ) :
    ∃ final,grid c i initial=some final ∧
      GridExecution.Post "C" (Geometry.address (Grid.geometry (params c i)))
        (fun x : Grid.Index (params c i) => .mk .fp16 (Real.mathematical c (input c initial) x.1.val x.2.val))
        (fun _ => True) initial final := by
  obtain ⟨final,he,hr,hf⟩ := GridExecution.compose "C" (Geometry.address (Grid.geometry (params c i)))
    (geometry_valid c h i).writable
    (fun x : Grid.Index (params c i) => Cell.mk .fp16 (Real.mathematical c (input c initial) x.1.val x.2.val))
    (Geometry.Owns (Grid.geometry (params c i))) (one c i) initial
    (Geometry.count (Grid.geometry (params c i)))
    (fun pid _ current hout => single c h i initial current hout pid)
    (List.range (Geometry.count (Grid.geometry (params c i)))) (fun _ hp => List.mem_range.mp hp)
    initial (fun _ _ _ => rfl)
  refine ⟨final,he,?_,?_⟩
  · intro x _
    obtain ⟨pid,hp,hx⟩ := Geometry.covers _ (geometry_valid c h i) x
    exact hr x ⟨pid,List.mem_range.mpr hp,hx⟩
  · intro r a hmiss
    exact hf r a (hmiss.imp id (fun h x _ => h x trivial))

structure Result where
  state : State ℝ
  output : Host.Tensor
  returned : Option Host.Tensor
  allocated : Bool

def prepare (c : Call) (s : State ℝ) (fresh : Nat → ℝ) : State ℝ :=
  if c.supplied then s else {s with mem := fun r a => if r="C" then .mk .fp16 (fresh a) else s.mem r a}
def call (c : Call) (device : Nat) (i : Fin 17) (s : State ℝ) (fresh : Nat → ℝ) : Option Result :=
  (grid c i (prepare c s fresh)).map (fun final =>
    ⟨final,Host.tensor c device,if c.supplied then none else some (Host.tensor c device),!c.supplied⟩)
def Claim (c : Call) (device : Nat) (i : Fin 17) (s : State ℝ) (fresh : Nat → ℝ) (result : Result) : Prop :=
  GridExecution.Post "C" (Geometry.address (Grid.geometry (params c i)))
    (fun x : Grid.Index (params c i) => .mk .fp16 (Real.mathematical c (input c s) x.1.val x.2.val))
    (fun _ => True) (prepare c s fresh) result.state ∧
  result.output=Host.tensor c device ∧
  result.returned=(if c.supplied then none else some result.output) ∧ result.allocated=(!c.supplied) ∧
  (∀ (r : RegionName) a,r≠"C" → result.state.mem r a=s.mem r a)

theorem prepare_outside (c : Call) (s : State ℝ) (fresh : Nat → ℝ) :
    GridExecution.Outside "C" s (prepare c s fresh) := by
  intro r a hr
  cases hc : c.supplied <;> simp [prepare,hc,hr]

theorem whole_call (c : Call) (h : c.Valid) (device : Nat) (i : Fin 17) (s : State ℝ) (fresh : Nat → ℝ) :
    ∃ result,call c device i s fresh=some result ∧ Claim c device i s fresh result := by
  obtain ⟨final,he,hr,hf⟩ := whole_grid c h i (prepare c s fresh)
  refine ⟨⟨final,Host.tensor c device,if c.supplied then none else some (Host.tensor c device),!c.supplied⟩,
    ?_,⟨?_,hf⟩,rfl,rfl,rfl,?_⟩
  · simp only [call,he,Option.map_some]
  · rw [inputs_preserved c s (prepare c s fresh) (prepare_outside c s fresh)] at hr
    exact hr
  · intro r a ha
    exact (hf r a (Or.inl ha)).trans (prepare_outside c s fresh r a ha)

#specNonCircular Claim avoiding [call,grid,one,stored,Real.value,Real.accumulator]
#axiomsClean inputs_preserved
#axiomsClean single
#axiomsClean whole_grid
#axiomsClean whole_call
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.RealHost
