import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Execution

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Source
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 5000000

theorem output_address (p : Problem) (b : Buffers) (tile : Nat) (lane : Lane)
    (ha : active p tile lane) :
    outputAddress p b tile lane=b.C.address (p.row tile lane.1.val) (p.column tile lane.2.val) := by
  simp [outputAddress,clamp,ha.1,ha.2]

theorem output_injective (p : Problem) (b : Buffers) (layout : Layout p b)
    (r c r' c' : Nat) (hc : c<p.N) (hc' : c'<p.N)
    (he : b.C.address r c=b.C.address r' c') : r=r' ∧ c=c' := by
  have index := congrArg Prod.snd he
  simp only [Tensor.address,layout.outputStrides.1,layout.outputStrides.2,Nat.mul_one,
    max_eq_left (show 1≤p.N by omega)] at index
  have eqn : r*p.N+c=r'*p.N+c' := by omega
  have cm := congrArg (fun n => n%p.N) eqn
  simp [Nat.add_mod,Nat.mod_eq_of_lt hc,Nat.mod_eq_of_lt hc'] at cm
  have rm : r*p.N=r'*p.N := by omega
  exact ⟨Nat.eq_of_mul_eq_mul_right (by omega) rm,cm⟩

theorem lanes_unique (p : Problem) (b : Buffers) (layout : Layout p b)
    (tile : Nat) (a c : Lane) (ha : active p tile a) (hc : active p tile c)
    (he : outputAddress p b tile a=outputAddress p b tile c) : a=c := by
  rw [output_address p b tile a ha,output_address p b tile c hc] at he
  have coordinates := output_injective p b layout _ _ _ _ ha.2 hc.2 he
  apply Prod.ext <;> apply Fin.ext
  · exact Nat.add_left_cancel coordinates.1
  · exact Nat.add_left_cancel coordinates.2

theorem value_at_lane {α β : Type} (ops : Arithmetic α β) (p : Problem) (b : Buffers)
    (layout : Layout p b) (entry : Memory α) (tile : Nat) (lane : Lane)
    (ha : active p tile lane) :
    value ops p b entry tile (outputAddress p b tile lane)=
      ops.cast .fp16 (ops.activation p.enabled (accumulator ops p b entry tile p.kTiles lane.1 lane.2)) := by
  have hw : writes p b tile (outputAddress p b tile lane) := ⟨lane,ha,rfl⟩
  unfold value
  rw [dif_pos hw]
  have chosen := Classical.choose_spec hw
  have same := lanes_unique p b layout tile (Classical.choose hw) lane chosen.1 ha chosen.2
  simp only [same]

theorem row_recover (p : Problem) (tile : Nat) (lane : Fin 128) :
    p.row tile lane.val/128=p.tileRow tile := by
  unfold Problem.row
  rw [Nat.add_comm,Nat.add_mul_div_right _ _ (by decide),Nat.div_eq_of_lt lane.isLt,Nat.zero_add]

theorem column_recover (p : Problem) (tile : Nat) (lane : Fin 256) :
    p.column tile lane.val/256=p.tileColumn tile := by
  unfold Problem.column
  rw [Nat.add_comm,Nat.add_mul_div_right _ _ (by decide),Nat.div_eq_of_lt lane.isLt,Nat.zero_add]

theorem tile_writes_unique (p : Problem) (b : Buffers) (layout : Layout p b)
    (hm : 0<p.M) (tile other : Nat) (ht : tile<p.tiles) (ho : other<p.tiles)
    (point : Address) (hw : writes p b tile point) (hw' : writes p b other point) : tile=other := by
  obtain ⟨lane,ha,he⟩ := hw
  obtain ⟨lane',ha',he'⟩ := hw'
  have same := he.trans he'.symm
  rw [output_address p b tile lane ha,output_address p b other lane' ha'] at same
  have coords := output_injective p b layout _ _ _ _ ha.2 ha'.2 same
  have rows := congrArg (fun n => n/128) coords.1
  have columns := congrArg (fun n => n/256) coords.2
  simp only [row_recover] at rows
  simp only [column_recover] at columns
  exact p.grouped_injective (p.group_positive hm) tile other ht ho rows columns

/-- The original grouped-schedule and quotient/remainder coverage results
are composed with the repaired concrete lane and host-grid definitions. -/
theorem output_covered (p : Problem) (row column : Nat)
    (hr : row<p.M) (hc : column<p.N) :
    ∃ tile∈tiles p, ∃ lane : Lane, active p tile lane ∧
      p.row tile lane.1.val=row ∧ p.column tile lane.2.val=column := by
  have rm : row/128<p.rows := by
    apply (Nat.div_lt_iff_lt_mul (by decide : 0<128)).mpr
    exact lt_of_lt_of_le hr (RealMath.blocks_cover p.M 128 (by decide))
  have cn : column/256<p.columns := by
    apply (Nat.div_lt_iff_lt_mul (by decide : 0<256)).mpr
    exact lt_of_lt_of_le hc (RealMath.blocks_cover p.N 256 (by decide))
  obtain ⟨tile,ht,hrow,hcol⟩ := p.grouped_covered (p.group_positive (by omega)) _ _ rm cn
  let lane : Lane := (⟨row%128,Nat.mod_lt _ (by decide)⟩,⟨column%256,Nat.mod_lt _ (by decide)⟩)
  have rowEq : p.row tile lane.1.val=row := by
    simp only [Problem.row,hrow,lane]
    omega
  have colEq : p.column tile lane.2.val=column := by
    simp only [Problem.column,hcol,lane]
    omega
  refine ⟨tile,(tiles_exact p tile).mpr ht,lane,?_,rowEq,colEq⟩
  exact ⟨rowEq ▸ hr,colEq ▸ hc⟩

theorem source_output {α β : Type} (ops : Arithmetic α β) (p : Problem) (b : Buffers)
    (layout : Layout p b) (entry : Memory α) (tile : Nat)
    (ht : tile∈tiles p) (lane : Lane) (ha : active p tile lane) :
    run ops p b entry (outputAddress p b tile lane)=
      ops.cast .fp16 (ops.activation p.enabled (accumulator ops p b entry tile p.kTiles lane.1 lane.2)) := by
  have activeWrite : writes p b tile (outputAddress p b tile lane) := ⟨lane,ha,rfl⟩
  have once := CrossEntropy1.Schedule.fixed_output (writes p b) (value ops p b entry)
    (tiles p) entry tile (outputAddress p b tile lane) activeWrite
    (by
      intro other ho hw
      exact tile_writes_unique p b layout (by have h:=ha.1; omega) other tile
        ((tiles_exact p other).mp ho) ((tiles_exact p tile).mp ht) _ hw activeWrite)
  unfold run
  rw [frozen ops p b layout entry (tiles p) entry (by intros; rfl)]
  rw [once,if_pos ht]
  exact value_at_lane ops p b layout entry tile lane ha

theorem source_frame {α β : Type} (ops : Arithmetic α β) (p : Problem) (b : Buffers)
    (layout : Layout p b) (entry : Memory α) (point : Address)
    (outside : ∀ row<p.M, ∀ column<p.N, b.C.address row column≠point) :
    run ops p b entry point=entry point := by
  unfold run
  rw [frozen ops p b layout entry (tiles p) entry (by intros; rfl)]
  apply CrossEntropy1.Schedule.fixed_frame
  intro tile _
  rintro ⟨lane,ha,he⟩
  rw [output_address p b tile lane ha] at he
  exact outside _ ha.1 _ ha.2 he

#axiomsClean source_output
#axiomsClean source_frame
#axiomsClean output_covered
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Source
