import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.GridGeometry

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem program_rows_positive (i : Contract.Inputs) (id : Program) (valid : id.Valid i) :
    0<(RmsMatmulRbe.Checked.shape i.original).M := by
  by_contra h
  have zero : (RmsMatmulRbe.Checked.shape i.original).M=0 := by omega
  have bound := valid.2
  simp [RmsMatmulRbe.tilesM,zero] at bound

theorem columns_positive (i : Contract.Inputs) (checked : Contract.InputContract i) :
    0<(RmsMatmulRbe.Checked.shape i.original).N := by
  have f := RmsMatmulRbe.Checked.facts i.original checked.1
  rw [f.valid.2.2.2.2]
  exact Nat.mul_pos f.valid.2.1 f.valid.2.2.1

theorem source_x_index (access : Access α Store Byte) (a : Arithmetic α) (i : Contract.Inputs)
    (q : RmsMatmulRbe.Contract.Tensor) (s : Store) (id : Program) (phase : Fin 3)
    (t : Nat) (lane : TileIndex [16,64]) :
    ((valuesAt a (useFP8 i) (qkvArgument (arguments i) phase) (state access a.zero i q s id) t).xptrs lane).2=
      id.batch*((RmsMatmulRbe.Checked.shape i.original).M*(RmsMatmulRbe.Checked.shape i.original).K)+
      (row i id lane.1%(RmsMatmulRbe.Checked.shape i.original).M)*(RmsMatmulRbe.Checked.shape i.original).K+
      (t*64+lane.2.1.val) := by
  rw [valuesAt_xptrs]
  simp [qkvArgument,arguments,state,Program.pids,programRow,columns,row,RmsMatmulRbe.tilesN]

theorem source_weight_index (access : Access α Store Byte) (a : Arithmetic α) (i : Contract.Inputs)
    (q : RmsMatmulRbe.Contract.Tensor) (s : Store) (id : Program) (phase : Fin 3)
    (t : Nat) (lane : TileIndex [64,64]) :
    ((valuesAt a (useFP8 i) (qkvArgument (arguments i) phase) (state access a.zero i q s id) t).wptrs lane).2=
      (col i id lane.2.1%(RmsMatmulRbe.Checked.shape i.original).N)*(RmsMatmulRbe.Checked.shape i.original).K+
      (t*64+lane.1.val) := by
  rw [valuesAt_wptrs]
  simp [qkvArgument,arguments,state,Program.pids,programCol,columns,col,RmsMatmulRbe.tilesN,Nat.add_comm]

theorem source_rms_index (access : Access α Store Byte) (a : Arithmetic α) (i : Contract.Inputs)
    (q : RmsMatmulRbe.Contract.Tensor) (s : Store) (id : Program) (phase : Fin 3)
    (t : Nat) (lane : TileIndex [1,64]) :
    ((valuesAt a (useFP8 i) (qkvArgument (arguments i) phase) (state access a.zero i q s id) t).rmsptrs lane).2=
      t*64+lane.2.1.val := by
  rw [valuesAt_rmsptrs]
  simp [qkvArgument,arguments]

/-- Every actual X load under the source tail mask is inside the caller
backing, including modulo-wrapped loads belonging to inactive output rows. -/
theorem x_load_safe (access : Access α Store Byte) (a : Arithmetic α) (i : Contract.Inputs)
    (checked : Contract.InputContract i) (ordinary : OrdinaryInputs i)
    (q : RmsMatmulRbe.Contract.Tensor) (s : Store) (id : Program) (valid : id.Valid i)
    (phase : Fin 3) (t : Nat) (lane : TileIndex [16,64])
    (mask : lane.2.1.val<(RmsMatmulRbe.Checked.shape i.original).K-t*64) :
    let j := ((valuesAt a (useFP8 i) (qkvArgument (arguments i) phase) (state access a.zero i q s id) t).xptrs lane).2
    inBacking i.original.x j ∧ RmsMatmulRbe.Checked.PointerValid i.original.x j := by
  dsimp only
  rw [source_x_index]
  have index := x_index_lt i checked id.batch
    (row i id lane.1%(RmsMatmulRbe.Checked.shape i.original).M) (t*64+lane.2.1.val)
    valid.1 (Nat.mod_lt _ (program_rows_positive i id valid)) (by omega)
  have ord := ordinary i.original.x (by simp [RmsMatmulRbe.Contract.Inputs.reads])
  exact ⟨inBacking_of_lt _ ord _ index,RmsMatmulRbe.Checked.pointer_valid _ ord
    (RmsMatmulRbe.Checked.facts i.original checked.1).xAdmitted _ index⟩

theorem weight_load_safe (access : Access α Store Byte) (a : Arithmetic α) (i : Contract.Inputs)
    (checked : Contract.InputContract i) (ordinary : OrdinaryInputs i)
    (q : RmsMatmulRbe.Contract.Tensor) (s : Store) (id : Program) (phase : Fin 3)
    (t : Nat) (lane : TileIndex [64,64])
    (mask : lane.1.val<(RmsMatmulRbe.Checked.shape i.original).K-t*64) :
    let j := ((valuesAt a (useFP8 i) (qkvArgument (arguments i) phase) (state access a.zero i q s id) t).wptrs lane).2
    inBacking (RmsMatmulRbe.Checked.weight i.original phase) j ∧
      RmsMatmulRbe.Checked.PointerValid (RmsMatmulRbe.Checked.weight i.original phase) j := by
  dsimp only
  rw [source_weight_index]
  have index := weight_index_lt i checked phase
    (col i id lane.2.1%(RmsMatmulRbe.Checked.shape i.original).N) (t*64+lane.1.val)
    (Nat.mod_lt _ (columns_positive i checked)) (by omega)
  have ord := ordinary (RmsMatmulRbe.Checked.weight i.original phase)
    (List.mem_append_left _ (RmsMatmulRbe.Checked.weight_mem i.original phase))
  have admitted : RmsMatmulRbe.Contract.TensorAdmitted (RmsMatmulRbe.Checked.weight i.original phase)
      RmsMatmulRbe.Contract.weightTypes := by
    apply (RmsMatmulRbe.Checked.facts i.original checked.1).others
    fin_cases phase <;> simp [RmsMatmulRbe.Checked.weight,RmsMatmulRbe.Contract.Inputs.others]
  exact ⟨inBacking_of_lt _ ord _ index,RmsMatmulRbe.Checked.pointer_valid _ ord admitted _ index⟩

theorem rms_load_safe (access : Access α Store Byte) (a : Arithmetic α) (i : Contract.Inputs)
    (checked : Contract.InputContract i) (ordinary : OrdinaryInputs i)
    (q : RmsMatmulRbe.Contract.Tensor) (s : Store) (id : Program) (phase : Fin 3)
    (t : Nat) (lane : TileIndex [1,64])
    (mask : lane.2.1.val<(RmsMatmulRbe.Checked.shape i.original).K-t*64) :
    let j := ((valuesAt a (useFP8 i) (qkvArgument (arguments i) phase) (state access a.zero i q s id) t).rmsptrs lane).2
    inBacking i.original.rms j ∧ RmsMatmulRbe.Checked.PointerValid i.original.rms j := by
  dsimp only
  rw [source_rms_index]
  have index := rms_index_lt i checked (t*64+lane.2.1.val) (by omega)
  have ord := ordinary i.original.rms (by simp [RmsMatmulRbe.Contract.Inputs.reads])
  exact ⟨inBacking_of_lt _ ord _ index,RmsMatmulRbe.Checked.pointer_valid _ ord
    ((RmsMatmulRbe.Checked.facts i.original checked.1).others i.original.rms
      (by simp [RmsMatmulRbe.Contract.Inputs.others])) _ index⟩

#axiomsClean x_load_safe
#axiomsClean weight_load_safe
#axiomsClean rms_load_safe
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel
