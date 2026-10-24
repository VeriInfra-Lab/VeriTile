import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Proofs.IOScatter
import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Proofs.IOCoverage

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PublicIO
open VeriTile Triton FP.Structural FlashAttn FlashAttn.ActualSource
noncomputable section
set_option maxHeartbeats 1000000
variable {A B : Type} [Inhabited A] [Inhabited B]

theorem lastWriter_present (x : Layout)
    (i : TileIndex [x.shape.batches,x.shape.heads,x.shape.length,x.shape.dim]) :
    ∃ w,Logical.lastWriter x (FlashAttn.address x.o i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val)=some w := by
  let address := FlashAttn.address x.o i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val
  let job : Program := (i.2.2.1.val/128,i.1.val*x.shape.heads+i.2.1.val)
  let lane : TileIndex [128,x.shape.dim] :=
    (⟨i.2.2.1.val%128,Nat.mod_lt _ (by decide)⟩,i.2.2.2.1,())
  have hj := output_program_present x i.1.val i.2.1.val i.2.2.1.val
    ⟨i.1.isLt,i.2.1.isLt,i.2.2.1.isLt⟩
  have hr : job.1*128+lane.1.val=i.2.2.1.val := by dsimp [job,lane];omega
  have hd := flatten_decode x.shape.heads i.1.val i.2.1.val i.2.1.isLt
  have member : (job,lane)∈(Logical.writers x).filter
      (fun w => decide (Logical.writerAddress x w=address)) := by
    have writer : (job,lane)∈Logical.writers x := by
      apply List.mem_flatMap.mpr
      refine ⟨job,hj,List.mem_map.mpr ⟨lane,?_,rfl⟩⟩
      exact List.mem_filter.mpr ⟨TileShape.mem_allIndices _ _,by simpa [hr] using i.2.2.1.isLt⟩
    apply List.mem_filter.mpr
    refine ⟨writer,?_⟩
    simp [address,Logical.writerAddress,job,lane,hr,hd.1,hd.2]
  cases hl : Logical.lastWriter x address with
  | some w => exact ⟨w,rfl⟩
  | none =>
    have empty : (Logical.writers x).filter (fun w => decide (Logical.writerAddress x w=address))=[] := by
      simpa only [Logical.lastWriter,List.getLast?_eq_none_iff] using hl
    rw [empty] at member
    exact False.elim (List.not_mem_nil member)

variable (M : Algebra B) (dot : Dot B) (encode : A → B) (x : Layout) (scale : B)
  (observe : (port : Fin (outputs x).length) → TileIndex (outputs x)[port].shape → B → A)
  (xs : Inputs A x) (s : State B)
  (hin : (sourceIO M dot encode x scale observe).InputBinding xs s)

include hin
theorem grid_output_tensor
    (i : TileIndex [x.shape.batches,x.shape.heads,x.shape.length,x.shape.dim]) :
    referenceGrid M dot (parameters x) scale (programs (parameters x)) s.mem "O"
      (FlashAttn.address x.o i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val)=
      .mk .real (Logical.outputPayload M dot encode x scale xs i) := by
  obtain ⟨w,hw⟩ := lastWriter_present x i
  rw [grid_output_last M dot encode x scale observe xs s hin,hw]
  simp only [Option.map_some,Option.getD_some,Logical.outputPayload,hw]
  rfl

theorem binding_of_inputsEq (mem : Mem B) (h : InputsEq mem s.mem) :
    (sourceIO M dot encode x scale observe).InputBinding xs {s with mem := mem} := by
  intro port i active
  have region : (inputs x)[port].buf∈(["Q","K","V"] : List RegionName) := by
    fin_cases port <;> simp [inputs]
  change (mem (inputs x)[port].buf _).read _=_
  rw [h _ region]
  exact hin port i active

theorem grid_log_tensor (i : TileIndex [x.shape.batches,x.shape.heads,x.shape.length]) :
    referenceGrid M dot (parameters x) scale (programs (parameters x)) s.mem "L"
      (FlashAttn.lAddress x.shape i.1.val i.2.1.val i.2.2.1.val)=
      .mk .real (Logical.logPayload M dot encode x scale xs i) := by
  let job : Program := (i.2.2.1.val/128,i.1.val*x.shape.heads+i.2.1.val)
  let ri : Fin 128 := ⟨i.2.2.1.val%128,Nat.mod_lt _ (by decide)⟩
  have member := output_program_present x i.1.val i.2.1.val i.2.2.1.val
    ⟨i.1.isLt,i.2.1.isLt,i.2.2.1.isLt⟩
  have valid : job.Valid x := (program_valid x job).mp member
  have hr : job.1*128+ri.val=i.2.2.1.val := by dsimp [job,ri];omega
  apply grid_cell M dot (parameters x) scale "L" _ job _ s.mem
  · intro mem inputs
    have binding := binding_of_inputsEq M dot encode x scale observe xs s hin mem inputs
    have write := program_log M dot (parameters x) scale job mem (ri,())
      (by simpa [lActive,stateAt,parameters,hr] using i.2.2.1.isLt)
    have addr : ActualSource.lAddress (parameters x) (stateAt (parameters x) job mem) (ri,())=
        FlashAttn.lAddress x.shape i.1.val i.2.1.val i.2.2.1.val := by
      simp [ActualSource.lAddress,stateAt,parameters,hr,job,FlashAttn.lAddress]
    rw [addr,accumulated_binding M dot encode x scale observe xs {s with mem := mem} binding job valid] at write
    exact write
  · exact member
  · intro j _ hw
    exact log_writer_unique x _ _ _ ⟨i.1.isLt,i.2.1.isLt,i.2.2.1.isLt⟩ j hw
  · exact fun _ _ _ => rfl

def tensorOutput (M : Algebra B) (dot : Dot B) (encode : A → B) (x : Layout) (scale : B)
    (xs : Inputs A x)
    (i : TileIndex [x.shape.batches,x.shape.heads,x.shape.length,x.shape.dim]) : B :=
  Logical.output M (parameters x)
    (Logical.rowAccumulator M dot encode x scale xs (i.2.2.1.val/128,i.1.val*x.shape.heads+i.2.1.val))
    (⟨i.2.2.1.val%128,Nat.mod_lt _ (by decide)⟩,i.2.2.2.1,())

theorem grid_output_injective (hs : x.shape.Valid) (ho : OutputInjective x)
    (i : TileIndex [x.shape.batches,x.shape.heads,x.shape.length,x.shape.dim]) :
    referenceGrid M dot (parameters x) scale (programs (parameters x)) s.mem "O"
      (FlashAttn.address x.o i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val)=
      .mk .real (tensorOutput M dot encode x scale xs i) := by
  let job : Program := (i.2.2.1.val/128,i.1.val*x.shape.heads+i.2.1.val)
  let ri : Fin 128 := ⟨i.2.2.1.val%128,Nat.mod_lt _ (by decide)⟩
  let lane : TileIndex [128,x.shape.dim] := (ri,i.2.2.2.1,())
  have member := output_program_present x i.1.val i.2.1.val i.2.2.1.val
    ⟨i.1.isLt,i.2.1.isLt,i.2.2.1.isLt⟩
  have valid : job.Valid x := (program_valid x job).mp member
  have hr : job.1*128+ri.val=i.2.2.1.val := by dsimp [job,ri];omega
  have hd := flatten_decode x.shape.heads i.1.val i.2.1.val i.2.1.isLt
  apply grid_cell M dot (parameters x) scale "O" _ job _ s.mem
  · intro mem inputs
    have binding := binding_of_inputsEq M dot encode x scale observe xs s hin mem inputs
    have write := program_output M dot x hs ho scale job valid mem lane
      (by simpa [active,stateAt,parameters,lane,hr] using i.2.2.1.isLt)
    have addr : ActualSource.address (parameters x) (stateAt (parameters x) job mem) lane=
        FlashAttn.address x.o i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val := by
      simp [ActualSource.address,base,stateAt,parameters,job,lane,hr,hd.1,hd.2,FlashAttn.address]
    rw [addr,accumulated_binding M dot encode x scale observe xs {s with mem := mem} binding job valid] at write
    exact write
  · exact member
  · intro j hj hw
    exact output_writer_unique x hs ho _ _ _ _
      ⟨i.1.isLt,i.2.1.isLt,i.2.2.1.isLt,i.2.2.2.1.isLt⟩ j ((program_valid x j).mp hj) hw
  · exact fun _ _ _ => rfl

#axiomsClean grid_output_tensor
#axiomsClean grid_log_tensor
#axiomsClean grid_output_injective
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PublicIO
