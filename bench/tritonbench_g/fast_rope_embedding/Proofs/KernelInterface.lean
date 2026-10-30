import bench.tritonbench_g.fast_rope_embedding.Kernels
import bench.tritonbench_g.rope_embedding.Proofs.KernelFP

namespace VeriTile.Bench.Repairs.FastRopeEmbedding.KernelInterface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding
noncomputable section
set_option maxHeartbeats 5000000

def rawParameters (r : Request) (qStride cosineStride sineStride : Nat) : Source.Parameters :=
  ⟨"Q","C","S",qStride,cosineStride,sineStride,r.sequence,r.dimension,r.heads,r.block⟩

def rawSource (r : Request) (qStride cosineStride sineStride : Nat) (backward : Bool) : ComputeKernel :=
  Kernels.rope "Q" "C" "S" qStride cosineStride sineStride
    r.sequence r.dimension r.heads r.block backward

/-- The existing complete source proof factored over raw state and scalar ABI.
Only the active row product is constrained, preserving arbitrary singleton strides. -/
theorem executes {α : Type} [Inhabited α] (op : Operations α)
    (r : Request) (qStride cosineStride sineStride : Nat) (backward : Bool)
    (s : State α) (row group : Nat) (launch : r.Launched row group)
    (strideEq : row*qStride=row*(r.heads*r.dimension))
    (cosineEq : cosineStride=r.half) (sineEq : sineStride=r.half) :
    ∃ final, FP.Structural.exec op.backend (rawSource r qStride cosineStride sineStride backward)
        (programState s row group)=some final ∧
      Kernel.Post r row group s final (fun h col=>scalarResult op r backward
        (fun o=>(s.mem "Q" o).read .real) (fun o=>(s.mem "C" o).read .real)
        (fun o=>(s.mem "S" o).read .real) row h col) := by
  have gh := group_head r group launch.2.2
  obtain ⟨final,he,hv,hf⟩ := Source.complete_correctness op.backend (rawParameters r qStride cosineStride sineStride) backward
    (programState s row group) (by simpa [programState,rawParameters] using Nat.le_of_lt gh)
  have values : ∀ h,group*4≤h → h<min (group*4+4) r.heads → ∀ side (i : Lane r),i.1.val<r.half →
      final.mem "Q" (Source.offset (parameters r) row h side i)=.mk .real
        (Source.value op.backend (parameters r) row h
          (Source.coefficient op.backend (parameters r) (programState s row group) "C" r.half)
          (fun j=>if backward then op.backend.unary (some .fp32) (.neg .real)
            (Source.coefficient op.backend (parameters r) (programState s row group) "S" r.half j)
            else Source.coefficient op.backend (parameters r) (programState s row group) "S" r.half j)
          (programState s row group) side i) := by
    intro h hlo hhi side i hi
    simpa only [rawParameters,parameters,Source.offset,Source.value,Source.coefficient,Source.active,
      programState,↓reduceIte,cosineEq,sineEq,strideEq] using hv h hlo hhi side i hi
  refine ⟨final,by simpa only [rawSource,rawParameters] using he,⟨?_,?_⟩⟩
  · intro h col hlo hhi hc
    change col<2*r.half at hc
    let j := if col<r.half then col else col-r.half
    have hj : j<r.half := by dsimp [j]; split <;> omega
    let i : Lane r := (⟨j,lt_of_lt_of_le hj (block_covers r)⟩,PUnit.unit)
    let side := decide (r.half≤col)
    let l : Location r := (row,group,h,side,i)
    have column_eq : column l=col := by
      by_cases h : col<r.half
      · simp [column,l,lane,RopeEmbedding.side,side,i,j,h,show ¬r.half≤col from by omega]
      · simp [column,l,lane,RopeEmbedding.side,side,i,j,h,show r.half≤col from by omega]
    have eq := values h hlo hhi side i hj
    have sourceEq : final.mem "Q" (r.offset row h (column l))=.mk .real (sourceValue op r backward s l) := by
      simpa only [sourceValue,programState,source_offset,column,
        RopeEmbedding.row,RopeEmbedding.group,head,RopeEmbedding.side,side,l,lane,↓reduceIte] using eq
    have act : active l := ⟨launch.2.1,lt_of_lt_of_le hhi (min_le_right _ _),hj⟩
    rw [source_value op r backward s l act] at sourceEq
    simpa only [column_eq,RopeEmbedding.row,head,l] using sourceEq
  · intro region o outside
    change ¬Kernel.Written r row group region o at outside
    apply hf region o
    by_cases hregion : region="Q"
    · right
      intro h hlo hhi side i hi equal
      apply outside
      refine ⟨hregion,h,i.1.val+if side then r.half else 0,?_,?_,?_,?_⟩
      · simpa [programState] using hlo
      · simpa [programState,rawParameters] using hhi
      · change i.1.val<r.half at hi
        cases side <;> simp only [Bool.false_eq_true,↓reduceIte] <;> omega
      · simp only [Source.offset,rawParameters,programState,↓reduceIte,strideEq] at equal
        simpa [Request.offset,Request.half,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using equal
    · exact Or.inl hregion

#axiomsClean executes
end
end VeriTile.Bench.Repairs.FastRopeEmbedding.KernelInterface
