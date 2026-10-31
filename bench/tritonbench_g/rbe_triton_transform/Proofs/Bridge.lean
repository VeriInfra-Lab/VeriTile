import bench.tritonbench_g.rbe_triton_transform.Proofs.RbeTritonTransform_Source_Contract
import bench.tritonbench_g.rbe_triton_transform.Proofs.Schedule
import bench.tritonbench_g.rbe_triton_transform.Proofs.Scalar

namespace VeriTile.Bench.Repairs.RbeTritonTransform
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000

def programState (s : State α) (l : Location) : State α :=
  { s with pids := fun d => if d=0 then batch l else if d=1 then program l else 0 }

@[simp] theorem source_parts (r : Request) (input output : RegionName) :
    Source.parts (parameters r input output) = parts r := by
  unfold Source.parts parameters parts
  congr 1

@[simp] theorem source_row (r : Request) (input output : RegionName) (s : State α) (l : Location) :
    Source.row (parameters r input output) (programState s l) (lane l) = row r l := by
  simp [Source.row,parameters,programState,Source.parts,parts,row]

@[simp] theorem source_column (r : Request) (input output : RegionName) (s : State α) (l : Location) :
    Source.column (parameters r input output) (programState s l) (lane l) = evenColumn r l := by
  simp [Source.column,parameters,programState,Source.parts,parts,evenColumn]

theorem source_offset (r : Request) (input output : RegionName) (s : State α) (l : Location) :
    Source.inputOffset (parameters r input output) (programState s l) (imaginary l) (lane l) =
      r.offset (batch l) (row r l) (column r l) ∧
    Source.outputOffset (parameters r input output) (programState s l) (imaginary l) (lane l) =
      r.offset (batch l) (row r l) (column r l) := by
  simp only [Source.inputOffset,Source.outputOffset,source_row,source_column]
  dsimp [parameters,programState,Request.offset,column]
  constructor <;> ring

theorem source_active (r : Request) (input output : RegionName) (s : State α) (l : Location)
    (h : active r l) : Source.active (parameters r input output) (programState s l) (imaginary l) (lane l) := by
  unfold Source.active
  rw [source_row,source_column]
  dsimp [parameters]
  refine ⟨h.2.1,?_⟩
  have := h.2.2
  unfold column at this
  cases hi : imaginary l <;> simp_all [Nat.add_comm]

theorem source_value {α : Type} [Inhabited α] (ops : Operations α) (r : Request)
    (input output : RegionName) (s : State α) (l : Location) (h : active r l) :
    Source.value ops.backend (parameters r input output) (programState s l) (imaginary l) (lane l) =
      scalarResult ops r (fun o => (s.mem input o).read .fp32) (batch l) (row r l) (column r l) := by
  have hr := h.2.1
  have hn : evenColumn r l < r.columns := by
    have := h.2.2
    unfold column at this
    split at this <;> omega
  have hp : ((lane l).1.val : Int) + (r.position + (program l / parts r * 2 : Nat)) =
      r.position+(row r l : Int) := by
    simp only [row,Nat.cast_add]
    omega
  have he := column_pair r l
  unfold scalarResult
  rw [he.1,he.2]
  simp only [Source.value,Source.component,Source.phase,Source.active,
    source_row,source_column,source_parts,Source.inputOffset]
  simp only [Operations.backend,parameters,programState,↓reduceIte,one_mul,mul_one,id_eq]
  simp only [scalarPair,scalarAngle,Request.offset,hr,hn,true_and,if_true]
  have hoff : batch l*(r.rows*r.columns)+r.columns*row r l+evenColumn r l =
      (batch l*r.rows+row r l)*r.columns+evenColumn r l := by ring
  rw [hoff]
  simp only [Source.parts,parameters,parts] at hp ⊢
  simp only [Nat.one_ne_zero, Bool.false_eq_true, ↓reduceIte, add_zero, hn]
  rw [hp]
  cases hi : imaginary l <;> norm_num [hi,hr,hn, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc]

theorem source_apart (r : Request) (input output : RegionName) (s : State α) :
    Source.WritesApart (parameters r input output) s := by
  let p := parameters r input output
  have hoff (im : Bool) (i : Source.Index p) : Source.outputOffset p s im i =
      r.offset (s.pids 0) (Source.row p s i) (Source.column p s i + if im then 1 else 0) := by
    simp only [Source.outputOffset,p,parameters,Request.offset]
    ring
  have bounds (im : Bool) (i : Source.Index p) (hi : Source.active p s im i) :
      Source.row p s i<r.rows ∧ Source.column p s i+(if im then 1 else 0)<r.columns := by
    cases im <;> simpa [Source.active,p,Nat.add_comm] using hi
  constructor
  · intro im i j hi hj he
    rw [hoff,hoff] at he
    have hc := offset_injective r (bounds im i hi).1 (bounds im i hi).2
      (bounds im j hj).1 (bounds im j hj).2 he
    apply Prod.ext
    · apply Fin.ext
      have hh := hc.2.1
      dsimp [Source.row,p] at hh
      omega
    · apply Prod.ext
      · apply Fin.ext
        have hh := hc.2.2
        dsimp [Source.column,p] at hh
        omega
      · exact Subsingleton.elim _ _
  · intro i j hi hj he
    rw [hoff,hoff] at he
    have hc := offset_injective r (bounds Bool.false i hi).1 (bounds Bool.false i hi).2
      (bounds Bool.true j hj).1 (bounds Bool.true j hj).2 he
    have h1 : Source.column p s i%2=0 := by simp [Source.column,p,Nat.add_mod,Nat.mul_mod]
    have h2 : Source.column p s j%2=0 := by simp [Source.column,p,Nat.add_mod,Nat.mul_mod]
    have h3 := hc.2.2
    simp only [Bool.false_eq_true,↓reduceIte,add_zero] at h3
    omega

#axiomsClean source_apart
#axiomsClean source_value
#axiomsClean source_offset
end
end VeriTile.Bench.Repairs.RbeTritonTransform
