import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelHalfComplete
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelDefaultComplete
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Coordinates
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Contract

namespace VeriTile.Bench.Repairs.Conv2d.KernelBinding
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-- Exact successful scalar binding, including the original redundant grouped
feature grid and the actual positive input/weight strides. Pointer bases and
storage offsets live in the memory observation, never in a dense-layout guess. -/
def parameters (s : Shape) (flags : Flags) : KernelSource.Parameters where
  Input := "input"
  Weight := "weight"
  Output := "output"
  batch_dim := s.n
  in_feat_dim := s.channels
  in_height := s.h
  in_width := s.w
  out_feat_dim := s.outputs
  out_height := s.oh
  out_width := s.ow
  input_batch_stride := s.sn
  input_in_feat_stride := s.sc
  input_height_stride := s.sx
  input_width_stride := s.sy
  weight_out_feat_stride := s.so
  weight_in_feat_stride := s.tc
  weight_height_stride := s.tx
  weight_width_stride := s.ty
  output_batch_stride := s.outputs*s.oh*s.ow
  output_out_feat_stride := s.oh*s.ow
  output_height_stride := s.ow
  output_width_stride := 1
  kernel_height := s.kh
  kernel_width := s.kw
  stride_height := s.sh
  stride_width := s.sw
  padding_height := s.ph
  padding_width := s.pw
  groups := s.groups
  BLOCK_SIZE_BATCH_HEIGHT_WIDTH := 128
  BLOCK_SIZE_IN_FEAT := 32
  BLOCK_SIZE_OUT_FEAT := 32
  tf32 := flags.tf32

abbrev Index := TileIndex [128,32]
def lane (i : Index) : Lane := (i.1.val,i.2.1.val)
def cell (pid : Program) (i : Index) : Conv2d.Cell := cellOf pid (lane i)
def atProgram {α} (pid : Program) (t : State α) : State α :=
  {t with pids := fun k => if k=0 then pid.1 else if k=1 then pid.2.1 else if k=2 then pid.2.2 else 0}

@[simp] theorem groups (s : Shape) (v : Valid s) (f : Flags) :
    KernelHalf.igd (parameters s f)=s.cg ∧ KernelHalf.ogd (parameters s f)=s.og :=
  group_division s v

theorem active_iff {α} (s : Shape) (v : Valid s) (f : Flags) (pid : Program)
    (hg : pid.2.2<s.groups) (t : State α) (i : Index) :
    KernelHalf.active (parameters s f) (atProgram pid t) i ↔ s.Active (cell pid i) := by
  have gd := (groups s v f).2
  have hp := plane_pos s v
  have hh := Nat.mod_lt ((pid.1*128+i.1.val)/s.ow) v.oh
  have hw := Nat.mod_lt (pid.1*128+i.1.val) v.ow
  simp only [KernelHalf.active,KernelHalf.batch,KernelHalf.bh,KernelHalf.bhw,
    KernelHalf.feat,KernelHalf.oh,KernelHalf.ow,KernelHalf.ogd,parameters,KernelHalf.BHW,
    KernelHalf.OF,atProgram,reduceIte,Shape.Active,cell,cellOf,lane,
    Shape.outputs,Nat.mul_div_cancel_left _ v.groups]
  rw [Nat.div_div_eq_div_mul, Nat.mul_comm s.ow s.oh]
  have hr : (pid.1*128+i.1.val)/(s.oh*s.ow)<s.n ↔ pid.1*128+i.1.val<s.spatial := by
    simpa [Shape.spatial,Shape.plane] using (Nat.div_lt_iff_lt_mul hp)
  simp only [hr,hh,hw,and_true]
  tauto

theorem output_address {α} (s : Shape) (v : Valid s) (f : Flags) (pid : Program)
    (t : State α) (i : Index) :
    KernelHalf.outputOffset (parameters s f) (atProgram pid t) i=outputPointer s (cell pid i) := by
  have gd := (groups s v f).2
  simp only [KernelHalf.outputOffset,KernelHalf.batch,KernelHalf.bh,KernelHalf.bhw,
    KernelHalf.feat,KernelHalf.oh,KernelHalf.ow,KernelHalf.ogd,parameters,KernelHalf.BHW,
    KernelHalf.OF,atProgram,reduceIte,outputPointer,batch,outH,outW,cell,cellOf,lane,
    Shape.outputs,Nat.mul_div_cancel_left _ v.groups]
  norm_num
  ring

theorem lane_mem (i : Index) : lane i∈lanes := by
  simp only [lane,lanes,List.mem_product,List.mem_range]
  exact ⟨i.1.isLt,i.2.1.isLt⟩

theorem index_injective {i j : Index} (h : lane i=lane j) : i=j := by
  have a : i.1=j.1 := Fin.ext (congrArg Prod.fst h)
  have b : i.2.1=j.2.1 := Fin.ext (congrArg Prod.snd h)
  exact Prod.ext a (Prod.ext b (Subsingleton.elim _ _))

theorem stores_unique {α} (s : Shape) (v : Valid s) (f : Flags) (pid : Program)
    (hg : pid.2.2<s.groups) (t : State α) :
    ∀ i j, KernelHalf.active (parameters s f) (atProgram pid t) i →
      KernelHalf.active (parameters s f) (atProgram pid t) j →
      KernelHalf.outputOffset (parameters s f) (atProgram pid t) i=
        KernelHalf.outputOffset (parameters s f) (atProgram pid t) j → i=j := by
  intro i j hi hj eq
  rw [output_address s v f,output_address s v f] at eq
  have h := address_injective s v _ _ ((active_iff s v f pid hg t i).mp hi)
    ((active_iff s v f pid hg t j).mp hj) eq
  exact index_injective (source_owner_unique pid pid _ _ (lane_mem i) (lane_mem j) h).2

/-- The source grid covers every logical output exactly once, including
channel tails and redundant feature programs from the unchanged wrapper. -/
theorem covers (s : Shape) (v : Valid s) (x : Conv2d.Cell) (hx : s.Active x) :
    ∃ pid∈programs s,∃ i : Index,cell pid i=x := by
  obtain ⟨pid,hp,l,hl,he⟩ := source_grid_covers s v x hx
  have hb : l.1<128 ∧ l.2<32 := by rcases l with ⟨a,b⟩; simpa only [lanes,List.mem_product,List.mem_range] using hl
  exact ⟨pid,hp,(⟨l.1,hb.1⟩,⟨l.2,hb.2⟩,PUnit.unit),he⟩

#axiomsClean active_iff
#axiomsClean output_address
#axiomsClean stores_unique
#axiomsClean covers
end
end VeriTile.Bench.Repairs.Conv2d.KernelBinding
