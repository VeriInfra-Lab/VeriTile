import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.PublicIO
import bench.repairs.tritonbench_g.rotary_transform.Proofs.Bounds

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.KernelValues
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural PublicIO
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

/-- Equality on logical input values only, independent of execution. -/
structure Agree (c : Config) (left right : Memory α) : Prop where
  input : ∀ i,i<dataSize c.shape → left .input i=right .input i
  cosine : ∀ i,i<coefficientSize c.shape → left .cosine i=right .cosine i
  sine : ∀ i,i<coefficientSize c.shape → left .sine i=right .sine i

theorem input_agree (p : Ops α) (obs : α → α) (c : Config) (cu64 off64 : Bool)
    (vals : ChannelValues α (inputs c)) (initial : State α)
    (hin : (io p obs c cu64 off64).InputBinding vals initial) :
    Agree c (KernelAlgebra.memory initial) (arrayMemory c vals) := by
  constructor
  · intro i hi
    simpa [io,inputs,arrayMemory,hi,KernelAlgebra.memory,KernelAlgebra.regionName,mapValue] using
      hin ⟨0,by simp [io,inputs]⟩ (⟨i,hi⟩,()) trivial
  · intro i hi
    simpa [io,inputs,arrayMemory,hi,KernelAlgebra.memory,KernelAlgebra.regionName,mapValue] using
      hin ⟨1,by simp [io,inputs]⟩ (⟨i,hi⟩,()) trivial
  · intro i hi
    simpa [io,inputs,arrayMemory,hi,KernelAlgebra.memory,KernelAlgebra.regionName,mapValue] using
      hin ⟨2,by simp [io,inputs]⟩ (⟨i,hi⟩,()) trivial

theorem real_formula (c : Config) (valid : c.Valid) (left right : Memory ℝ)
    (agree : Agree c left right) (batch row head col : Nat)
    (hb : batch<c.shape.batches) (hr : row<c.length batch)
    (hh : head<c.shape.heads) (hd : col<c.shape.width) :
    realFormula c left batch row head col=realFormula c right batch row head col := by
  by_cases rotary : col<2*c.shape.half
  · obtain ⟨hp,_⟩ := column_inverse c.shape col rotary
    let w : Work := ⟨batch,row,head,c.shape.pairOf col⟩
    have live : c.Active w := ⟨hb,hr,hh,hp⟩
    have a := agree.input (c.cell w Bool.false) (cell_bound c valid w live Bool.false)
    have b := agree.input (c.cell w Bool.true) (cell_bound c valid w live Bool.true)
    have co (cached : row+c.shift batch<c.shape.coefficientRows) :=
      agree.cosine (c.coeff w) (coeff_bound c w live cached)
    have sn (cached : row+c.shift batch<c.shape.coefficientRows) :=
      agree.sine (c.coeff w) (coeff_bound c w live cached)
    simp only [w,Config.cell,Config.coeff] at a b co sn
    simp only [realFormula,if_pos rotary,a,b]
    by_cases cached : row+c.shift batch<c.shape.coefficientRows
    · simp only [if_pos cached,co cached,sn cached]
    · simp only [if_neg cached]
  · simp only [realFormula,if_neg rotary]
    exact agree.input _ (Host.address_bound c.shape _ _ _ (row_bound c valid batch row hb hr) hh hd)

theorem native_formula (c : Config) (valid : c.Valid) (p : Native.Instructions)
    (left right : Memory Native.Node) (agree : Agree c left right)
    (batch row head col : Nat) (hb : batch<c.shape.batches) (hr : row<c.length batch)
    (hh : head<c.shape.heads) (hd : col<c.shape.width) :
    nativeFormula c p (Native.project left) batch row head col=
      nativeFormula c p (Native.project right) batch row head col := by
  by_cases rotary : col<2*c.shape.half
  · obtain ⟨hp,_⟩ := column_inverse c.shape col rotary
    let w : Work := ⟨batch,row,head,c.shape.pairOf col⟩
    have live : c.Active w := ⟨hb,hr,hh,hp⟩
    have a := congrArg Native.Node.value
      (agree.input (c.cell w Bool.false) (cell_bound c valid w live Bool.false))
    have b := congrArg Native.Node.value
      (agree.input (c.cell w Bool.true) (cell_bound c valid w live Bool.true))
    have co (cached : row+c.shift batch<c.shape.coefficientRows) := congrArg Native.Node.value
      (agree.cosine (c.coeff w) (coeff_bound c w live cached))
    have sn (cached : row+c.shift batch<c.shape.coefficientRows) := congrArg Native.Node.value
      (agree.sine (c.coeff w) (coeff_bound c w live cached))
    simp only [w,Config.cell,Config.coeff] at a b co sn
    simp only [nativeFormula,if_pos rotary,Native.project,a,b]
    by_cases cached : row+c.shift batch<c.shape.coefficientRows
    · simp only [if_pos cached,co cached,sn cached]
    · simp only [if_neg cached]
  · simp only [nativeFormula,if_neg rotary,Native.project]
    exact congrArg Native.Node.value
      (agree.input _ (Host.address_bound c.shape _ _ _ (row_bound c valid batch row hb hr) hh hd))

theorem real_result (c : Config) (valid : c.Valid) (cu64 off64 : Bool)
    (vals : ChannelValues ℝ (inputs c)) (legal : domain c vals) (initial : State ℝ)
    (hin : (io realOps id c cu64 off64).InputBinding vals initial) :
    ∃ final,runSchedule ((io realOps id c cu64 off64).schedule vals) initial=some final ∧
      (io realOps id c cu64 off64).OutputReadback vals (expected c (realFormula c) vals) final ∧
      (io realOps id c cu64 off64).Frame vals initial final := by
  obtain ⟨final,ran,values,unchanged⟩ := KernelEndpoints.real c valid initial
    (metadata_binding realOps id c cu64 off64 vals legal initial hin)
    (tail_ready realOps id c cu64 off64 vals legal initial hin) cu64 off64
  refine ⟨final,(schedule_eq realOps id c cu64 off64 vals initial).trans ran,?_,
    frame realOps id c valid cu64 off64 vals initial final unchanged⟩
  intro port lane active
  fin_cases port
  have got := (values lane.1.val lane.2.1.val lane.2.2.1.val lane.2.2.2.1.val
    lane.1.isLt active lane.2.2.1.isLt lane.2.2.2.1.isLt).trans
    (KernelValues.real_formula c valid _ _
      (KernelValues.input_agree realOps id c cu64 off64 vals initial hin)
      _ _ _ _ lane.1.isLt active lane.2.2.1.isLt lane.2.2.2.1.isLt)
  simpa only [io,outputs,expected,mapValue,id_eq,output_region] using got

theorem native_result (c : Config) (valid : c.Valid) (p : Native.Instructions) (cu64 off64 : Bool)
    (vals : ChannelValues Native.Node (inputs c)) (legal : domain c vals) (initial : State Native.Node)
    (hin : (io (Native.ops p) (fun node => Native.atom node.value) c cu64 off64).InputBinding vals initial) :
    ∃ final,runSchedule ((io (Native.ops p) (fun node => Native.atom node.value) c cu64 off64).schedule vals) initial=some final ∧
      (io (Native.ops p) (fun node => Native.atom node.value) c cu64 off64).OutputReadback vals
        (expected c (fun values b r h d => Native.atom (nativeFormula c p (Native.project values) b r h d)) vals) final ∧
      (io (Native.ops p) (fun node => Native.atom node.value) c cu64 off64).Frame vals initial final := by
  let obs : Native.Node → Native.Node := fun node => Native.atom node.value
  obtain ⟨final,ran,values,unchanged⟩ := KernelEndpoints.native c valid p initial
    (metadata_binding (Native.ops p) obs c cu64 off64 vals legal initial hin)
    (tail_ready (Native.ops p) obs c cu64 off64 vals legal initial hin) cu64 off64
  refine ⟨final,(schedule_eq (Native.ops p) obs c cu64 off64 vals initial).trans ran,?_,
    frame (Native.ops p) obs c valid cu64 off64 vals initial final unchanged⟩
  intro port lane active
  fin_cases port
  have got := (values lane.1.val lane.2.1.val lane.2.2.1.val lane.2.2.2.1.val
    lane.1.isLt active lane.2.2.1.isLt lane.2.2.2.1.isLt).trans
    (KernelValues.native_formula c valid p _ _
      (KernelValues.input_agree (Native.ops p) obs c cu64 off64 vals initial hin)
      _ _ _ _ lane.1.isLt active lane.2.2.1.isLt lane.2.2.2.1.isLt)
  simpa only [io,outputs,expected,mapValue,output_region] using congrArg Native.atom got

#axiomsClean real_result
#axiomsClean native_result
#axiomsClean input_agree
#axiomsClean real_formula
#axiomsClean native_formula
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.KernelValues
