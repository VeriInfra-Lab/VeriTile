import bench.tritonbench_g.matmul_dequant_int4.Sources.Autotune
import bench.tritonbench_g.matmul_dequant_int4.Sources.HelperBytes
import bench.tritonbench_g.matmul_dequant_int4.Sources.S1
namespace VeriTile.Bench.Repairs.MatmulDequantInt4.Pipeline
open Physical
set_option maxHeartbeats 5000000
noncomputable section

/-- Ordinary torch.ones allocation payload semantics, before the launch. -/
def ones (s : Shape) (one : α) (initial : State α) : State α :=
  {initial with values := fun r a => if r=.dequant ∧ a<s.k*s.n then one else initial.values r a}
theorem ones_readonly (s : Shape) (one : α) (initial : State α) :
    Writes.Readonly .dequant initial (ones s one initial) := by
  refine ⟨?_,rfl⟩
  intro r a hr;simp [ones,hr]
def onesBytes (codec : Codec α) (s : Shape) (one : α) (initial : Bytes) : Bytes :=
  fun r b => if r=.value .dequant ∧ b/codec.width<s.k*s.n then
    codec.encode one ⟨b%codec.width,Nat.mod_lt _ codec.positive⟩ else initial r b
theorem ones_load (codec : Codec α) (s : Shape) (one : α) (initial : Bytes) :
    load codec (onesBytes codec s one initial)=ones s one (load codec initial) := by
  have hd (a : Nat) (i : Fin codec.width) : (codec.width*a+i.val)/codec.width=a := by
    rw [Nat.mul_comm codec.width a];exact div_coordinate _ _ _ codec.positive i.isLt
  have hm (a : Nat) (i : Fin codec.width) : (codec.width*a+i.val)%codec.width=i.val := by
    simp [Nat.add_mod,Nat.mod_eq_of_lt i.isLt]
  change State.mk _ _=State.mk _ _
  rw [State.mk.injEq]
  constructor
  · funext r a
    by_cases h : r=.dequant ∧ a<s.k*s.n
    · rcases h with ⟨rfl,ha⟩
      simp only [load,onesBytes,hd,ha,and_self,ite_true,ones]
      have he : (fun i : Fin codec.width => codec.encode one
          ⟨(codec.width*a+i.val)%codec.width,Nat.mod_lt _ codec.positive⟩)=codec.encode one := by
        funext i;congr 1;exact Fin.ext (hm a i)
      rw [he,codec.decode_encode]
    · simp [load,onesBytes,hd,ones,h]
  · funext r a;simp [load,onesBytes,ones]

def trialsBytes (codec : Codec α) (s : Shape) (p : DequantOps α) : List (List Work) → Bytes → Bytes
  | [],m => m
  | order::rest,m => trialsBytes codec s p rest (Physical.run codec .dequant s.n (dequantValue s p) order m)
theorem trials_load (codec : Codec α) (s : Shape) (p : DequantOps α) (trials : List (List Work)) (initial : Bytes) :
    load codec (trialsBytes codec s p trials initial)=Autotune.dequantTrials s p trials (load codec initial) := by
  induction trials generalizing initial with
  | nil => rfl
  | cons order rest ih =>
    change load codec (trialsBytes codec s p rest _)=_
    rw [ih,Physical.run_load,←dequant_run_writes];rfl

def prepared (s : Shape) (one : α) (p : DequantOps α) (trials : List (List Work)) (initial : State α) :=
  Autotune.dequantTrials s p trials (ones s one initial)
def preparedBytes (codec : Codec α) (s : Shape) (one : α) (p : DequantOps α)
    (trials : List (List Work)) (initial : Bytes) := trialsBytes codec s p trials (onesBytes codec s one initial)
theorem prepared_load (codec : Codec α) (s : Shape) (one : α) (p : DequantOps α)
    (trials : List (List Work)) (initial : Bytes) :
    load codec (preparedBytes codec s one p trials initial)=prepared s one p trials (load codec initial) := by
  unfold preparedBytes prepared;rw [trials_load,ones_load]
theorem prepared_readonly (s : Shape) (one : α) (p : DequantOps α)
    (trials : List (List Work)) (initial : State α) :
    Writes.Readonly .dequant initial (prepared s one p trials initial) :=
  Autotune.dequant_readonly s p trials initial _ (ones_readonly s one initial)

def dequantBytes (codec : Codec α) (s : Shape) (one : α) (p : DequantOps α)
    (trials : List (List Work)) (order : List Work) (initial : Bytes) :=
  Physical.run codec .dequant s.n (dequantValue s p) order (preparedBytes codec s one p trials initial)
theorem dequant_load (codec : Codec α) (s : Shape) (one : α) (p : DequantOps α)
    (trials : List (List Work)) (order : List Work) (initial : Bytes) :
    load codec (dequantBytes codec s one p trials order initial)=
      dequantRun s p order (prepared s one p trials (load codec initial)) := by
  unfold dequantBytes;rw [Physical.run_load,prepared_load,←dequant_run_writes]
theorem dequant_result (codec : Codec α) (s : Shape) (one : α) (p : DequantOps α)
    (trials : List (List Work)) (g : Grid s.k s.n) (order : List Work) (complete : g.Complete order)
    (initial : Bytes) (k n : Nat) (hk : k<s.k) (hn : n<s.n) (i : Fin codec.width) :
    dequantBytes codec s one p trials order initial (.value .dequant) (codec.width*(k*s.n+n)+i.val)=
      codec.encode (dequantValue s p (load codec initial) ⟨k,n⟩) i := by
  rw [←original_byte codec (dequantBytes codec s one p trials order initial) .dequant (k*s.n+n) i,dequant_load]
  unfold prepared
  rw [Autotune.dequant_absorbed s p trials g order complete (load codec initial)
    (ones s one (load codec initial)) (ones_readonly s one _) k n hk hn]

def s1Bytes (codec : Codec α) (s : Shape) (one : α) (p : DequantOps α) (lib : Library.Ops α β)
    (plan : Library.Selected s) (trials : List (List Work)) (dequantOrder outputOrder : List Work) (initial : Bytes) :=
  Physical.run codec .output s.n (Library.value s lib plan) outputOrder
    (dequantBytes codec s one p trials dequantOrder initial)
theorem s1_load (codec : Codec α) (s : Shape) (one : α) (p : DequantOps α) (lib : Library.Ops α β)
    (plan : Library.Selected s) (trials : List (List Work)) (dequantOrder outputOrder : List Work) (initial : Bytes) :
    load codec (s1Bytes codec s one p lib plan trials dequantOrder outputOrder initial)=
      S1.run s p lib plan dequantOrder outputOrder (prepared s one p trials (load codec initial)) := by
  unfold s1Bytes S1.run Library.run;rw [Physical.run_load,dequant_load]

theorem reference_readonly (s : Shape) (p : DequantOps α) (lib : Library.Ops α β)
    (plan : Library.Selected s) (initial current : State α) (same : Writes.Readonly .dequant initial current)
    (r n : Nat) : S1.reference s p lib plan current r n=S1.reference s p lib plan initial r n := by
  have ha : current.values .a=initial.values .a := funext (fun k => same.1 .a k (by decide))
  have hd : (fun k : Fin s.k => dequantValue s p current ⟨k,n⟩)=
      (fun k : Fin s.k => dequantValue s p initial ⟨k,n⟩) :=
    funext (fun k => dequant_stable s p initial current same ⟨k,n⟩)
  unfold S1.reference;rw [ha,hd]

theorem s1_result (codec : Codec α) (s : Shape) (one : α) (p : DequantOps α) (lib : Library.Ops α β)
    (plan : Library.Selected s) (trials : List (List Work)) (dg : Grid s.k s.n) (og : Grid s.m s.n)
    (ds os : List Work) (hd : dg.Complete ds) (ho : og.Complete os) (initial : Bytes)
    (r n : Nat) (hr : r<s.m) (hn : n<s.n) (i : Fin codec.width) :
    s1Bytes codec s one p lib plan trials ds os initial (.value .output) (codec.width*(r*s.n+n)+i.val)=
      codec.encode (S1.reference s p lib plan (load codec initial) r n) i := by
  rw [←original_byte codec (s1Bytes codec s one p lib plan trials ds os initial) .output (r*s.n+n) i,s1_load,S1.whole s p lib plan dg og ds os hd ho _ r n hr hn,
    reference_readonly s p lib plan _ _ (prepared_readonly s one p trials _) r n]

theorem trials_frame (codec : Codec α) (s : Shape) (p : DequantOps α) (trials : List (List Work))
    (initial : Bytes) (r : Allocation) (b : Nat) (hr : r≠.value .dequant) :
    trialsBytes codec s p trials initial r b=initial r b := by
  induction trials generalizing initial with
  | nil => rfl
  | cons order rest ih =>
    change trialsBytes codec s p rest _ r b=_
    rw [ih];apply Physical.frame
    intro w hw h;exact hr h.1
theorem caller_frame (codec : Codec α) (s : Shape) (one : α) (p : DequantOps α) (lib : Library.Ops α β)
    (plan : Library.Selected s) (trials : List (List Work)) (ds os : List Work) (initial : Bytes)
    (r : Allocation) (b : Nat) (hd : r≠.value .dequant) (ho : r≠.value .output) :
    s1Bytes codec s one p lib plan trials ds os initial r b=initial r b := by
  unfold s1Bytes
  rw [Physical.frame codec .output s.n _ os _ r b (fun _ _ h => ho h.1)]
  unfold dequantBytes
  rw [Physical.frame codec .dequant s.n _ ds _ r b (fun _ _ h => hd h.1)]
  unfold preparedBytes
  rw [trials_frame codec s p trials _ r b hd]
  simp [onesBytes,hd]

#axiomsClean ones_readonly
#axiomsClean ones_load
#axiomsClean trials_load
#axiomsClean prepared_load
#axiomsClean prepared_readonly
#axiomsClean dequant_load
#axiomsClean dequant_result
#axiomsClean s1_load
#axiomsClean reference_readonly
#axiomsClean s1_result
#axiomsClean caller_frame
end
end VeriTile.Bench.Repairs.MatmulDequantInt4.Pipeline
