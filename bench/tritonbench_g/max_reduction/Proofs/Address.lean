import bench.tritonbench_g.max_reduction.Proofs.ReductionModel

namespace VeriTile.Bench.Repairs.MaxReduction.Address
open VeriTile.Triton
noncomputable section
open Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 1000000
set_option linter.unusedSimpArgs false

def count (n width : Nat) : Nat := (n+width-1)/width
def start (width segment : Nat) : Nat := segment*width
def length (n width segment : Nat) : Nat := min (n-start width segment) width

def segments (n width : Nat) : List (Nat × Nat) :=
  (List.range (count n width)).map (fun c => (start width c,length n width c))

def coordinates (n width : Nat) : List Nat :=
  (List.range (count n width)).flatMap fun c =>
    (List.range (length n width c)).map (fun i => start width c+i)

theorem length_bound (n width c : Nat) : length n width c≤width := Nat.min_le_right _ _
theorem active (n width c i : Nat) (hi : i<length n width c) : start width c+i<n := by
  unfold length at hi
  have h := lt_of_lt_of_le hi (Nat.min_le_left _ _)
  omega

theorem ceil_covers (n width : Nat) (hw : 0<width) : n≤count n width*width := by
  have hrem := Nat.mod_lt (n+width-1) hw
  have hdiv := Nat.mod_add_div (n+width-1) width
  rw [Nat.mul_comm width] at hdiv
  unfold count
  omega

theorem segment_owner (n width i : Nat) (hw : 0<width) (hi : i<n) :
    i/width<count n width ∧ i%width<length n width (i/width) ∧
      start width (i/width)+i%width=i := by
  have hd := Nat.mod_add_div i width
  rw [Nat.mul_comm width] at hd
  have hr := Nat.mod_lt i hw
  have hc := ceil_covers n width hw
  refine ⟨(Nat.div_lt_iff_lt_mul hw).mpr (hi.trans_le hc),?_,?_⟩
  · unfold length start
    exact lt_min (by omega) hr
  · unfold start; omega

theorem full_coverage (n width i : Nat) (hw : 0<width) :
    i∈coordinates n width ↔ i<n := by
  simp only [coordinates,List.mem_flatMap,List.mem_range,List.mem_map]
  constructor
  · rintro ⟨c,_,j,hj,rfl⟩
    exact active n width c j hj
  · intro hi
    obtain ⟨hc,hr,he⟩ := segment_owner n width i hw hi
    exact ⟨i/width,hc,i%width,hr,he⟩

/-- Unique ownership is arithmetic, not an assumption about the output. -/
theorem owner_unique (width c i d j : Nat) (hw : 0<width)
    (hi : i<width) (hj : j<width) (same : start width c+i=start width d+j) :
    c=d ∧ i=j := by
  have hc : (start width c+i)/width=c := by rw [start,Nat.mul_comm c width,Nat.mul_add_div hw,Nat.div_eq_of_lt hi,Nat.add_zero]
  have hd : (start width d+j)/width=d := by rw [start,Nat.mul_comm d width,Nat.mul_add_div hw,Nat.div_eq_of_lt hj,Nat.add_zero]
  have he : c=d := by rw [←hc,←hd,same]
  exact ⟨he,by simpa [start,he] using same⟩

def inputOffset (N K m n k : Nat) := m*N*K+n*K+k
def outputOffset (K m k : Nat) := m*K+k

theorem input_bound (M N K m n k : Nat) (hm : m<M) (hn : n<N) (hk : k<K) :
    inputOffset N K m n k<M*N*K := by
  have hmn : m*N+n<M*N := by
    have h₁ := Nat.mul_le_mul_right N hm
    have h₂ : m*N+N≤M*N := by simpa [Nat.add_mul] using h₁
    omega
  have h₁ := Nat.mul_le_mul_right K hmn
  have h₂ : (m*N+n)*K+K≤M*N*K := by simpa [Nat.add_mul] using h₁
  simp only [inputOffset]
  nlinarith

theorem output_bound (M K m k : Nat) (hm : m<M) (hk : k<K) :
    outputOffset K m k<M*K := by
  have h := Nat.mul_le_mul_right K hm
  simp only [outputOffset]
  nlinarith

theorem output_injective (K m k m' k' : Nat) (hk : k<K) (hk' : k'<K)
    (same : outputOffset K m k=outputOffset K m' k') : m=m' ∧ k=k' :=
  owner_unique K m k m' k' (Nat.zero_lt_of_lt hk) hk hk' same

def word (n : Nat) : BitVec 64 := BitVec.ofNat 64 n

def nativeInput (N K m n k : Nat) : BitVec 64 :=
  word m*word N*word K+word n*word K+word k

def nativeOutput (K m k : Nat) : BitVec 64 := word m*word K+word k

theorem native_input_polynomial (N K m n k : Nat) :
    nativeInput N K m n k=word (inputOffset N K m n k) := by
  simp [nativeInput,inputOffset,word,BitVec.ofNat_add,BitVec.ofNat_mul]

theorem native_output_polynomial (K m k : Nat) : nativeOutput K m k=word (outputOffset K m k) := by
  simp [nativeOutput,outputOffset,word,BitVec.ofNat_add,BitVec.ofNat_mul]

end
end VeriTile.Bench.Repairs.MaxReduction.Address
