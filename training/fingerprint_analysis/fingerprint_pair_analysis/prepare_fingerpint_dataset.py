import os, random, shutil, glob

SEED      = 42
SRC_ROOT  = "datasets/vox1_cleaned/wav_dev"
DST_ROOT  = "datasets/vox1_fingerprint_analysis"
N_PERSONS = 5
N_RECORDS = 3
N_SAMPLES = 4
MIN_FILES = 4

random.seed(SEED)

persons = sorted(os.listdir(SRC_ROOT))
random.shuffle(persons)

selected = []
for person in persons:
    person_dir = os.path.join(SRC_ROOT, person)
    if not os.path.isdir(person_dir):
        continue
    records = sorted(os.listdir(person_dir))
    valid_records = [
        r for r in records
        if os.path.isdir(os.path.join(person_dir, r))
        and len(glob.glob(os.path.join(person_dir, r, "*.wav"))) >= MIN_FILES
    ]
    if len(valid_records) < N_RECORDS:
        continue
    chosen = random.sample(valid_records, N_RECORDS)
    selected.append((person, chosen))
    if len(selected) == N_PERSONS:
        break

if len(selected) < N_PERSONS:
    raise RuntimeError(f"Only found {len(selected)} qualifying persons (need {N_PERSONS})")

for person, records in selected:
    for record in records:
        src_dir = os.path.join(SRC_ROOT, person, record)
        dst_dir = os.path.join(DST_ROOT, person, record)
        os.makedirs(dst_dir, exist_ok=True)
        wav_files = sorted(glob.glob(os.path.join(src_dir, "*.wav")))[:N_SAMPLES]
        for wav in wav_files:
            shutil.copy2(wav, dst_dir)
        print(f"  copied {len(wav_files)} files → {os.path.relpath(dst_dir)}")

print(f"\nDone: {N_PERSONS} persons × {N_RECORDS} records × {N_SAMPLES} samples "
      f"= {N_PERSONS * N_RECORDS * N_SAMPLES} total files")
