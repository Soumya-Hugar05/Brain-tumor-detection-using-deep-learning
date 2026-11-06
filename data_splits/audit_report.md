# Dataset audit and split report

Status: **PASSED**  
Generated: 2026-10-31 18:26:20  
Hash rule: identical MD5, or both pHash and dHash distance <= 4 of 64 bits  
Pixel rule: 64px thumbnail corr >= 0.93 (original or mirrored) -> 256px corr >= 0.95 and 32px-block corr p10 >= 0.85 in the better orientation  
Related candidates linked: False  
Ratios train/val/test (of eligible images): [0.7, 0.15, 0.15]  |  Seed: 1337  
Dataset fingerprint (SHA-256 of path + MD5 list): `00bcb879ebc8a95a902b39bd25901b244bf4f7f93d8468487761b4a0b071d131`  
Manifest SHA-256: `4d09b01d782fa076890f32bf30bcbbe1a2319c135ead2e58cbd5e1f9ec0d930f`

## Totals

- Original images: **7200**
- Eligible images (assigned to train/val/test): **6968**
- Excluded images: **232**

## Excluded images

| Reason | Count | By class | By original folder | Why |
|---|---|---|---|---|
| `label_conflict_copy` | 29 | {'glioma': 2, 'meningioma': 14, 'pituitary': 13} | {'Testing': 8, 'Training': 21} | Copy of a scan that is also filed under another class folder; the true label is unknown. |
| `pre_augmented` | 203 | {'meningioma': 203} | {'Testing': 103, 'Training': 100} | Pre-augmented copy shipped with the dataset (filename contains '-aug-'); geometric transforms evade perceptual hashing, so it could leak across splits. |

## Splits

| Split | Images | Groups | glioma | meningioma | notumor | pituitary | From Training | From Testing |
|---|---|---|---|---|---|---|---|---|
| train | 4877 | 2391 | 1258 | 1108 | 1260 | 1251 | 3831 | 1046 |
| val | 1046 | 506 | 270 | 238 | 270 | 268 | 825 | 221 |
| test | 1045 | 508 | 270 | 237 | 270 | 268 | 823 | 222 |
| excluded | 232 | 208 | 2 | 217 | 0 | 13 | 121 | 111 |

## Duplicates found (all original images)

- Groups: 3577  |  multi-image groups: 1975 (all members byte-identical 67, other 1908)
- Images sharing a group with at least one other image: 5598
- Largest group: 46 images
- Links by type: {'pixel_mirror': 5630, 'near_hash': 4406, 'exact_md5': 227, 'pixel_same': 605}
- Pixel-confirmed links not found by hashing: 6235 (mirrored 5630, same orientation 605)
- Unconfirmed pixel candidates (not linked, see review_pairs.csv): 4848

## Mixed-label groups

### g00067 (3 images, all excluded)

| Path | Label | Original folder | Split | Exclusion |
|---|---|---|---|---|
| `Testing/glioma/Te-gl_160.jpg` | glioma | Testing | excluded | label_conflict_copy |
| `Testing/meningioma/Te-me_159.jpg` | meningioma | Testing | excluded | label_conflict_copy |
| `Training/meningioma/Tr-me_1122.jpg` | meningioma | Training | excluded | label_conflict_copy |

Cross-label links: `Testing/glioma/Te-gl_160.jpg` ~ `Testing/meningioma/Te-me_159.jpg` (near_hash, same, corr 0.9963, block p10 0.9901); `Testing/glioma/Te-gl_160.jpg` ~ `Training/meningioma/Tr-me_1122.jpg` (near_hash, same, corr 0.9963, block p10 0.9901)

### g00327 (7 images, all excluded)

| Path | Label | Original folder | Split | Exclusion |
|---|---|---|---|---|
| `Testing/glioma/Te-gl_68.jpg` | glioma | Testing | excluded | label_conflict_copy |
| `Testing/meningioma/Te-me_50.jpg` | meningioma | Testing | excluded | label_conflict_copy |
| `Testing/meningioma/Te-me_61.jpg` | meningioma | Testing | excluded | label_conflict_copy |
| `Testing/meningioma/Te-me_96.jpg` | meningioma | Testing | excluded | label_conflict_copy |
| `Training/meningioma/Tr-me_458.jpg` | meningioma | Training | excluded | label_conflict_copy |
| `Training/meningioma/Tr-me_468.jpg` | meningioma | Training | excluded | label_conflict_copy |
| `Training/meningioma/Tr-me_714.jpg` | meningioma | Training | excluded | label_conflict_copy |

Cross-label links: `Testing/glioma/Te-gl_68.jpg` ~ `Testing/meningioma/Te-me_50.jpg` (near_hash, same, corr 0.9978, block p10 0.9128); `Testing/glioma/Te-gl_68.jpg` ~ `Testing/meningioma/Te-me_61.jpg` (near_hash, same, corr 0.999, block p10 0.9588); `Testing/glioma/Te-gl_68.jpg` ~ `Testing/meningioma/Te-me_96.jpg` (near_hash, same, corr 0.999, block p10 0.9591); `Testing/glioma/Te-gl_68.jpg` ~ `Training/meningioma/Tr-me_458.jpg` (near_hash, same, corr 0.999, block p10 0.9591); `Testing/glioma/Te-gl_68.jpg` ~ `Training/meningioma/Tr-me_468.jpg` (near_hash, same, corr 0.9978, block p10 0.9128); `Testing/glioma/Te-gl_68.jpg` ~ `Training/meningioma/Tr-me_714.jpg` (near_hash, same, corr 0.999, block p10 0.9588)

### g00334 (5 images, has eligible images)

| Path | Label | Original folder | Split | Exclusion |
|---|---|---|---|---|
| `Testing/glioma/Te-gl_75.jpg` | glioma | Testing | train | - |
| `Testing/meningioma/Te-aug-me_36.jpg` | meningioma | Testing | excluded | pre_augmented |
| `Testing/meningioma/Te-me_283.jpg` | meningioma | Testing | train | - |
| `Training/glioma/Tr-gl_988.jpg` | glioma | Training | train | - |
| `Training/meningioma/Tr-me_79.jpg` | meningioma | Training | train | - |

Cross-label links: `Testing/glioma/Te-gl_75.jpg` ~ `Testing/meningioma/Te-me_283.jpg` (near_hash, same, corr 0.7306, block p10 -0.049, documented different scans)

### g00388 (4 images, all excluded)

| Path | Label | Original folder | Split | Exclusion |
|---|---|---|---|---|
| `Testing/meningioma/Te-aug-me_35.jpg` | meningioma | Testing | excluded | pre_augmented |
| `Training/meningioma/Tr-me_409.jpg` | meningioma | Training | excluded | label_conflict_copy |
| `Training/pituitary/Tr-pi_1309.jpg` | pituitary | Training | excluded | label_conflict_copy |
| `Training/pituitary/Tr-pi_261.jpg` | pituitary | Training | excluded | label_conflict_copy |

Cross-label links: `Testing/meningioma/Te-aug-me_35.jpg` ~ `Training/pituitary/Tr-pi_1309.jpg` (near_hash, same, corr 0.8897, block p10 0.2981); `Testing/meningioma/Te-aug-me_35.jpg` ~ `Training/pituitary/Tr-pi_261.jpg` (near_hash, same, corr 0.8897, block p10 0.2981); `Training/meningioma/Tr-me_409.jpg` ~ `Training/pituitary/Tr-pi_1309.jpg` (pixel_mirror, mirror, corr 0.9998, block p10 0.998); `Training/meningioma/Tr-me_409.jpg` ~ `Training/pituitary/Tr-pi_261.jpg` (pixel_mirror, mirror, corr 0.9998, block p10 0.998)

### g00476 (2 images, all excluded)

| Path | Label | Original folder | Split | Exclusion |
|---|---|---|---|---|
| `Testing/meningioma/Te-me_115.jpg` | meningioma | Testing | excluded | label_conflict_copy |
| `Testing/pituitary/Te-pi_206.jpg` | pituitary | Testing | excluded | label_conflict_copy |

Cross-label links: `Testing/meningioma/Te-me_115.jpg` ~ `Testing/pituitary/Te-pi_206.jpg` (pixel_mirror, mirror, corr 0.9998, block p10 0.993)

### g02341 (5 images, all excluded)

| Path | Label | Original folder | Split | Exclusion |
|---|---|---|---|---|
| `Training/meningioma/Tr-me_105.jpg` | meningioma | Training | excluded | label_conflict_copy |
| `Training/pituitary/Tr-pi_1023.jpg` | pituitary | Training | excluded | label_conflict_copy |
| `Training/pituitary/Tr-pi_1071.jpg` | pituitary | Training | excluded | label_conflict_copy |
| `Training/pituitary/Tr-pi_1207.jpg` | pituitary | Training | excluded | label_conflict_copy |
| `Training/pituitary/Tr-pi_991.jpg` | pituitary | Training | excluded | label_conflict_copy |

Cross-label links: `Training/meningioma/Tr-me_105.jpg` ~ `Training/pituitary/Tr-pi_1023.jpg` (pixel_mirror, mirror, corr 0.9998, block p10 0.9968); `Training/meningioma/Tr-me_105.jpg` ~ `Training/pituitary/Tr-pi_1071.jpg` (pixel_mirror, mirror, corr 0.9998, block p10 0.9968); `Training/meningioma/Tr-me_105.jpg` ~ `Training/pituitary/Tr-pi_1207.jpg` (pixel_mirror, mirror, corr 0.9998, block p10 0.9968); `Training/meningioma/Tr-me_105.jpg` ~ `Training/pituitary/Tr-pi_991.jpg` (pixel_mirror, mirror, corr 0.9998, block p10 0.9968)

### g02424 (3 images, all excluded)

| Path | Label | Original folder | Split | Exclusion |
|---|---|---|---|---|
| `Training/meningioma/Tr-me_116.jpg` | meningioma | Training | excluded | label_conflict_copy |
| `Training/pituitary/Tr-pi_1133.jpg` | pituitary | Training | excluded | label_conflict_copy |
| `Training/pituitary/Tr-pi_1320.jpg` | pituitary | Training | excluded | label_conflict_copy |

Cross-label links: `Training/meningioma/Tr-me_116.jpg` ~ `Training/pituitary/Tr-pi_1133.jpg` (pixel_mirror, mirror, corr 0.9997, block p10 0.9948); `Training/meningioma/Tr-me_116.jpg` ~ `Training/pituitary/Tr-pi_1320.jpg` (pixel_mirror, mirror, corr 0.9997, block p10 0.9948)

### g02694 (2 images, all excluded)

| Path | Label | Original folder | Split | Exclusion |
|---|---|---|---|---|
| `Training/meningioma/Tr-me_533.jpg` | meningioma | Training | excluded | label_conflict_copy |
| `Training/pituitary/Tr-pi_91.jpg` | pituitary | Training | excluded | label_conflict_copy |

Cross-label links: `Training/meningioma/Tr-me_533.jpg` ~ `Training/pituitary/Tr-pi_91.jpg` (pixel_mirror, mirror, corr 0.9998, block p10 0.9978)

### g02777 (4 images, all excluded)

| Path | Label | Original folder | Split | Exclusion |
|---|---|---|---|---|
| `Training/meningioma/Tr-me_964.jpg` | meningioma | Training | excluded | label_conflict_copy |
| `Training/pituitary/Tr-pi_1098.jpg` | pituitary | Training | excluded | label_conflict_copy |
| `Training/pituitary/Tr-pi_1155.jpg` | pituitary | Training | excluded | label_conflict_copy |
| `Training/pituitary/Tr-pi_1380.jpg` | pituitary | Training | excluded | label_conflict_copy |

Cross-label links: `Training/meningioma/Tr-me_964.jpg` ~ `Training/pituitary/Tr-pi_1098.jpg` (pixel_mirror, mirror, corr 0.9998, block p10 0.9977); `Training/meningioma/Tr-me_964.jpg` ~ `Training/pituitary/Tr-pi_1155.jpg` (pixel_mirror, mirror, corr 0.9998, block p10 0.9977); `Training/meningioma/Tr-me_964.jpg` ~ `Training/pituitary/Tr-pi_1380.jpg` (pixel_mirror, mirror, corr 0.9998, block p10 0.9977)

## Verification

| Check | Result | Detail |
|---|---|---|
| Every original image accounted for exactly once | PASS | 7200 rows, 7200 files on disk; repeated 0, missing on disk 0, not in manifest 0 |
| Excluded images are exactly the listed ones and in no train/val/test split | PASS | excluded 232: {'label_conflict_copy': 29, 'pre_augmented': 203}; inconsistent rows 0 |
| Label-conflict exclusions cover complete duplicate groups | PASS | 8 groups; with eligible images left: [] |
| No undocumented cross-label duplicate link among eligible images | PASS | 1 cross-label links, 1 documented as different scans; undocumented 0 |
| No duplicate group spans more than one of train/val/test | PASS | 3405 groups with eligible images; crossing 0 |
| No duplicate link (hash or mirror-aware pixel copy) connects two different splits | PASS | 10868 links checked {'pixel_mirror': 5630, 'near_hash': 4406, 'exact_md5': 227, 'pixel_same': 605}; crossing 0 |
| No oversized duplicate group (chaining guard) | PASS | largest group 46 images; limit 2% of 7200 = 144 |
| All classes present in train, val and test | PASS | missing: none |
| Per-class split ratios within 2% of target | PASS | off target: none |
| Class balance in each split (smallest/largest class >= 0.75) | PASS | train 0.88, val 0.88, test 0.88 |

### Warnings

- 205 test image(s) have an UNCONFIRMED pixel candidate in train (64px corr >= 0.93, failed 256px confirmation); by 256px corr band {'0.95-1.00': 1, '0.90-0.95': 113, '0.85-0.90': 91, '0.00-0.85': 0}. Mostly reprocessed no-tumor copies or adjacent slices of the same patient; see review_pairs.csv.
- 187 val image(s) have an UNCONFIRMED pixel candidate in train (64px corr >= 0.93, failed 256px confirmation); by 256px corr band {'0.95-1.00': 1, '0.90-0.95': 89, '0.85-0.90': 97, '0.00-0.85': 0}. Mostly reprocessed no-tumor copies or adjacent slices of the same patient; see review_pairs.csv.
- 34 test image(s) have an UNCONFIRMED pixel candidate in val (64px corr >= 0.93, failed 256px confirmation); by 256px corr band {'0.95-1.00': 0, '0.90-0.95': 19, '0.85-0.90': 15, '0.00-0.85': 0}. Mostly reprocessed no-tumor copies or adjacent slices of the same patient; see review_pairs.csv.

## Remaining possible leakage

Held-out images with an unconfirmed pixel candidate (64px corr >= 0.93, either orientation) in another split, by best 256px correlation:

| Direction | 0.95-1.00 | 0.90-0.95 | 0.85-0.90 | 0.00-0.85 |
|---|---|---|---|---|
| test->train | 1 | 113 | 91 | 0 |
| val->train | 1 | 89 | 97 | 0 |
| test->val | 0 | 19 | 15 | 0 |

Hash-distance sensitivity (held-out images with a neighbour in the other split within distance d; must be 0 for d <= threshold):

| d | test->train | val->train | test->val |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 |
| 5 | 23 | 18 | 2 |
| 6 | 160 | 122 | 32 |
| 7 | 208 | 170 | 36 |
| 8 | 408 | 343 | 83 |
| 9 | 450 | 394 | 102 |
| 10 | 599 | 589 | 186 |

## Limitations

- No patient IDs are available, so this is NOT a patient-level split. Adjacent slices of the same patient (visible as unconfirmed candidates with low block correlation) can sit in different splits, so test scores may still be somewhat optimistic.
- A link means the images are near-identical after optional horizontal mirroring and brightness/contrast normalization, not proof of the same scan. Thresholds were calibrated by visual review of sampled pairs, not every pair.
- Heavily reprocessed copies (sharpening, blurring, strong gamma), mostly in the no-tumor class, can fail pixel confirmation and remain in different splits. They are counted in 'Remaining possible leakage' and listed in review_pairs.csv.
- Rotated, cropped or zoomed copies are not detected; only horizontal mirroring is handled. The '-aug-' exclusion removes the known pre-augmented files.
- Exclusions are rule-based ('-aug-' filenames) or explicit lists of label-conflict copies verified by pixel correlation. Label errors that are not duplicates cannot be detected.
- Hash and thumbnail values depend on Pillow resampling and floating-point arithmetic; versions are recorded in audit_report.json and --verify-only recomputes all links.

## Approval record and known limitations (added 2026-11-04)

- **Approved split:** this manifest (SHA-256 `4d09b01d782fa076890f32bf30bcbbe1a2319c135ead2e58cbd5e1f9ec0d930f`) was approved by the project owner on 2026-11-04 as the final train/val/test split. It replaces the previous split (manifest SHA-256 `f70c46dec1f0671f9459bb577c92b291b58add4edb0c8aeeb3d74714bd6d4df6`), backed up unchanged in `backups/data_splits_v1_f70c46de/`.
- **Approved extra exclusions:** 19 images in six meningioma/pituitary mirror-copy groups (256px corr 0.9997-0.9998, visually confirmed). The same scan appears under conflicting class folders and the correct label cannot be determined, so every copy was excluded instead of guessing a label. The list is in `approved_extra_exclusions.csv`.
- **Reproducibility:** an independent rebuild with the same settings and exclusions produced a byte-identical manifest.
- **`--link-related` not used:** visual review showed the blanket option would merge different patients across classes.
- **Remaining unconfirmed near-matches (not merged, by decision):**
  - 205 test images have near-matches in train.
  - 34 test images have near-matches in validation.
  - 187 validation images have near-matches in train.

  These are *unconfirmed* near-matches (64px thumbnail corr >= 0.93 but failing 256px / block confirmation). They are mainly neighbouring MRI slices of the same patient (tumor classes) or reprocessed copies of the same scan (no-tumor class). They are listed in `review_pairs.csv`.
- **No patient-level separation guarantee:** the dataset has no patient IDs, so patient-independent separation of train, validation and test cannot be guaranteed. The audit removes exact, near-hash and mirror-aware pixel duplicates across splits. Held-out metrics may still be somewhat optimistic because related slices from the same patient can appear in different splits.
