Complete Feature List

## 1. 61-Feature Baseline

### 1.1 Elemental Composition (45 features)

**Elements**: Si, Al, P, Na, K, Li, Sr, Rb, Cs, Ba, Ca, F, Ge, Ti, In, B, Mg, Ga, Ni, Mn, Fe, Co, Cr, Zn, Nb, Be, W, Ce, Cu, Sn, Gd, La, Y, Dy, Sm, Ag, Cd, Zr, V, Ta, Ru, Hf, Yb, Tl, As

**Description**: Normalized molar fractions of 45 elements commonly used in zeolite synthesis

### 1.2 OSDA Indices (3 features)

- osda1_index: Integer index for OSDA 1
- osda2_index: Integer index for OSDA 2  
- osda3_index: Integer index for OSDA 3

### 1.3 Synthesis Conditions (4 features)

- cryst_temp: Crystallization temperature (°C)
- cryst_time: Crystallization time (hours)
- seed: Seed crystal presence (0/1)
- rotation: Stirring or rotation during synthesis (0/1)

### 1.4 Aging Conditions (2 features)

- aging_temp: Aging temperature (°C)
- aging_time: Aging time (hours)

### 1.5 pH Conditions (2 features)

- acid: Acid concentration
- OH: Hydroxide concentration

### 1.6 Gel Ratios (5 features)

- H2O_T: Water to T-atom ratio
- OH_T: Hydroxide to T-atom ratio
- Gel_Si_Al: Si/Al ratio in gel
- Gel_P_Al: P/Al ratio in gel
- Gel_P_Si: P/Si ratio in gel

---

## 2. 104-Feature Improvement (Additional 43 features)

### 2.1 OSDA Molecular Descriptors (33 features)

**For each OSDA (osda1, osda2, osda3)**:

1. bertz_ct_mean_0: Bertz Complexity Index
2. free_sasa_mean_0: Solvent Accessible Surface Area
3. asphericity_mean_0: Asphericity
4. eccentricity_mean_0: Eccentricity
5. axes_mean_0: First principal axis length
6. axes_mean_1: Second principal axis length
7. box_mean_0: X-direction bounding box dimension
8. box_mean_1: Y-direction bounding box dimension
9. box_mean_2: Z-direction bounding box dimension
10. getaway_mean_0: First GETAWAY descriptor
11. getaway_mean_1: Second GETAWAY descriptor

### 2.2 Aggregated Features (10 features)

1. osda_avg_asphericity: Average asphericity across all OSDAs
2. osda_max_asphericity: Maximum asphericity among all OSDAs
3. osda_min_asphericity: Minimum asphericity among all OSDAs
4. osda_avg_sasa: Average solvent accessible surface area
5. osda_max_sasa: Maximum solvent accessible surface area
6. osda_min_sasa: Minimum solvent accessible surface area
7. osda_avg_bertz: Average Bertz complexity index
8. osda_max_bertz: Maximum Bertz complexity index
9. osda_min_bertz: Minimum Bertz complexity index
10. osda_total_volume: Total volume of all OSDAs combined

---

## 3. CBU Encoding for Transfer Learning (Additional 43 features)

**CBU Types**: afy, abw, afs, aft, atn, ats, bph, d3r, ddr, doh, dnc, imf, los, ltl, mei, nat, pau, rut, sod, etc. (64 total)

**Description**: One-hot encoding of 65 CBU types (including "none" or "empty") representing structural building units

---

## 4. Feature Summary

**Total Features by Category**:

| Category | 61-Feature Baseline | 104-Feature | 
|----------|---------------------|--------------|--------------------------------------|
| Elemental Composition | 45 | 45 | 
| OSDA Indices | 3 | 3 | 
| Synthesis Conditions | 4 | 4 | 
| Aging Conditions | 2 | 2 | 
| pH Conditions | 2 | 2 | 
| Gel Ratios | 5 | 5 | 
| OSDA Molecular Descriptors | 0 | 33 | 
| Aggregated Features | 0 | 10 | 

| **Total** | **61** | **104** |

**Note**: The CBU transfer learning 104-feature concise version uses only 104 pure synthetic condition features (no CBU encoding), with CBUs used solely for data grouping.