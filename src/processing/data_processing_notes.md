# Data Processing Notes: G99 Slicing Rationale

This document explains why specific page ranges were selected from the **ENA Engineering Recommendation G99 Issue 2 (2025)** for the Automated Grid Compliance LLM Pipeline.

## 1. Pages 26–44 (Definitions & Categorisation)

- **Rationale**: Defines "Vehicle to Grid Electric Vehicle," "Electricity Storage," and the thresholds for "Type A" modules (<50kW). Establishes fundamental classification rules to distinguish between standard load and V2G export sources.
- **Key Content**: Figure 4.6 (Storage connection), Figures 6.10–6.12 (V2G topologies).

## 2. Pages 61–65 (Small Generation Installation Procedures)

- **Rationale**: Core "Connect & Notify" logic for small-scale EV/Storage. Details SGI-1, SGI-2, and SGI-3 procedures.
- **Key Content**: SGI Summary Table (Page 62), Procedure SGI-2/3 (G100 Export Limitation).

## 3. Pages 75–80 (Phase Balance & Shared Systems)

- **Rationale**: Hard engineering rules for phase imbalance (16A/phase limit) and general 32A limit. Essential for assessing clusters of single-phase chargers.
- **Key Content**: Section 7.5.3 (Unbalance limit), Section 7.8 (Sharing systems).

## 4. Pages 89–98 (Earthing for LV Connections)

- **Rationale**: Mandatory safety grounding for LV connections involving storage/EVs. Critical safety domain.
- **Key Content**: Figures 8.12–8.14 (Domestic/LV earthing with PME/TT).

## 5. Pages 125–126 (Protection Settings)

- **Rationale**: Exact voltage and frequency trip settings. These are high-precision requirements that must not hallucinate.
- **Key Content**: Table 10.1 (Trip settings).

## 6. Pages 137–144 (Type A Technical Requirements)

- **Rationale**: Frequency response and power output rules for Type A modules. Specific logic for when storage must stop import/export.
- **Key Content**: Figure 11.2 (Active Power response), Section 11.2.3.3 (Storage import logic).

## 7. Page 199 (V2G Compliance)

- **Rationale**: Defines legal responsibility for V2G compliance (owner vs manufacturer).
- **Key Content**: Section 15.5.

## 8. Pages 312–317 (Storage Exceptions)

- **Rationale**: Lists technical exclusions for EVs/Storage compared to standard generators to prevent incorrect model reasoning.
- **Key Content**: Annex A.4.2.

## 9. Pages 318–319 (Phase Imbalance Calculations)

- **Rationale**: Provides the specific mathematical examples for imbalance calculations.
- **Key Content**: Annex A.5.

---

# Data Processing Notes: UKPN EDS 08-5050 Rationale

## 1. Pages 8–9 (Process Logic & Supply Matrix)

- **Rationale**: Core logic for "Connect & Notify" decision tree. Essential for the model to determine if immediate installation is safe or if DNO permission is required based on kVA limits.
- **Key Content**: Figure 4-1 (Installer Flowchart), Table 5-1 (EVCP Supply Overview).

## 2. Page 11 (Unmetered Supply Technical Limits)

- **Rationale**: Hard engineering thresholds for Unmetered Supplies (UMS) (e.g., street lamps). Precise impedance limits for CNE vs SNE cables.
- **Key Content**: Table 6-1 (Maximum Loop Impedance).

## 3. Pages 11–13 (Earthing & Separation Rules)

- **Rationale**: Critical safety section. Defines where PME is forbidden, mandatory TT islands, and the **2.5m rule** for separation.
- **Key Content**: Table 7-1 (Permissible Earthing), Section 7.3 (Mandatory 2.5m rule).

## 4. Page 14 (Mandatory Notification)

- **Rationale**: Administrative routing for the model to know _where_ to send data (Smart Connect vs ENA).
- **Key Content**: Notification routing by installer type.

## 5. Page 17 (Approved Open PEN Devices)

- **Rationale**: List of assessed/approved hardware to avoid earth rods.
- **Key Content**: Table B-1 (List of assessed Open PEN devices).

---

# Important Constraints for UKPN Document:

1. **IGNORE V2G/V2L**: This document explicitly excludes V2G/V2L. The model must refer back to G99 for these.
2. **2.5m Separation**: Central safety rule for TT islands.
3. **No Diversity Guidance**: Version 6.0 removed this; model should use Table 5-1 (MPR Available) instead.

---

# Data Processing Notes: SPEN EV Fleet Guide Rationale

## 1. Pages 5–6 (Capacity Assessment & The "30% Rule")

- **Rationale**: Teaches the model a specific commercial heuristic: checking if EV demand exceeds 30% of total site demand. Essential for fleet planning logic.
- **Key Content**: The "30% Rule" for notification, Authorised Capacity vs Maximum Demand comparison.

## 2. Page 10 (Operational Mitigation Strategies)

- **Rationale**: Provides "soft" engineering alternatives (Timed Profile Connections, Storage) to suggest when grid reinforcement is too costly.
- **Key Content**: Timed Profiles (varying capacity by time), On-site storage leveling.

## 3. Pages 11–13 (Cost & Timeline Case Studies)

- **Rationale**: **Gold Standard** for cost/time estimation. Includes benchmark figures for Small (£70k-100k), Flexible (£200k-500k), and Large (£3m-5m) projects.
- **Key Content**: Case studies with specific Scope, Cost, and Lead Time.

## 4. Pages 14–15 (DNO-Specific Definitions)

- **Rationale**: Classifies charger speeds (Rapid >50kW, Ultra-Rapid 100kW-350kW) specifically for this network operator.
- **Key Content**: Categorisation of charger power ratings.

---

# Important Constraints for SPEN Document:

1. **Consultant Persona**: Use this to train project estimation, timelines, and commercial flows.
2. **Ignore Engineering Specs**: For earthing, protection, or G99 physics, strictly prioritize EDS 08-5050 and G99.
3. **Timed Profile connections**: Highlight this as a primary suggestion for constrained grids.
