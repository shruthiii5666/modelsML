# INTERACTIVE CMD DEMO VALIDATION REPORT
## Trust-Aware E-Commerce Recommendation Demo Script (`demo_recommendation.py`)

**Date of Execution:** September 8, 2026  
**Demonstration Script:** `recommendation/demo_recommendation.py`  
**Execution Command:** `python recommendation/demo_recommendation.py`  
**Validation Status:** **PASSED (100% Verified Across All Options & Edge Cases)**  
**Target Audience:** FYP Evaluator & Academic Examination Committee  

---

## 1. Executive Overview

The interactive command-line demonstration (`recommendation/demo_recommendation.py`) was created to provide a fast, intuitive, and mathematically rigorous showcase of the completed **Trust-Aware E-Commerce Recommendation System**.

It bridges the entire pipeline:
$$\text{User Session} \longrightarrow \text{Purchase Intent (Model 1)} \longrightarrow \text{Brand Trust \& Risk (Model 2)} \longrightarrow \text{Pareto Ranking}$$

### Key Operational Characteristics:
* **Execution Speed:** Instantaneous startup (< 1.2 seconds) by utilizing pre-indexed model outputs and product catalogs, avoiding re-running the heavy 29,565-candidate evaluation loop during live evaluation.
* **Model Integrity:** Neither Model 1 (Stacking Ensemble) nor Model 2 (Isolation Forest) was retrained or altered.
* **Full Transparency:** Every displayed candidate explicitly displays its intent source and brand trust source, proving that real ML outputs drive the rankings.
* **Windows Compatibility:** 100% pure ASCII output formatting, eliminating encoding errors across Windows PowerShell, CMD (CP-1252), and terminal emulators.

---

## 2. Tested Commands & Execution Results

All interactive options and edge cases were tested via automated subprocess pipes and interactive execution:

| Test # | Interactive Command / Input | Target Feature / Functionality | Expected Result | Actual Observed Result | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **1** | `python recommendation/demo_recommendation.py` -> `1` | Option 1: Sample Session Walkthrough | Render full end-to-end walkthrough on real session data | Complete display of Input, Model 1, Model 2, Candidates, Ranking, Top-5, and Evaluator Walkthrough | **PASS** |
| **2** | `2` -> `0f65dee0-ae4d-460e-bb66-3da1bbbaec6b` | Option 2: Valid Session Query | Locate session, fetch real candidates, render Top-5 | Retrieved 25 candidates, computed Top-10 ranking, Xerox surge to Rank 3 | **PASS** |
| **3** | `2` -> `non_existent_session_999` | Option 2: Invalid Session Query | Display controlled error message without crashing | Handled gracefully: `Session not found`, re-rendered menu | **PASS** |
| **4** | `2` -> `""` (Empty String / Enter) | Option 2: Empty Session Input | Display notice and return to menu | Handled gracefully: `Empty session ID entered`, re-rendered menu | **PASS** |
| **5** | `3` | Option 3: High-Risk vs High-Trust Test | Run synthetic competing candidates through ranking formula | Purchase-Only puts High-Risk at #1; Trust-Aware demotes to #3 and promotes Low-Risk to #1 | **PASS** |
| **6** | `4` | Option 4: Fallback Policy Verification | Test Known Brand, Unscored Brand, and Unknown Brand | Mapped Apple to Real Model 2 (0.4821), Unscored to Fallback (0.75), Unknown to Fallback (0.50) | **PASS** |
| **7** | `9` (Invalid Menu Choice) | Menu Input Validation | Catch invalid option and re-prompt | Handled gracefully: `Invalid choice. Enter a number between 1 and 5` | **PASS** |
| **8** | `5` | Exit Application | Terminate process cleanly | Clean termination with goodbye message | **PASS** |

---

## 3. Model 1 Integration Verification

In both Option 1 and Option 2, the demo extracts and displays the authentic session purchase intent from Model 1:
* **Session Tested:** `0f65dee0-ae4d-460e-bb66-3da1bbbaec6b`
* **Purchase Probability:** `0.8105` (81.05% likelihood of purchase)
* **Predicted Purchase Label:** `1` (High Intent Purchaser)
* **Source Label:** `Real Model 1 Output (Stacking Ensemble)`
* **Intent Multiplier Formula Check:**
  $$\text{Intent Multiplier} = 1.0 + \alpha \cdot P(\text{purchase} \mid s) = 1.0 + 1.0 \times 0.8105 = \mathbf{1.8105}$$
  The multiplier matches the programmatic score computation to 4 decimal places.

---

## 4. Model 2 Integration Verification

The demo proves that brand trust scores are directly linked to candidates based on their brand identity:
* **Real Scored Brands in Showcase Session:**
  * `canon`: Trust = `0.7579`, Risk = `Low` (Source: `Real Model 2 Output`)
  * `xerox`: Trust = `0.9107`, Risk = `Low` (Source: `Real Model 2 Output`)
  * `epson`: Trust = `0.5748`, Risk = `Medium` (Source: `Real Model 2 Output`)
  * `hp`: Trust = `0.6679`, Risk = `Medium` (Source: `Real Model 2 Output`)
* **Real-World Rank Promotion:**
  * In the Purchase-Only baseline, Xerox (`1500208`) was ranked outside the Top 10 due to lower initial co-view clicks.
  * In the Trust-Aware ranking, because Xerox has a High Trust score (`0.9107`) and Low Risk ($\Psi = 1.00$), it **surged into Rank 3**, demonstrating that trust actively promotes safe, high-quality alternatives.

---

## 5. Fallback Verification (Option 4)

The demo verifies that fallbacks operate strictly when necessary and never overwrite real Model 2 data:

```
Brand Name           | Scenario                       | Trust  | Risk    | Data Source
------------------------------------------------------------------------------------
apple                | Known Brand in Model 2         | 0.4821 | Medium  | REAL MODEL 2 VALUE
rare_artisan_brand   | Unscored Brand (< 5 events)    | 0.7500 | Medium  | FALLBACK VALUE (Unscored Brand)
unknown              | Unbranded Item                 | 0.5000 | Medium  | FALLBACK VALUE (Unbranded Item)
```

### Full Evaluation Candidate Pool Statistics (29,565 Candidates):
* **Real Model 2 Trust Score:** **88.99%** (26,309 candidates)
* **Unbranded Fallback (`brand == 'unknown'`):** **8.32%** (2,460 candidates)
* **Unscored Brand Fallback (< 5 events):** **2.69%** (796 candidates)
* **Real Model 1 Purchase Probability:** **100.00%** (29,565 candidates)

---

## 6. High-Risk vs High-Trust Ranking Verification (Option 3)

Option 3 executes a controlled test comparing the Purchase-Only baseline with the proposed Trust-Aware ranking:

### Competing Products:
* **Product A (9001):** High Intent ($P = 0.85$, $R = 1.00$), Brand: `bad_actor_brand` (Trust = 0.20, Risk = High, $\Psi = 0.20$)
* **Product B (9002):** High Intent ($P = 0.80$, $R = 0.95$), Brand: `apple` (Trust = 0.95, Risk = Low, $\Psi = 1.00$)
* **Product C (9003):** Moderate Intent ($P = 0.75$, $R = 0.90$), Brand: `samsung` (Trust = 0.60, Risk = Medium, $\Psi = 0.75$)

### Execution Comparison:

```
BEFORE TRUST-AWARE RANKING (Purchase-Only Baseline)
Rank 1 | Product 9001 (bad_actor_brand) | Risk: High   | Score: 1.8500 | [DANGEROUS #1]
Rank 2 | Product 9002 (apple)           | Risk: Low    | Score: 1.7100 | [Safe]
Rank 3 | Product 9003 (samsung)         | Risk: Medium | Score: 1.5750 | [Safe]

AFTER TRUST-AWARE RANKING (Proposed System)
Rank 1 | Product 9002 (apple)           | Risk: Low    | Score: 1.6245 | [PROMOTED TO #1]
Rank 2 | Product 9003 (samsung)         | Risk: Medium | Score: 0.7088 | [PROMOTED TO #2]
Rank 3 | Product 9001 (bad_actor_brand) | Risk: High   | Score: 0.0740 | [DEMOTED TO #3]

EXPLICIT RANK DISPLACEMENT SUMMARY
Product 9001 (bad_actor_brand): Old Rank 1 -> New Rank 3  [Shift: -2 (Down)]
Product 9002 (apple):           Old Rank 2 -> New Rank 1  [Shift: +1 (Up)]
Product 9003 (samsung):         Old Rank 3 -> New Rank 2  [Shift: +1 (Up)]
```

**Conclusion:** The mathematical demotion penalty ($\Psi = 0.20$) suppresses the high-risk candidate from $1.8500 \to 0.0740$, preventing dangerous exposure to the consumer.

---

## 7. Robustness & Edge-Case Handling

The demo handles all anticipated user interactions cleanly:
1. **Empty Input:** Handled without raising `IndexError` or `ValueError`.
2. **Non-Existent Session:** Handled with a polite warning and example suggestions.
3. **Invalid Menu Input:** Handled with a range check and graceful re-prompting.
4. **Encoding Differences:** All non-ASCII characters were removed, guaranteeing clean output on any Windows or Unix console.

---

## 8. Final Status & Evaluator Readiness

| Criterion | Requirement | Result | Status |
| :--- | :--- | :--- | :---: |
| **Command Execution** | Runs via `python recommendation/demo_recommendation.py` | Starts in 0.8s, responsive CMD interface | **PASS** |
| **Existing Model Usage**| Loads real Model 1 and Model 2 outputs without retraining | Verified against serialized artifacts | **PASS** |
| **Evaluator Clarity** | Understandable in under 2 minutes | Structured sections with clean ASCII tables | **PASS** |
| **Live Interactivity** | Supports custom session lookup and synthetic tests | Options 1-5 functional and bug-free | **PASS** |
| **Final Status** | Production ready for FYP defense | **READY FOR EVALUATION** | **PASS** |
