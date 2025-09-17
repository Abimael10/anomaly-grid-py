
# Challenge Report

## Challenge Success Assessment

- **Maximum F1 achieved**: 0.7102
- **Average F1 across all models**: 0.5400

## Challenge Datasets

### 🔐 Protocol State Machine (16 States)
- **Samples**: 15000 protocol sequences
- **Contamination**: 2.5%
- **Challenge**: Complex multi-order state transition patterns with subtle violations
- **Avg Length**: 124.9 states
- **Max Length**: 200 states

### 🧬 Protein Folding (20 Amino Acids)
- **Samples**: 15000 protein sequences
- **Contamination**: 2.0%
- **Challenge**: Biological sequence patterns with rare misfolding events
- **Avg Length**: 190.3 amino acids
- **Max Length**: 300 amino acids

### 📡 Communication Protocol (12 Symbols)
- **Samples**: 15000 communication sequences
- **Contamination**: 2.0%
- **Challenge**: Steganographic timing attacks in digital communication
- **Avg Length**: 75.3 symbols
- **Max Length**: 122 symbols

## Design Principles

1. **🔤 SMALL VOCABULARIES**: 12-20 symbols maximum (finite, well-defined)
2. **🔗 COMPLEX DEPENDENCIES**: Multi-order sequence patterns (orders 1-4)
3. **⏰ TEMPORAL PATTERNS**: State transitions, biological constraints, timing protocols

## Challenge Results by Dataset

### Winners by Alphabet Size

**Protocol State Machine** (Alphabet: 16 symbols)
- Winner: anomaly-grid-py (order=4) (Sequence-Based)
- Macro F1: 0.7102 ± 0.011
- ROC AUC: 0.9029 ± 0.006

**Communication Protocol** (Alphabet: 12 symbols)
- Winner: anomaly-grid-py (order=2) (Sequence-Based)
- Macro F1: 0.5159 ± 0.020
- ROC AUC: 0.5756 ± 0.030

### Sequence vs Traditional ML Battle Results

| Dataset | Alphabet Size | Winner | Traditional F1 | Sequence F1 | Improvement |
|---------|---------------|--------|----------------|-------------|-------------|
| Protocol State Machine | 16 | 🚀 Sequence | 0.6687 | 0.7102 | +6.20% |
| Communication Protocol | 12 | 🚀 Sequence | 0.5011 | 0.5159 | +2.95% |

### Statistical Analysis

- **Total experiments**: 28 (with 3-fold CV)
- **Performance ranges**:
  - Traditional ML: 0.4111 - 0.6687
  - Sequence-based: 0.4533 - 0.7102

### Alphabet Size Impact

- **Alphabet 16**: Traditional=0.579, Sequence=0.641, Advantage=+0.062
- **Alphabet 12**: Traditional=0.480, Sequence=0.491, Advantage=+0.011

## Key Findings

1. **⏰ TEMPORAL PATTERNS MATTER**: State transitions and timing require sequence understanding

---
*Challenge completed on 2025-09-16 19:48:56*
