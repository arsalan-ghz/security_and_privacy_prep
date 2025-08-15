# Week 7-Bonus: Interpreting Finite-Difference Slopes

## Introduction

This flashcard is designed to refresh your understanding of the finite-difference approach used at a witness point for a neuron. In this context, you'll recall how the raw slopes, $g_+^i$ and $g_-^i$, are measured when moving slightly in the positive and negative directions along each input axis, and how their difference, defined as

$$
\Delta_i \approx A^{(1)}_{j,i}\,\alpha_j
$$

reveals the neuron’s sensitivity to those directions. Review these concepts and equations to solidify your grasp of how the baseline responses are subtracted out, keeping in mind the nuances without revealing detailed answers.

---

## Questions

1. **Understanding the Difference:**
   
   What is the fundamental distinction between the raw slopes $g_+^i$ and $g_-^i$ and their computed difference $\Delta_i$ at a neuron’s witness point?

2. **Role of the Baseline Slope:**
   
   How does the baseline slope from other neurons factor into the finite-difference calculation, and how is it accounted for in the equation

   $$
   \Delta_i \approx A^{(1)}_{j,i}\,\alpha_j
   $$

   ?

3. **Interpreting Zero Values:**
   
   Why are neither $g_+^i$ nor $g_-^i$ expected to be near zero individually at the witness point, yet $\Delta_i$ can be zero for some input directions? Discuss the significance of $A^{(1)}_{j,i}$ in this context.

4. **Implications of a Null Difference:**
   
   In the case where, for a given input direction, $\Delta_i = 0$, what does this indicate about the neuron's weighting in that direction? Explain your reasoning using the mathematical relationship provided.

---

## Answers

1. **Understanding the Difference:**
   
   The raw slopes $g_+^i$ and $g_-^i$ capture the change in output when you nudge the input slightly in the positive and negative directions, respectively. They include not only the effect of neuron $j$ when it is active but also a baseline contribution from all the other neurons. By computing the difference, $$\Delta_i = g_+^i - g_-^i,$$ this common baseline cancels out. As a result, what remains is a value that is directly proportional to the weight $A^{(1)}_{j,i}$ of neuron $j$ (scaled by the downstream gain $\alpha_j$). This isolates the contribution of neuron $j$, revealing its sensitivity in that specific input direction.

2. **Role of the Baseline Slope:**
   
   When measuring the finite differences $g_+^i$ and $g_-^i$, the baseline slope – which represents the contribution from all other neurons – appears in both values. This shared component is why, even though the raw slopes are nonzero, their difference $$\Delta_i \approx A^{(1)}_{j,i}\,\alpha_j$$ effectively subtracts out the baseline. Thus, the equation tells you that the remaining difference is solely due to the neuron’s own weight in that direction, independent of the constant background effect from other neurons.

3. **Interpreting Zero Values:**
   
   Neither $g_+^i$ nor $g_-^i$ is expected to be near zero individually because both include the baseline contribution. However, if a particular direction does not influence neuron $j$ (i.e., if $A^{(1)}_{j,i}=0$), then the additional contribution from activating neuron $j$ vanishes, and the difference $$\Delta_i = g_+^i - g_-^i$$ becomes zero. This zero difference explicitly signals that neuron $j$ does not respond to perturbations in that input direction, even though the raw slopes remain nonzero due to the baseline.

4. **Implications of a Null Difference:**
   
   A zero difference ($\Delta_i=0$) indicates that the weight $A^{(1)}_{j,i}$ is zero. In other words, if changes in the input direction $i$ do not produce a change in the neuron’s contribution (reflected by $\Delta_i$), it means the neuron is insensitive to that direction. This is a clear sign that neuron $j$ does not utilize that particular input coordinate, as its influence is effectively nullified in that dimension.
