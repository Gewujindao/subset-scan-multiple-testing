# Statements and proofs

## Prefix property (independent traits)

Write a_k=w_k z_k, b_k=w_k², r_k=a_k/b_k=z_k/w_k, a(S)=Σ_S a_k, b(S)=Σ_S b_k. Suppose S is a local maximum under single-element additions and removals, and a=a(S)>0. Adding j∉S does not increase the objective ⇔ a_j(2a+a_j) ≤ (a²/b) b_j; removing i∈S does not increase it ⇔ a_i(2a−a_i) ≥ (a²/b) b_i. For a_i,a_j>0 this gives r_j < a/(2b) < r_i; adding a trait with a_k≤0 can only lower the objective. Hence S={k: a_k>0, r_k>a/(2b)}, which is a prefix in decreasing order of r; the global maximum is among the K prefixes (when all are negative, take the largest single z). This holds verbatim for the statistic of the original paper (w_k=√n_k or 1/s_k).

## Block-diagonal correlation

When the blocks are mutually independent, a and b are additive over blocks; the same algebra shows that in the optimal solution the subset chosen in each block maximizes a_B−λb_B (λ=a/(2b)), so it suffices to sweep by slope between the vertices of the upper convex hull of each block's (b_B,a_B). Cost O(Σ_B 2^{|B|}(|B|²+convex hull)).
