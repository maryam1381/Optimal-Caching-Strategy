import env_pool


# result = env_pool._zipf_probs(10, 1.5)
# print(result)


# import numpy as np

# def test_dirichlet_posterior_samples():
#     # Parameters for the test
#     n = np.array([3, 5, 2])  # observed counts for 3 categories
#     N_pool = 10              # number of posterior samples to draw
#     alpha0 = 1.0             # symmetric Dirichlet prior concentration

#     # Generate posterior samples
#     posterior_samples = env_pool.dirichlet_posterior_samples_from_counts(n, N_pool, alpha0)

#     # Test assertions
#     assert posterior_samples.shape == (N_pool, len(n)), "Shape of output is incorrect"
#     assert np.allclose(posterior_samples.sum(axis=1), 1.0), "Rows of the output don't sum to 1"
    
#     # Print a few samples for inspection
#     print(posterior_samples[:3])  # Print the first 3 samples
    


# import torch
# from losses import project_capped_simplex
# y = torch.tensor([0.5, 0.7, 1.2, 0.3, 0.6]) 
# S = 2.5  
# result = project_capped_simplex(y, S)
# print("Projected tensor:", result) # all of them should be between 0,1
# print("Sum of projected tensor:", result.sum().item()) # should be equal to S
