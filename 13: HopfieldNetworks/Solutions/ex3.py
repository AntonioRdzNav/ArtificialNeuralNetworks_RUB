import numpy as np
import matplotlib.pyplot as plt
from general import calculate_weights, retrieve_sync, retrieve_async


# CALCULATE NETWORK CAPACITY
network_size = 100
load_factor = 0.5
error_rates = []
nums_patterns = np.linspace(1, network_size*load_factor, 40).astype(int)

for num_patterns in nums_patterns:
    print(f"Storing {num_patterns} patterns...")

    # generate n random patterns
    patterns = np.random.choice((-1, 1), (network_size, num_patterns))

    # cc_im_mat = np.corrcoef(patterns.T)
    # fig, ax = plt.subplots()
    # ax.hist(cc_im_mat.flatten(), bins=100)
    # im_h = ax.imshow(cc_im_mat)
    # fig.colorbar(im_h)
    # plt.show()

    # calculate weight matrix
    w = calculate_weights(patterns)

    # retrieve patterns and calculate errors
    error_count = 0
    for pattern_num, pattern in enumerate(patterns.T):
        if pattern_num % network_size == 0:
            print(f"pattern number: {pattern_num}")

        retrieved = retrieve_sync(pattern.copy(), w, num_iter=100)

        error_count += np.count_nonzero(retrieved != pattern)
    error_rate = error_count / network_size / num_patterns
    print(f"error rate: {error_rate}")
    error_rates.append(error_rate)

fig, ax = plt.subplots()
ax.plot(nums_patterns/network_size, error_rates)
ax.set_xlabel("Network load")
ax.set_ylabel("Error rate")
fig.savefig(f"figures/error_rates_{load_factor}.pdf")
plt.show()
