import numpy as np
import matplotlib.pyplot as plt

noiseFreq = np.load("results/noise_results_outside.npy")
hestFreq = np.load("results/hest_results_outside.npy")

noiseAveraged = np.mean(noiseFreq, axis=0)
hestAveraged = np.mean(hestFreq, axis=0)

noisePower = np.mean(np.abs(noiseAveraged) ** 2)

print("Noise power: " + str(noisePower))

# plt.plot(hestFreq)
# plt.show()

powerSymbol = np.sqrt(2)

# Find lambda 

lmbdaCandidates = [0.1 * i for i in range(1000, 4000)]
bestlmbda = 0    
minDifference = 5

for lmbda in lmbdaCandidates:
    totalPStar = 0
    for hest in hestAveraged:
        pstar = max(1 / lmbda - noisePower / (np.abs(hest + 1e-9) ** 2), 0)
        totalPStar += pstar

    if np.abs(totalPStar - powerSymbol) < minDifference:
        difference = np.abs(totalPStar - powerSymbol)
        minDifference = difference
        # print(minDifference)
        bestlmbda = lmbda

# Calculate channel capacity
channelCapacity = 0
for hest in hestAveraged:
    pstar = max(1 / bestlmbda - noisePower / (np.abs(hest + 1e-9) ** 2), 0)
    channelMax = 1 + (np.abs(hest + 1e-9) ** 2) * (pstar ** 2) / noisePower
    channelCapacity += channelMax

channelCapacity = channelCapacity * (44100 / (2048 + 256)) / 2

print("Channel Capacity:" + str(channelCapacity))
