import numpy as np

brain_data = np.array([
    [12, 45, 67, 23, 89],
    [34, 0, 78, 56, 90],
    [np.nan, 54, 32, 11, 76],
    [88, 91, 45, 66, 39],
    [21, 73, 84, 0, 95]
])

patient_names = np.array([
    "Alice",
    "Bob",
    "Charlie",
    "David",
    "Eva"
])

brain_regions = np.array([
    "Frontal",
    "Temporal",
    "Parietal",
    "Occipital",
    "Cerebellum"
])

# Part 1
rows_of_brain_data, cols_of_brain_data = np.shape(brain_data)
print(rows_of_brain_data, " rows, ", cols_of_brain_data, " columns.")
print(np.size(brain_data), " elements.")

print("Lowest intensity value: ", np.nanmin(brain_data), ", located at index ", np.nanargmin(brain_data))
print("Highest intensity value: ", np.nanmax(brain_data), ", located at index ", np.nanargmax(brain_data))

print("Highest intensities in each brain region: ", (((str((np.nanmax(brain_data, axis = 0))).replace("[", "")).replace("]", "")).replace(".", ".0")))
print("Lowest intensities in each patient: ", (((str((np.nanmin(brain_data, axis = 1))).replace("[", "")).replace("]", "")).replace(".", ".0")))

# Part 2
brain_data[brain_data == 0] = 1
mean_value = np.nanmean(brain_data)
brain_data[np.isnan(brain_data)] = mean_value
print(brain_data)

