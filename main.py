import numpy as np
import func as func

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
print("New brain_data array, with Nan replaced with overall mean and zeros replaced with ones: ")
print(brain_data)

# Part 3
print("Patient 1: ", patient_names[0]) 
print("Patient 2: ", patient_names[1])
print("Last two brain regions: ", brain_regions[-2], brain_regions[-1])
print("Every other patient: ", patient_names[::2])
print("Reversed patient order: ", patient_names[::-1])

print("Temporal lobe data: ", brain_data[:,1], "Parietal lobe data: ", brain_data[:,2])
max_activity_per_row = np.nanmax(brain_data, axis = 1)
for i in range(len(max_activity_per_row)):
    if max_activity_per_row[i] > 30:
        print(patient_names[i])

# Part 4
flattened_brain_data = brain_data.flatten()
print(flattened_brain_data)

print(flattened_brain_data.reshape(5,5))
print(flattened_brain_data.reshape(1,25))
print(flattened_brain_data.reshape(25,1))

print(brain_data.transpose(1,0))

print(np.identity(5))
print(np.zeros(5))
print(np.ones(5))

# Part 5
print("Means activity:")
patient_activity_means = np.array([])
brain_region_activity_means = np.array([])
for i in range(0, rows_of_brain_data, 1):
    print(patient_names[i], ": ", np.mean(brain_data[i]))
    patient_activity_means = np.append(patient_activity_means, np.mean(brain_data[i]))
for i in range(0, rows_of_brain_data, 1):
    print(brain_regions[i], ": ", np.mean(brain_data[i], axis = 0))
    brain_region_activity_means = np.append(brain_region_activity_means, np.mean(brain_data[i]))
print("SD: ", np.std(brain_data))
print("Overall mean: ", np.mean(brain_data))

highest_mean_row_brain_data = np.argmax(patient_activity_means)
least_mean_column_brain_data = np.argmin(patient_activity_means, axis = 0)
print("Highest mean activity: ", patient_names[highest_mean_row_brain_data])
print("Lowest mean region: ", brain_regions[least_mean_column_brain_data])

# Part 6

sorted_algorithm = np.argsort(patient_activity_means)
patient_activity_means = (patient_activity_means[sorted_algorithm])[::-1]
patient_names_updated = (patient_names[sorted_algorithm])[::-1]
print("Ranked list of highest to lowest means of activity:")
for i in range(0, len(patient_activity_means), 1):
      print(patient_names_updated[i], ": ", patient_activity_means[i])

# Part 7

for i in range(0, len(max_activity_per_row), 1):
      if (np.min(brain_data[i], axis = 0) < np.mean(brain_data[i], axis = 0)):
        print("Patient ", patient_names[i], " has low ", brain_regions[i].lower(), " activity.")
      if (np.max(brain_data[i] > np.mean(brain_data[i], axis = 0))):
        print("Patient ", patient_names[i], " has high ", brain_regions[i].lower(), " activity.")

# Part 8

func.detect_abnormal_regions(brain_data, 30)