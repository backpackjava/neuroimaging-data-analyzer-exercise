import numpy as np
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
def detect_abnormal_regions(data, threshold):
    where_represents_name, where_represents_region = np.where(data > threshold)
    for i in range(0, len(where_represents_name), 1):
        print(str((patient_names[where_represents_name[i]])).upper(), " has extreme ", str(brain_regions[where_represents_region[i]]).lower(), " activity.")