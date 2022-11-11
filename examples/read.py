# Example on how to use the Reader


import pandas as pd
from DeepRetail.preprocessing.input import Reader


####################################
# Example -> write on new file #####
# Suggested for big files ##########
####################################

# Define the filepath of the file
case_1_filepath = "..."

# Define the outpath
out_path = ".../case_1_week.csv"

# Call the reader
r = Reader(case=1, filepath=case_1_filepath)

# Save
r.save(out_path, frequency="W", format="pivoted")

# Then load it
# df = pd.read_csv(out_path)

####################################
# Example -> load from file #######
# Suggested for small files #######
####################################

# Define the filepath of the file
case_5_filepath = "..."

# Call the reader
r = Reader(case=4, filepath=case_1_filepath)

# Load
df = r.load(frequency="W", format="pivoted")
