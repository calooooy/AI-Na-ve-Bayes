import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv('coupon.csv') # specify the path where the dataset file is located

# Remove the 'car' column
df = df.drop(columns=['car'])

# Handle missing values
# For categorical columns, fill missing values with the mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    mode_value = df[col].mode()[0]
    df[col] = df[col].fillna(mode_value)

# For numerical columns, fill missing values with the mean
numerical_cols = df.select_dtypes(include=['number']).columns
for col in numerical_cols:
    mean_value = df[col].mean()
    df[col] = df[col].fillna(mean_value)

# Treat 'temperature' as a categorical column
df['temperature'] = df['temperature'].astype(str)

# Update categorical columns to include 'temperature'
categorical_cols = df.select_dtypes(include=['object']).columns

# Applying OneHotEncoder to categorical columns
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_array = encoder.fit_transform(df[categorical_cols])

# Create a DataFrame from the encoded array
df_encoded = pd.DataFrame(encoded_array, index=df.index, columns=encoder.get_feature_names_out(categorical_cols))

# Concatenate encoded categorical and original numerical columns
df_final = pd.concat([df.drop(categorical_cols, axis=1), df_encoded], axis=1)

# Save the preprocessed dataset to a new CSV file in the specified directory
output_path = 'cleaned_dataset.csv' # specify the path where you want to save the clean dataset
df_final.to_csv(output_path, index=False)

print(f"Dataset preprocessed and saved as '{output_path}'.")