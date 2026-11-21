import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("cards_data.csv")

print("Original Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())


# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Remove sensitive and unnecessary columns
df.drop(columns=["card_number", "cvv", "id"], inplace=True)

# Remove constant column (all values are No)
df.drop(columns=["card_on_dark_web"], inplace=True)

print("\nColumns after removing unnecessary data:")
print(df.columns)


# Numerical columns
num_cols = df.select_dtypes(include=np.number).columns

# Categorical columns
cat_cols = df.select_dtypes(include="object").columns

# Fill numerical missing values with median
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values with mode
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Remove dollar sign
df["credit_limit"] = df["credit_limit"].replace(
    r"[$,]", "", regex=True
)

# Convert into numeric
df["credit_limit"] = pd.to_numeric(
    df["credit_limit"], errors="coerce"
)

# Convert dates
df["expires"] = pd.to_datetime(
    df["expires"], format="%m/%Y", errors="coerce"
)

df["acct_open_date"] = pd.to_datetime(
    df["acct_open_date"], format="%m/%Y", errors="coerce"
)

# Fill invalid credit limits with median
df["credit_limit"] = df["credit_limit"].fillna(
    df["credit_limit"].median()
)

print("\nUpdated Data Types:")
print(df.dtypes)

total_customers = df["client_id"].nunique()

print("Total Customers:", total_customers)

print("\nCard Type Analysis:")

print(df["card_type"].value_counts())

print("\nChip Card Analysis:")

print(df["has_chip"].value_counts())

average_limit = df["credit_limit"].mean()

print("Average Credit Limit:", average_limit)

print("Maximum Credit Limit:",
      df["credit_limit"].max())

print("Minimum Credit Limit:",
      df["credit_limit"].min())



# Convert credit limit into NumPy array
limits = df["credit_limit"].to_numpy()

# Mean
print("Mean:", np.mean(limits))

# Median
print("Median:", np.median(limits))

# Standard Deviation
print("Standard Deviation:", np.std(limits))

# Minimum
print("Minimum:", np.min(limits))

# Maximum
print("Maximum:", np.max(limits))

# 25th and 75th Percentile
print("25th Percentile:", np.percentile(limits, 25))

print("75th Percentile:", np.percentile(limits, 75))


# Save cleaned dataset
df.to_csv("cleaned_credit_card_data.csv", index=False)

print("\nFinal Dataset:")
print(df.head())

print("\nFinal Shape:")
print(df.shape)

print("\nProject Completed Successfully!")