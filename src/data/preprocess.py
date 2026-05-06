import pandas as pd

def preprocess_data(df: pd.DataFrame, target_col: str = "churn") -> pd.DataFrame:
    """
    Basic cleaning for Telco churn
    - trim column names
    - drop customerID
    - fix Totalcharges to numeric
    - map target churn to 0/1 if needed
    - simple NA handling
    """

    # tidy headers
    df.columns = df.columns.str.strip() # remove leading/trailing whitespace

    # drop customerID
    for col in ["customerID", "Customer_ID", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=col)

    # target to 0/1 if it's yes/no
    if target_col in df.columns and df[target_col].dtype=="object":
        df[target_col] = df[target_col].str.strip().map({"No":0 , "Yes":1})

    # TotalCharges to numeric, coerce errors to NaN
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # SeniorCitizen to numeric if it's 0/1 into if present
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    # simple NA strategy:
    # - numeric: fill with 0
    # - others: leave for encoders to handle (get_dummies ignores NaN safely)

    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df    