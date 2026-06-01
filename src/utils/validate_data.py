import pandas as pd
import great_expectations as ge
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Validate Telco Customer Churn dataset using Great Expectations.
    Returns:
        (is_valid, failed_expectations)
    """

    print("Starting data validation with Great Expectations... - validate_data.py:13")

    # Create copy to avoid modifying original dataframe
    df = df.copy()

    # Fix data types
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = pd.to_numeric(
            df["MonthlyCharges"],
            errors="coerce"
        )

    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(
            df["tenure"],
            errors="coerce"
        )

    # Create GE dataset
    ge_df = ge.dataset.PandasDataset(df)

    print("Validating schema and required columns... - validate_data.py:40")

    # Required columns
    required_columns = [
        "customerID",
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "InternetService",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    for col in required_columns:
        ge_df.expect_column_to_exist(col)

    # customerID cannot be null
    ge_df.expect_column_values_to_not_be_null("customerID")

    print("Validating business logic constraints... - validate_data.py:62")

    ge_df.expect_column_values_to_be_in_set(
        "gender",
        ["Male", "Female"]
    )

    ge_df.expect_column_values_to_be_in_set(
        "Partner",
        ["Yes", "No"]
    )

    ge_df.expect_column_values_to_be_in_set(
        "Dependents",
        ["Yes", "No"]
    )

    ge_df.expect_column_values_to_be_in_set(
        "PhoneService",
        ["Yes", "No"]
    )

    ge_df.expect_column_values_to_be_in_set(
        "InternetService",
        ["DSL", "Fiber optic", "No"]
    )

    ge_df.expect_column_values_to_be_in_set(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    print("Validating numeric ranges... - validate_data.py:94")

    ge_df.expect_column_values_to_be_between(
        "tenure",
        min_value=0,
        max_value=120
    )

    ge_df.expect_column_values_to_be_between(
        "MonthlyCharges",
        min_value=0,
        max_value=200
    )

    ge_df.expect_column_values_to_be_between(
        "TotalCharges",
        min_value=0
    )

    print("Validating null values... - validate_data.py:113")

    ge_df.expect_column_values_to_not_be_null(
        "tenure"
    )

    ge_df.expect_column_values_to_not_be_null(
        "MonthlyCharges"
    )

    print("Validating statistical properties... - validate_data.py:123")

    ge_df.expect_column_mean_to_be_between(
        "tenure",
        min_value=0,
        max_value=120
    )

    ge_df.expect_column_median_to_be_between(
        "MonthlyCharges",
        min_value=0,
        max_value=200
    )

    print("Running complete validation suite... - validate_data.py:137")

    results = ge_df.validate()

    failed_expectations = []

    for result in results["results"]:
        if not result["success"]:
            failed_expectations.append(
                result["expectation_config"]["expectation_type"]
            )

    total_checks = len(results["results"])
    passed_checks = sum(
        1 for result in results["results"]
        if result["success"]
    )
    failed_checks = total_checks - passed_checks

    if results["success"]:
        print(
            f"✅ Data validation PASSED: "
            f"{passed_checks}/{total_checks} checks successful"
        )
    else:
        print(
            f"❌ Data validation FAILED: "
            f"{failed_checks}/{total_checks} checks failed"
        )
        print("Failed expectations: - validate_data.py:166", failed_expectations)

    return results["success"], failed_expectations