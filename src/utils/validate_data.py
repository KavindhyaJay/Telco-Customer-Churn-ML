import great_expectations as ge
from typing import Tuple, List

def validation_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.if the data is valid and a list of validation errors if any.

    """
    print("Starting data validation with Great Expectations... - validate_data.py:13")

    ge_df = ge.dataset.PandasDataset(df)

    print ("Validating schema and required columns... - validate_data.py:17" )

    ge_df.expect_column_to_exist("customerID")
    ge_df.expect_column_values_to_not_be_null("customerID")
    
    # Core demographic features
    ge_df.expect_column_to_exist("gender") 
    ge_df.expect_column_to_exist("Partner")
    ge_df.expect_column_to_exist("Dependents")
    
    # Service features (critical for churn analysis)
    ge_df.expect_column_to_exist("PhoneService")
    ge_df.expect_column_to_exist("InternetService")
    ge_df.expect_column_to_exist("Contract")
    
    # Financial features (key churn predictors)
    ge_df.expect_column_to_exist("tenure")
    ge_df.expect_column_to_exist("MonthlyCharges")
    ge_df.expect_column_to_exist("TotalCharges")

    print("Validation business logic constraints... - validate_data.py:37")

    ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])

    ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])

    ge_df.expect_column_values_to_be_in_set(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )

    ge_df.expect_column_values_to_be_between(
        "Internetservice",
        ["DSL", "Fiber optic", "No"]
    )

    print("Validating numeric ranges and business constraints... - validate_data.py:54")

    ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=72)

    ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0)

    print("Validation statistics properties... - validate_data.py:60")

    ge_df.expect_column_mean_to_be_between("tenure", min_value=0, max_value=120)

    ge_df.expect_column_median_to_be_between("MonthlyCharges", min_value=0, max_value=200)

    ge_df.expect_column_values_to_be_null("tenure")
    ge_df.expect_column_values_to_be_null("MonthlyCharges")

    print("Validation data consistency... - validate_data.py:69")

    ge_df.expect_column_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal = True,
        mostly = 0.95
    )

    print("Running Complete validation suite... - validate_data.py:78")
    results = ge_df.validate()

    failed_expectations = []
    for r in results["results"]:
        if not r["success"]:
            expectation_type = r["expectation_config"]["expectation_type"]
            failed_expectations.append(expectation_type)
    
    # Print validation summary
    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks
    
    if results["success"]:
        print(f"Data validation PASSED: {passed_checks}/{total_checks} checks successful - validate_data.py:93")
    else:
        print(f"Data validation FAILED: {failed_checks}/{total_checks} checks failed - validate_data.py:95")
        print(f"Failed expectations: {failed_expectations} - validate_data.py:96")
    
    return results["success"], failed_expectations