import rpy2.robjects as robjects
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from rpy2.robjects import pandas2ri
import pandas as pd
import polars as pl

# Read the R code from the file
with open("src/R/art.r", "r") as f:
    r_code = f.read()

# Create the R package
tda_package = SignatureTranslatedAnonymousPackage(r_code, "tda_package")

def analyze_with_r(pl_df, homology_dimension):
    """
    Convert a Polars DataFrame to Pandas, then analyze it using the R function

    Args:
        pl_df: A Polars DataFrame with tda data

    Returns:
        The results from the R analysis
    """
    # Convert Polars to Pandas
    pd_df = pl_df.to_pandas()

    # Activate the pandas converter for rpy2
    pandas2ri.activate()

    # Convert pandas dataframe to R dataframe
    r_df = pandas2ri.py2rpy(pd_df)

    # Call the R function
    results = tda_package.analyze_tda_data(r_df, homology_dimension)

    # Deactivate the pandas converter
    pandas2ri.deactivate()

    return results
