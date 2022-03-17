"""
Import, format and tidy data.

This script imports the data stored in the "data" folder, and save a .csv (or .nc, TBD.) with the formatted dataset.
The data sources can vary, therefore, the script needs to identify the source (not explicitly provided) and adapt to
them. Some meta-data (e.g. the podcast name) is in the file name itself, so the script needs to identify it as well.
"""

import numpy as np
import pandas as pd
