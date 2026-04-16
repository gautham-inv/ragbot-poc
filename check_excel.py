import pandas as pd
import json

file_path = r"C:\Users\gauth\Downloads\260410 EXCEL GLORIA 2026_V60.xlsx"
try:
    xls = pd.ExcelFile(file_path)

    results = {}
    all_columns_sets = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet, nrows=0)
        cols = df.columns.tolist()
        # Convert columns to string, handle Unnamed columns if any
        clean_cols = [str(c).strip() for c in cols]
        results[sheet] = clean_cols
        all_columns_sets.append(set(clean_cols))

    if all_columns_sets:
        common_cols = set.intersection(*all_columns_sets)
        union_cols = set.union(*all_columns_sets)
    else:
        common_cols = set()
        union_cols = set()

    output = {
        "status": "success",
        "tabs": list(results.keys()),
        "columns_per_tab": results,
        "common_columns": list(common_cols),
        "all_unique_columns": list(union_cols)
    }
except Exception as e:
    output = {
        "status": "error",
        "message": str(e)
    }

print(json.dumps(output, indent=2))
