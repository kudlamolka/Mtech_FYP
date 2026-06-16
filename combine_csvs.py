import logging
import os
import glob
import sys

import pandas as pd

logger = logging.getLogger(__name__)


def combine_csv_files(folder_path, output_file_name="combined_output.csv"):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    search_path = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(search_path)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {folder_path}")

    print(f"Found {len(csv_files)} CSV files. Combining them now...")

    df_list = []
    failed_files = []
    for file in csv_files:
        file_name = os.path.basename(file)
        stock_name = file_name.split('_')[0]

        try:
            df = pd.read_csv(file)
            df.insert(0, 'Stock', stock_name)
            df_list.append(df)
        except Exception as e:
            logger.warning("Could not read %s: %s", file_name, e)
            failed_files.append(file_name)
            continue

    if failed_files:
        logger.warning(
            "%d of %d files failed to load: %s",
            len(failed_files), len(csv_files), failed_files,
        )

    if not df_list:
        raise ValueError(
            f"All {len(csv_files)} CSV files failed to load — nothing to combine."
        )

    combined_df = pd.concat(df_list, ignore_index=True)
    output_path = os.path.join(folder_path, output_file_name)
    combined_df.to_csv(output_path, index=False)
    print(f"Success! Combined file saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    TARGET_FOLDER = "./results"
    try:
        combine_csv_files(TARGET_FOLDER)
    except Exception:
        logger.exception("combine_csv_files failed")
        sys.exit(1)
