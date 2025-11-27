# drop_columns_script.py

import pandas as pd
from pathlib import Path
from typing import List

def drop_unwanted_columns(
    input_path: str,
    output_path: str,
    columns_to_drop: List[str],
    sep: str = ","
) -> None:
    """
    Load CSV, drop specified columns, save result.

    :param input_path: đường dẫn file dữ liệu đầu vào
    :param output_path: đường dẫn file lưu kết quả
    :param columns_to_drop: danh sách tên cột muốn drop
    :param sep: separator của CSV (mặc định ",")
    """
    df = pd.read_csv(input_path, sep=sep)
    print("Columns before drop:", df.columns.tolist())

    df = df.drop(columns=columns_to_drop, errors="ignore", axis=1)

    print("Columns after drop:", df.columns.tolist())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    # Cấu hình
    input_file = "bank.csv"      # sửa đường dẫn nếu cần
    output_file = "bank1.csv"
    # Các cột bạn muốn loại bỏ
    columns_to_drop = ["day", "contact", "default", "previous", "y"]

    # Nếu file dùng separator khác (ví dụ ";" thì sửa sep=";")
    drop_unwanted_columns(
        input_path=input_file,
        output_path=output_file,
        columns_to_drop=columns_to_drop,
        sep=";"
    )
