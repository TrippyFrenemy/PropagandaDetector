import pandas as pd
import os

from utils.google_translate import check_ua

CSV_FILE = "datasets/propaganda_on_sentence_level_ua.csv"


def add_data_to_csv(input_text: str):
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"The dataset file at {CSV_FILE} does not exist.")

    new_data = []
    for line in input_text.strip().split('\r\n'):
        parts = [part.strip() for part in line.split(";")]
        if len(parts) != 2 or not check_ua(parts[0]) or parts[1] not in ["non-propaganda", "propaganda"]:
            raise ValueError(f"Invalid input format: '{line}'. Each line must contain exactly one semicolon. The text must be Ukrainian. The class must be either 'non-propaganda' or 'propaganda'")
        new_data.append(parts)

    # print([[l.strip() for l in line.split(";")] for line in input_text.strip().split('\r\n')])
    # # Split input text into lines and then split each line into text and class
    # new_data = [[line.strip() for line in lines.split(";")] for lines in input_text.strip().split('\r\n')]

    # Create a DataFrame from the new data
    new_df = pd.DataFrame(new_data, columns=['Text', 'Class'])

    # Load the existing dataset
    existing_df = pd.read_csv(CSV_FILE, sep=';', encoding='utf-8')

    # Append the new data to the existing dataset
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)

    # Save the updated dataset back to the file
    updated_df.to_csv(CSV_FILE, sep=';', index=False, encoding='utf-8')
    print(f"Appended the provided text to {CSV_FILE}")


# Example usage
if __name__ == "__main__":
    input_text = "Text1; propaganda\r\nText2; non-propaganda\r\n"

    try:
        add_data_to_csv(input_text)
    except FileNotFoundError as e:
        print(e)