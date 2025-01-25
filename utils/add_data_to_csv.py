import pandas as pd
import os

from utils.google_translate import check_lang

CSV_FILE = "datasets/propaganda_on_sentence_level_ua.csv"


def add_data_to_csv(input_text: str):
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"The dataset file at {CSV_FILE} does not exist.")

    new_data = []
    for line in input_text.strip().split('\r\n'):
        parts = [part.lower().strip() for part in line.split(";")]
        if len(parts) != 2 or not check_lang(parts[0], "en") or parts[1] not in ["non-propaganda", "propaganda"]:
            if parts[0] == "text" or parts[1] == "class":
                continue
            raise ValueError(f"Invalid input format: '{line}'. Each line must contain exactly one semicolon. The text "
                             f"must be Ukrainian. The class must be either 'non-propaganda' or 'propaganda'")
        new_data.append(parts)

    # Create a DataFrame from the new data
    new_df = pd.DataFrame(new_data, columns=['Text', 'Class'])

    # Load the existing dataset
    existing_df = pd.read_csv(CSV_FILE, sep=';', encoding='utf-8')

    # Remove duplicates that already exist in the dataset
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.drop_duplicates(subset=['Text', 'Class'], keep='first', inplace=True)

    # Filter only the new rows that are not in the existing dataset
    updated_df = combined_df[~combined_df[['Text', 'Class']].isin(existing_df[['Text', 'Class']].to_dict(orient='list')).all(axis=1)]

    # Append the new data to the existing dataset
    final_df = pd.concat([existing_df, updated_df], ignore_index=True)

    # Save the updated dataset back to the file
    final_df.to_csv(CSV_FILE, sep=';', index=False, encoding='utf-8')
    print(f"Appended the provided text to {CSV_FILE}")


# Example usage
if __name__ == "__main__":
    CSV_FILE = "../datasets/propaganda_on_sentence_level_ua.csv"
    input_text = "Текст українською мовою;non-propaganda\r\nТрохи тексту не завадить;propaganda\r\nБагато багато тексту;non-propaganda\r\n"

    try:
        add_data_to_csv(input_text)
    except FileNotFoundError as e:
        print(e)
