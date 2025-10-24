# import pandas as pd

# def main():
#     input_file = "data/extracted_with_www.csv"   # from previous script
#     output_file = "data/extracted_final.csv"

#     print("Loading CSV...")
#     df = pd.read_csv(input_file)

#     if "Normalized_URL" not in df.columns:
#         raise ValueError("Missing 'Normalized_URL' column. Run the normalization script first.")

#     # Replace the URL column with the normalized version
#     print("Replacing original URLs with normalized URLs...")
#     df["URL"] = df["Normalized_URL"]

#     # Drop helper columns if they exist
#     for col in ["Normalized_URL", "Status", "Original_URL"]:
#         if col in df.columns:
#             df.drop(columns=[col], inplace=True)

#     # Save the cleaned CSV
#     df.to_csv(output_file, index=False)
#     print(f"✅ Done! Cleaned file saved as {output_file}")

# if __name__ == "__main__":
#     main()

import pandas as pd

# PhiUSIIL standard feature columns (must match ALL_COLUMNS)
PHIUSIIL_COLUMNS = [
    "FILENAME",
    "URL",
    "URLLength",
    "Domain",
    "DomainLength",
    "IsDomainIP",
    "TLD",
    "URLSimilarityIndex",
    "CharContinuationRate",
    "TLDLegitimateProb",
    "URLCharProb",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
    "LineOfCode",
    "LargestLineLength",
    "HasTitle",
    "Title",
    "DomainTitleMatchScore",
    "URLTitleMatchScore",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
    "HasDescription",
    "NoOfPopup",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfExternalRef",
    "label",
]

# def main():
#     input_file = "data/extracted_final.csv"  # already updated URLs
#     output_file = "data/extracted_final2.csv"

#     print("Loading CSV...")
#     df = pd.read_csv(input_file)

#     # Keep only columns that are part of PhiUSIIL schema
#     cols_to_keep = [c for c in PHIUSIIL_COLUMNS if c in df.columns]
#     cols_to_drop = [c for c in df.columns if c not in PHIUSIIL_COLUMNS]

#     if cols_to_drop:
#         print(f"🧹 Dropping {len(cols_to_drop)} extra columns: {cols_to_drop}")
#     if missing := [c for c in PHIUSIIL_COLUMNS if c not in df.columns]:
#         print(f"⚠️ Warning: Missing {len(missing)} expected columns: {missing}")

#     df_cleaned = df[cols_to_keep]
#     df_cleaned.to_csv(output_file, index=False)

#     print(f"✅ Cleaned file saved as: {output_file}")
#     print(f"Final columns count: {len(df_cleaned.columns)} (PhiUSIIL standard)")

# if __name__ == "__main__":
#     main()

# import pandas as pd

# def main():
#     input_file = "data/extracted_final.csv"
#     output_file = "data/extracted_labeled.csv"

#     print("Loading CSV...")
#     df = pd.read_csv(input_file)

#     if "label" not in df.columns:
#         raise ValueError("❌ The 'label' column is missing from the input CSV.")

#     print(f"Setting all {len(df)} label values to 1...")
#     df["label"] = 1

#     df.to_csv(output_file, index=False)
#     print(f"✅ Done! All labels set to 1 and saved as {output_file}")

# if __name__ == "__main__":
#     main()

