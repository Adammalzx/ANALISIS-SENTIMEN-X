import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
import os

print("MEMBUAT TOKENIZER...")
print("=" * 50)

# Cek file data
if not os.path.exists('train_preprocess_ori.tsv'):
    print("ERROR: File 'train_preprocess_ori.tsv' tidak ditemukan!")
    print("Files in current directory:")
    for file in os.listdir('.'):
        print("   - " + file)
    exit()

# Load data
try:
    print("Loading data...")
    train_df = pd.read_csv('train_preprocess_ori.tsv', sep='\t')
    print("Data loaded: " + str(train_df.shape))
    print("Columns: " + str(train_df.columns.tolist()))
    
    # Tampilkan sample data
    print("\nSample data:")
    print(train_df.head(2))
    
except Exception as e:
    print("Error loading data: " + str(e))
    exit()

# Tentukan kolom teks
text_columns = ['text', 'tweet', 'review', 'comment', 'sentence']
text_column = None

for col in text_columns:
    if col in train_df.columns:
        text_column = col
        break

if text_column is None:
    print("Using first column as text")
    text_column = train_df.columns[0]

print("Using column: '" + text_column + "'")

# Ambil teks
try:
    texts = train_df[text_column].astype(str).values
    print("Texts extracted: " + str(len(texts)) + " samples")
    if len(texts) > 0:
        print("Sample: '" + str(texts[0][:50]) + "...'")
except Exception as e:
    print("Error extracting texts: " + str(e))
    exit()

# Buat tokenizer
try:
    print("\nCreating tokenizer...")
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    print("Tokenizer created!")
    print("Vocabulary size: " + str(len(tokenizer.word_index)))
except Exception as e:
    print("Error creating tokenizer: " + str(e))
    exit()

# Simpan tokenizer
try:
    print("\nSaving tokenizer...")
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print("Tokenizer saved as 'tokenizer.pkl'")
    
    # Cek file size
    file_size = os.path.getsize('tokenizer.pkl')
    print("File size: " + str(file_size) + " bytes")
    
except Exception as e:
    print("Error saving tokenizer: " + str(e))
    exit()

# Simpan config
try:
    config = {'max_length': 100}
    with open('model_config.pkl', 'wb') as f:
        pickle.dump(config, f)
    print("Config saved as 'model_config.pkl'")
except Exception as e:
    print("Error saving config: " + str(e))

print("\nTOKENIZER BERHASIL DIBUAT!")
print("Sekarang jalankan: streamlit run app.py")