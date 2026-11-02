# Crypto Scanner Mobile

Aplikasi Streamlit untuk pemindaian market crypto.

## Jalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud
1. Upload `app.py` dan `requirements.txt` ke repository GitHub.
2. Buka https://share.streamlit.io/ dan login dengan GitHub.
3. Pilih repository, branch, dan file `app.py`.
4. Deploy. Setelah selesai, buka URL aplikasi dari HP.

Aplikasi mengambil data publik dari CoinGecko dan Binance Spot API. Tidak ada API key atau kredensial exchange yang diperlukan. Sinyal bersifat heuristik dan bukan jaminan hasil trading.
