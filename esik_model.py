##################################################
# EŞİK - MODEL EĞİTİM GİRİŞİ
# İlk GridSearch çalışması reference/v1/esik_model.py içinde korunur.
# Yeni açıklamalı eğitim akışı esik_optuna.py dosyasındadır.
# python esik_model.py pilot --name yeni-pilot --trials 5
##################################################
from esik_optuna import main

if __name__ == "__main__":
    main()
