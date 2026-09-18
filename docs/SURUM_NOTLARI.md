# XGBoost sürüm notları

Bu kopyada son model, skor dosyaları, 74.880 yerel SHAP katkısı, 1.170 şirketin olasılığı, 117 şirket raporu ve sunum birlikte yenilendi. Eski LightGBM projesi korunur.

XGBoost dış kat ortalama yakalamada %79,15, LightGBM %77,61. Tarihsel testte XGBoost 67, LightGBM 70 iflas yakaladı. Seçim gerekçesi ve karşıt bulgular docs/MODEL_KARARI_VE_KARSILASTIRMA.md dosyasında.

75 model/uygulama/bağlantı testi ve 6 Optuna protokol testi geçti. Sekiz Streamlit sayfası açıldı. Kaydedilmiş 1.170 ham skor ve kalibre olasılık yeniden hesaplanarak doğrulandı. 400 XGBoost boosting turu kayıttan okundu. Skor/SHAP yenileme aracı ayrı klasöre üretim yaparak doğrulandı.

Mevcut yerel n8n ve OpenAI bağlantısından XGBoost şirket sorgusu doğrulandı. Aynı bağlantı yerel yeni uygulamada etkin. Paylaşılan ZIP'te API anahtarları ve kişisel webhook tokenları bulunmaz. Ses kaydı yeniden alınmadı.

Varsayılan Streamlit portu 8502. Eski kopyanın 8501 portuyla karışmasını önlemek için bu kopyadaki başlatma betikleri güncellendi. Kanal sorgu API'si kullanılacaksa baslat-v3.ps1 aynı model hash'ine ait servisi kontrol eder; farklı kopyanın servisine sessizce bağlanmaz.

prepare_dashboard_data.py ve esik_shap.py aynı kayıtlı modelin çıktılarını yeni releases/refresh-* klasörüne üretir. Çalışan modeli otomatik değiştirmez. scripts/evaluate_calibration.py kayıtlı XGBoost kalibrasyonunun bütünlüğünü kontrol eder; yeniden kalibrasyon karsilastirma/compare_esik_families.py ile yapılır. Özgün Optuna aday dosyaları geçmiş deney kanıtıdır.
