# EŞİK: XGBoost karşılaştırma sürümü

Ayşe Simal Alpözer ve Berat Bolat. Çalışmanın tüm aşamaları ortak yürütüldü.

Bu klasördeki model XGBoost'tur. `docs/MODEL_KARARI_VE_KARSILASTIRMA.md` seçim gerekçesini, LightGBM'in daha iyi olduğu sonuçları ve sınırlamaları açıklar.

## Çalıştırma

PowerShell'de bu klasöre gelin. İlk defa `powershell -ExecutionPolicy Bypass -File .\kurulum.ps1`, ardından `powershell -ExecutionPolicy Bypass -File .\baslat-xgboost.ps1` çalıştırın. Adres http://127.0.0.1:8502.

Mevcut Python ortamınızda bağımlılıklar kuruluysa: `python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502`.

Yerel şirket API'si: `python esik_query_server.py` (varsayılan adres ve yetkilendirme için `docs/v3-baglantilar.md`). n8n'in model hesabına gerek duyduğu akışlarda bu hizmet çalışmalıdır. Streamlit sayfasındaki metin ve şirket paketi HTTP ile n8n webhook'una gider. n8n Python dosyası açmaz; tanımlı iş akışını ve yapılandırılmış OpenAI bağlantısını kullanır.

`.streamlit/secrets.example.toml` dosyasını `.streamlit/secrets.toml` olarak kopyalayıp kendi yerel bağlantı ayarlarınızı girin. API anahtarını kaynak koda veya paylaşılan ZIP'e koymayın. n8n akışları önce içe alınmalı, kendi kimlik bilgilerinize bağlanmalı ve etkinleştirilmelidir. XGBoost bağlamıyla yerel n8n ve OpenAI üzerinden canlı şirket sorgusu doğrulandı. Ses kaydı bu sürümde yeniden alınmadı. Test kaydı docs/ESIK_XGBoost_canli_baglanti_testi.json dosyasında.

## Dosyalar

- `esik_core.py`: veri, imputer, modeller ve metrikler.
- `esik_optuna.py`: özgün 40 denemelik aile aramaları ve iç/dış kat ayrımları.
- `experiments/optuna-v2`: özgün deney kanıtı. Bu deneyin ortak iç seçim kazananı LightGBM'dir; buradaki `candidate.pkl` eski ortak arama adayını temsil eder. Yeni XGBoost modeli `models/esik_final_model_pipeline.pkl` dosyasıdır.
- `karsilastirma/compare_esik_families.py`: iki aileyi kaydedilmiş aynı parametrelerle yeniden eğiten, skor/SHAP/ayrı kalibrasyon ve karşılaştırma üreten betik.
- `esik_feature_engineering.py`, `esik_pca_karsilastirmasi.py`: XGBoost ile yeniden hesaplanan ek deneyler; sonuçları ilgili klasörlerde.
- `esik_calibration.py`, `esik_assessment.py`: olasılık katmanı ve deterministik şirket raporu.
- `app.py`, `esik_ui.py`, `assets`: Streamlit arayüzü.
- `esik_company_query.py`, `esik_query_server.py`, `n8n_client.py`, `n8n`: yerel şirket sorgusu ve otomasyon.

## Karşılaştırmayı yeniden üretme

Paketin `reference/ESIK_GUNCEL_TAM_PROJE_Berat_15_Eylul_2026.zip` dosyasını ayrı `reference/orijinal` klasörüne çıkarın. Sonra bu proje klasöründe:

```powershell
python karsilastirma/compare_esik_families.py --source ../reference/orijinal/esik-profesyonel --output ../yeniden_karsilastirma
```

Bu betik kaynak LightGBM projesini değiştirmez. Çıktı klasörü yeni olmalı; uzun süren eğitim ve kalibrasyon yapar. Aynı deney sırasında kaynak kod veya veri değişirse doğrulama durur. Yeni Optuna taraması değildir; kayıtlı aynı bütçenin seçtiği ayarları yeniden eğitir. `esik_model.py` özgün genel Optuna komutlarına giriş verir; onu çalıştırmak bu XGBoost sürümünü otomatik olarak yeniden üretmek anlamına gelmez.

Eski sunumlar ve eski kod öğrenme rehberi orijinal ZIP'te arşivdir. Bu sürümde güncel sayıların kaynağı `outputs` ve karşılaştırma raporudur.
