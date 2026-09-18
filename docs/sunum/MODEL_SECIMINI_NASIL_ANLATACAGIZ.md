# Model seçimi için kısa konuşma

“LightGBM ve XGBoost'u aynı veri ayrımları ve aynı Optuna deneme bütçesiyle karşılaştırdık. İlk %10 inceleme listesinde iflasları yakalamayı ana ölçüt aldık. Beş dış katın ortalamasında XGBoost %79,15, LightGBM %77,61 verdi. Bu karşılaştırma sürümünde XGBoost'u bu nedenle tercih ettik. Ancak aynı tarihsel testte LightGBM 82 iflasın 70'ini, XGBoost 67'sini yakaladı. XGBoost'un her ölçütte üstün olduğunu iddia etmiyoruz. Bu katlar ve test daha önce incelendi; yeni bağımsız veride doğrulama gerekiyor.”

“%79,15 veya %81,71 genel doğruluk değildir. Gerçek iflasların ne kadarını ilk %10'luk inceleme listesine aldığımızdır. XGBoost tarihsel testte 117 şirketi seçti, bunların 67'si gerçekten iflas etiketliydi. Bu yüzden yakalama 67/82=%81,71, listedeki isabet 67/117=%57,26.”

“Model zaten ham olasılık tahmini verir. Sigmoid kalibrasyon bu yüzdeleri eğitim dışı tahminler üzerinden düzeltir. Kalibrasyon şirketlerin sırasını değiştirmedi. ESIK-05817 XGBoost'ta üçüncü sırada; kalibre tahmini %99,0. Bu bireysel tahmini kesin sonuç olarak sunmuyoruz.”

Eski ortak iç seçim yönteminin LightGBM'i seçtiği gerçeği korunur. Yeni sürümde karar kuralının ailelerin dış kat ortalamalarını karşılaştırmak olduğu açıkça belirtilir. “Önceden hep XGBoost seçmiştik” veya “LightGBM yanlış modeldi” denmemelidir.
