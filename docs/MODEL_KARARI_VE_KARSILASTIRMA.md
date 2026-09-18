# EŞİK: XGBoost ve LightGBM karşılaştırması

15 Eylül 2026. Ayşe Simal Alpözer ve Berat Bolat.

Bu paketin çalışan modeli **XGBoost**. Bu sürümü, beş dış katın ortalama ilk %10'da yakalama ölçütüne göre tercih ediyoruz. Tarihsel test LightGBM lehine; XGBoost'un her bakımdan üstün olduğu sonucu çıkmıyor.

| Ölçüm | XGBoost | LightGBM | Daha iyi sonuç |
|---|---:|---:|---|
| 5 dış katta ortalama ilk %10'da yakalama | %79,15 | %77,61 | XGBoost |
| Katlar arası standart sapma, yüzde puan | 2,76 | 3,71 | XGBoost daha az değişti |
| Dış kat ortalama Average Precision | 0,7459 | 0,7520 | LightGBM |
| Tarihsel testte yakalanan iflas / toplam | 67 / 82 | 70 / 82 | LightGBM |
| Tarihsel testte ilk %10'da yakalama | %81,71 | %85,37 | LightGBM |
| 117 incelemede isabet | %57,26 | %59,83 | LightGBM |
| Yanlış alarm | 50 | 47 | LightGBM |
| Kaçırılan iflas | 15 | 12 | LightGBM |
| Kalibre Brier, tarihsel test; düşük daha iyi | 0,02901 | 0,02689 | LightGBM |
| Kalibre log loss, tarihsel test; düşük daha iyi | 0,10761 | 0,10325 | LightGBM |

## Karar ve önceki anlatımın düzeltilmesi

Önceki ortak iç arama, tüm geliştirme verisinde LightGBM'i seçmişti: iç yakalama %79,13; XGBoost için %77,30. Bu seçim kodda gerçekten vardı. Eski tabloda XGBoost'un %79,15 dış kat ortalaması da gerçekten daha yüksekti. Aynı ölçüm değiller. Eski seçim gerekçesini açıklamak ile LightGBM'in genel olarak daha iyi olduğunu söylemek farklı iddialar. İkincisini bu tablo tek başına desteklemiyordu.

Yeni pakette seçim gerekçesi açıkça değişti: karşılaştırılan ailelerin beş dış kat ortalamasında ilk %10'da yakalaması. Bu ölçütte XGBoost +1,54 yüzde puan önde. Tarihsel test sonucuna bakarak ayar değiştirmedik. Testte LightGBM üç iflas daha fazla yakaladı; bu dezavantajı XGBoost sunumunda da gösteriyoruz.

**Öneri:** İlk %10'da yakalamanın gelişim katları ortalamasını karar ölçütü olarak koruyorsak, bu XGBoost sürümünü kullanabiliriz. Üretim için kesin kazanan ilan etmiyoruz. LightGBM güçlü alternatif olarak korunuyor. Yeni, daha önce kullanılmamış veri üzerinde iki sabit adayı değerlendirmek, hangisinin yeni şirketlerde daha iyi olduğunu belirlemek için sonraki adımdır.

## Karşılaştırma nasıl yapıldı?

- Aynı 5.850 temiz kayıt, aynı 4.680 geliştirme ve 1.170 tarihsel test kaydı, aynı 64 oran.
- Aynı kaydedilmiş 5 dış kat, her eğitim bölümünde aynı 3 iç kat. Median imputer yalnız eğitim verisinde öğrenir.
- Her aile için önceki Optuna aramasının aynı 40 denemelik bütçesinde seçilen parametreler kullanıldı. Yeni Optuna denemesi yapılmadı. İki ailenin dış kat sonuçları yeniden eğitilerek eski sonuçlarla sayısal olarak eşleştirildi.
- Her final aday, kendi tam geliştirme iç aramasındaki parametreleriyle 4.680 kayıtta yeniden eğitildi. LightGBM'in final tahminleri eski çalışan modelle birebir aynı çıktı.
- Her aile için ayrı eğitim dışı ham tahminler, ayrı sigmoid kalibratörü, ayrı SHAP ve ayrı dosya hash'leri üretildi.
- Dış katlar daha önce incelendi ve şimdi aile tercihinde kullanılıyor. %79,15'i yeni bağımsız doğrulama diye sunmuyoruz. Tarihsel test de daha önce görülmüş bir karşılaştırma örneklemidir; tarihe göre ileriye dönük yeni test değildir.

## Kalibrasyon

Model predict_proba ile zaten ham olasılık tahmini üretir. Sigmoid, ham marj ile eğitim dışı tahminlerin gerçek etiketleri arasındaki ilişkiyi öğrenerek bu tahmini ayarlar. Bir şirketin yüzdesi bireysel model çıktısıdır; grafikteki gruplar yalnız uyumu değerlendirmek içindir. Pozitif eğim sıralamayı korur; kalibrasyon ilk 117 şirketi değiştirmedi.

Her dış katın etiketleri o katın kalibratör eğitiminde kullanılmadı. Ancak model ayarları ve nihai kalibrasyon tasarımı daha önce incelenmiş veriye dayanır; deney seçimin ardından yapılmış keşifsel ölçümdür. Brier hem kalibrasyon hem ayrım gücünü içerir.

| Geliştirme dış katlarında Brier | Ham | Kalibre |
|---|---:|---:|
| XGBoost | 0,03445 | 0,03178 |
| LightGBM | 0,03358 | 0,03025 |

## Özellik mühendisliği ve PCA

XGBoost ile tekrar hesaplandı. Sabit 400 ağaçlı karşılaştırmada 64 ham oran %77,60, dört ek göstergeyle 68 girdi %76,07 yakalama verdi. Fark −1,53 yüzde puan. Ana sürüm 64 oranı korur. Bu keşifsel karşılaştırma yeni bağımsız doğrulama değildir.

PCA deneyinde her katın XGBoost ayarları sabit tutuldu. Ham 64 oran %79,15, ölçeklenmiş 64 oran %78,84; PCA %90 varyans %41,70, %95 varyans %43,23, %99 varyans %43,24, 64 bileşenli dönüşüm %54,91 yakalama verdi. PCA'ya özel yeni optimizasyon yapılmadı. Bu kurulumda PCA eklemek desteklenmedi; PCA'nın her problemde kötü olduğu iddia edilmiyor.

## Belirsizlik ve kanıt dosyaları

Tarihsel testte 3.000 eşlenmiş, sınıfa göre örneklemeli bootstrap tekrarında XGBoost eksi LightGBM yakalama farkının yüzde 95 yüzdelik aralığı −9,76 ile 0,00 yüzde puan. Bu yalnız mevcut örneklemin duyarlılık analizidir; geçmiş seçim yanlılığını gidermez ve yeni nüfusta üstünlük kanıtı değildir.

Tam sayısal kayıt: `karsilastirma/comparison_report.json`. Şirket bazında eşlenmiş sonuç: `paired_historical_company_results.csv`. Dış katlar: `paired_outer_folds.csv`. Model ile skor/SHAP uyumu: `outputs/artifact_manifest.json`. Kalibrasyon uyumu: `outputs/calibration_v1/manifest.json`.

Yöntem kaynakları: [scikit-learn iç içe doğrulama](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), [olasılık kalibrasyonu](https://scikit-learn.org/stable/modules/calibration.html).
