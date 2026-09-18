# EŞİK · Sahne ve konuşmacı notları

16 Eylül 2026 sürümü. 15 görünür slayt + 4 gizli teknik ek. Yaklaşık 13 dakika; canlı demo buna dahildir.

Sunumu masaüstü PowerPoint’te Slayt Gösterisi modunda açın. 2, 6 ve 14. slaytlarda içeriği açmak için bir ek tıklama vardır. Slaytlar kendiliğinden ilerlemez.

Eski Miuul sunumundaki 11 Morph geçişi geri yüklendi. Ortak EŞİK çizgisi slaytlar arasında eşleştirildi. Diğer geçişler kısa Fade olarak korundu.

Geçiş ve animasyon tanımları dosyada doğrulandı; bu bilgisayarda masaüstü PowerPoint oynatımı denenmedi. Sunum bilgisayarında kısa bir F5 provası yapın. PDF ve statik önizleme hareketi göstermez.

Kaynaklar: [Microsoft Morph açıklaması](https://support.microsoft.com/en-us/powerpoint/morph-transition-tips-and-tricks), [PresentationML animasyon yapısı](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.presentation.animateeffect?view=openxml-3.0.1).

## 1

00:00–00:35 | EŞİK, sınırlı analist zamanını önce hangi şirkete ayıracağımızı belirleyen bir karar destek prototipi. Ayşe Simal Alpözer ve Berat Bolat olarak tüm çalışmaları birlikte yürüttük. Miuul projemiz birinci oldu. Bugün hem verideki kanıtı hem çalışan inceleme akışını göstereceğiz.

## 2

00:35–01:20 | Önce yalnız soldaki doğruluğa dikkat çekin. Her şirkete iflas etmez demek tarihsel testte %92,99 doğruluk sağlar. Bir kez tıklayın: fakat 82 iflastan sıfırını yakalar. Bu nedenle doğruluk yerine ilk %10 inceleme listesinde yakalanan gerçek iflas oranını önceliklendirdik. %10 bir kapasite varsayımıdır, iflas olasılığı eşiği değildir.

## 3

01:20–02:00 | Kaynak UCI Polish Companies Bankruptcy, bir yıllık ufuk. 5.910 kayıttan 60 tam tekrarı çıkardık. 5.850 kaydı 4.680 geliştirme ve 1.170 tarihsel test olarak ayırdık. Bu anonim, tarihsel Polonya verisidir; güncel Türkiye şirketlerinde doğrulanmış bir sistem olduğunu söylemiyoruz.

## 4

02:00–02:45 | İflas sınıfı %6,97 ile azınlık. En az bir girdisi eksik kayıt oranı %48,80; Attr37 eksikliği %43,16. Bölmeleri sınıf oranını koruyarak yaptık. Medyan ve sınıf ağırlığı yalnız eğitim verisinden hesaplandı. Test medyanını öğrenmedik. Uç finansal değerleri otomatik olarak silmedik.

## 5

02:45–03:25 | Dört ek özellik denedik: eksiklik oranı, kısa vadeli borç payı, zarar ile likidite ve satış düşüşünü birlikte işaretleyen göstergeler. Bu sabit ayarlı keşifsel karşılaştırmada yakalama 1,53 yüzde puan düştü. Final model 64 kaynak oranı koruyor. Özellik sayısını artırmak tek başına başarı garantisi değil.

## 6

03:25–04:30 | Her dış turda 3.744 kayıtla çalışıyoruz. Bunun içinde üç katlı doğrulama ve her aile için 40 Optuna denemesi ile ayarlar seçiliyor. Bir kez tıklayın: 936 kayıt bu ayar aramasının dışında tutuluyor ve değerlendirmede kullanılıyor. Beş dış tur ve her aile için son ayar araması toplam 1.440 iç eğitime karşılık geliyor. Son aile tercihi dış kat ortalamalarına dayandığı için bu katları yeni bağımsız test gibi sunmuyoruz.

## 7

04:30–05:20 | Aynı dış katlarda ilk %10 yakalamada Optuna XGBoost %79,15, LightGBM %77,61. İş hedefimizde 1,54 yüzde puanlık ortalama fark nedeniyle XGBoost sürümünü tercih ettik. AP bütün sıralamayı özetleyen ayrı metriktir; bu metrikte LightGBM bir miktar önde. XGBoost her ölçütte üstün iddiasında bulunmuyoruz.

## 8

05:20–05:55 | Beş çubuk, beş dış değerlendirme sonucudur. Ortalama %79,15; katlar arası standart sapma 2,76 yüzde puan. Bu, bir şirketin %79,15 iflas edeceği anlamına gelmez. İlk %10 listeye gerçek iflasların ne kadarının girdiğini ölçer.

## 9

05:55–06:45 | Tarihsel testte 1.170 kaydın ilk 117 tanesini inceliyoruz. 82 gerçek iflasın 67 tanesi burada: yakalama %81,71. Listedeki isabet 67/117, yani %57,26. 50 yanlış alarm ve 15 kaçırılan iflas var. Aynı tarihsel testte LightGBM 70 iflas yakalamıştı; bu karşı kanıtı saklamıyoruz. Test önceden incelendi; yeni bağımsız doğrulama değil.

## 10

06:45–07:35 | XGBoost zaten ham bir olasılık tahmini verir. Sigmoid kalibrasyon ile bu tahminlerin gözlenen iflas sıklıklarıyla uyumunu iyileştirmeyi denedik. Brier .0344’ten .0318’e, log loss .1391’den .1207’ye düştü. Bu ölçümlerde küçük daha iyidir. Yeniden kullanılan 4.680 dış kat kaydında keşifsel bir deney; kesin bireysel olasılık veya yeni dış doğrulama iddiası yok.

## 11

07:35–08:25 | ESIK-05817 üçüncü sırada. En güçlü pozitif SHAP katkısı Attr21’den geliyor fakat bu girdi eksik ve eğitim medyanıyla doldurulmuş. Burada analistin ilk işi kaynak mali tabloyu doğrulamak. Üç çubuğun üçü de ham skoru artırıyor. SHAP log-odds katkısıdır; yüzde puan, nedensellik veya kesin iflas nedeni değildir.

## 12

08:25–09:00 | 117’de durmak bir kapasite kararıdır. 125. sıradaki ESIK-05511, düşük görünen kalibre tahminine rağmen tarihsel olarak iflas etmiş. Eksik girdisi de yok. Model hata yapar; düşük skor iflası dışlamaz.

## 13

09:00–09:40 | Kapasiteyi %10’dan %20’ye çıkarınca 117 ek inceleme ile sekiz ek iflas yakalanıyor. Yakalama %91,46’ya yükseliyor. Bunun ekonomik karşılığı için inceleme maliyeti ve kaçırılan vakaların maliyeti ayrıca ölçülmeli. Veride olmayan bir parasal kazanç söylemiyoruz.

## 14

09:40–12:10 | CANLI DEMO. Bir kez tıklayarak ses/rapor adımını açın. http://127.0.0.1:8503 adresine geçin. Öncelik listesinden bir satır seçip şirketi açın; ESIK-05817 örneğini gösterin. Eksik Attr21 ile model girdisini ayırın. Senaryo kısayolunda tek gözlenen oranı değiştirip model skorunun duyarlılığını gösterin. Rapor paylaşımında HTML önizlemesini açın; e-posta yalnız Gönder düğmesi ile gider. Mikrofon opsiyoneldir: sesi metne çevirir, kullanıcı metni kontrol eder. Sayısal değerleri Python/model üretir; dil modeli açıklamaya yardım eder. Bağlantı aksarsa raporu yerel HTML olarak açın; tekrar tekrar gönderim yapmayın.

## 15

12:10–13:00 | 117 incelemede 67 tarihsel iflası önceliklendiren; gerekçeyi, eksik veriyi, senaryoyu ve paylaşılabilir raporu bir araya getiren prototip. Bir sonraki adım güncel yerel veride dış doğrulama ve analist geri bildirimi ile pilot. Soruları alabiliriz.

## 16

TEKNİK EK | PCA için aynı kayıtlı XGBoost ayarları kullanıldı, PCA’ya özel hiperparametre araması yapılmadı. Bu sonuç PCA her zaman kötüdür anlamına gelmez. Bu deneyde 64 özgün oranı koruduk.

## 17

TEKNİK EK | Streamlit WAV kaydını n8n’e gönderir; whisper-1 metne çevirir. Kullanıcı kontrol edip gönderir. Şirketin sayısal bilgileri kayıtlı model ve kalibrasyon çıktısından gelir. OpenAI dil modeli verilen bağlamı açıklar. API anahtarı bağlantıda saklanır; slaytta veya raporda bulunmaz.

## 18

TEKNİK EK | Gerçek uygulamadan ESIK-05817 şirketi. Ham skor, portföydeki yüzdelik sıra ve kalibre olasılık ayrı kavramlardır. %99,0 kalibrasyon tahmini keşifseldir.

## 19

TEKNİK EK | Bu örnek yanıt kayıtlı model çıktısından deterministik olarak üretilmiştir; örnek için OpenAI çağrısı yapılmamıştır. Dil modelinin hesaplaması gibi sunmayın.
