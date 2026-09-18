# Finansal gösterge tanımlarının kaynağı

Uygulamadaki Türkçe açıklamalar `feature_descriptions.json` dosyasında tutulur. Bunlar sütun açıklamalarıdır; kaynak ARFF değerleri yeniden hesaplanmaz veya değiştirilmez.

Bu revizyonda Attr11'de eksik olan olağandışı kalemler, Attr25'in sermaye düzeltmesi, Attr40'ın alacakları da dışlayan payı ve Attr57'nin uzun formülü açıklaştırıldı. Kısa finansal adlar birimler/formüller yerine geçmez; özellikle gün cinsinden devir göstergeleri birbirinin aynısı kabul edilmemelidir.

**Attr48 ve Attr49'da kaynak belirsizliği vardır.** UCI açıklaması EBITDA adını kullanırken parantez içinde faaliyet kârından amortismanı çıkaran ifade verir. Ad ile aritmetik tarif uyumlu değildir. Bu yüzden eski sözlükteki toplama varsayımı kaldırıldı; ekranda kaynak etiketi ve tanım notu gösterilir. Ham mali tablolar veya veri sağlayıcısı doğrulaması olmadan bu sütunları kesin bir muhasebe formülüyle yeniden üretmek doğru değildir. Yeni gerçek şirket verisi kabulünden önce bu tanım çözülmelidir.

Attr5'in kısa adı, nakit/menkul kıymet/alacak/yükümlülük ve gider/amortisman kalemlerini içeren gün ölçekli bir yeterlilik ölçüsünü özetler. Attr41 de 12/365 ölçek çarpanı içerir. Dashboarddaki kısa adlardan hareketle yeni veri formülleri tahmin edilmemelidir.

Kaynak ve veri atfı: [Tomczak, S. (2016), Polish Companies Bankruptcy, UCI](https://www.archive.ics.uci.edu/dataset/365/polish%2Bcompanies%2Bbankruptcy%2Bdata), [DOI: 10.24432/C5F600](https://doi.org/10.24432/C5F600). Kaynak veri CC BY 4.0 lisanslıdır. Bu proje 5year altkümesini, tam tekrarları çıkarılmış ve demo kimlikleri eklenmiş biçimde kullanır. Kaynak tanımlarını kontrol tarihi: 13 Eylül 2026.
