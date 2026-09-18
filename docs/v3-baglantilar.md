# EŞİK v3 — sesli soru ve mesajlaşma

14 Eylül 2026. **v3 uygulama güncellemesidir; ana model hâlâ optuna-v2 LightGBM'dir.** Orijinal ZIP değiştirilmedi. Bu rehber, önceki ses eklentisi rehberinin kurulum ve durum bilgisini günceller.

## Şu anki durum

| Özellik | Hazırlanan ve doğrulanan | Canlı kullanım için kalan |
|---|---|---|
| Tek satırda şirket sorgusu | Uygulama formu, doğru kayıt eşleştirme, skor/sıra/SHAP özeti; tarayıcıda denendi | Hesap gerekmiyor |
| Mikrofonla soru | Ses akışı etkin; sentetik WAV gerçek OpenAI servisine iletildi ve metin döndü. Düzenlenebilir taslak; 60 saniye/4 MiB sınırı | Kullanıcı kendi mikrofon denemesinin çalıştığını bildirdi; geniş doğruluk ölçümü yapılmadı |
| AI sohbeti | OpenAI bağlı; GPT-5.6 Luna gerçek istekte `ESIK-05511 | 135 / 1170` yanıtını doğru verdi | Kullanıma açık; kişisel hesap bağlantısı teslim ZIP’ine dahil değil |
| Telegram | Özel konuşmada gelen metin → şirket sorgusu → aynı konuşmaya yanıt; n8n taslağı içe aktarıldı | Bot, erişilebilir HTTPS webhook ve izin verilen kullanıcı kimliği |
| WhatsApp | Gelen metin → şirket sorgusu → aynı numaraya yanıt; n8n taslağı içe aktarıldı | Meta/WhatsApp Business Cloud kurulumu, HTTPS webhook ve izin verilen telefon numarası |

Yerel n8n'e iki Header Auth bağlantısı ve dört akış eklendi. Kullanıcı OpenAI bağlantısını kaydetti; **ses ve AI sohbet akışları yayımlandı**, Telegram/WhatsApp akışları pasif. Yerel uygulama ve şirket servisi etkin AI adresleriyle yeniden başlatıldı. OpenAI anahtarı yalnız n8n’de tutuluyor. Telegram/WhatsApp hesapları henüz bağlı değil; dış mesaj gönderilmedi. Kullanıcının önceki akışları değiştirilmedi.

Gerçek servis kontrolünde sohbet yanıtı yaklaşık **4,21 saniyede** geldi. Ses isteği yaklaşık **3,48 saniyede** metin döndürdü; bu süreler tek denemeye aittir. Test sesi İngilizce Microsoft Zira motoruyla üretilmiş Türkçe bir cümleydi; beklenen “Merhaba. Bu bir test.” yerine “Merhaba. Uluddin Birtest.” geldi. Bu test bağlantıyı doğrular, Türkçe mikrofon doğruluğunu kanıtlamaz. Kullanıcı metni gözden geçirip düzelttikten sonra gönderir.

## Hesapsız açılan demo

Tam proje ZIP'ini yeni bir klasöre çıkarın. Bu paket model, veri, kaynak kod ve kayıtlı deneyleri içerir; `.venv`, yerel anahtarlar ve çalışma günlüklerini içermez. Önceki eklenti ZIP'lerini üst üste uygulamanız gerekmez.

Gereken Python bağımlılıkları mevcutsa proje klasöründe:

```powershell
.\baslat-v3.ps1
```

Bu bilgisayardaki hazır Python ortamını açıkça seçmek gerekirse:

```powershell
.\baslat-v3.ps1 -Python C:\Python314\python.exe
```

Yeni bilgisayarda önce `kurulum.ps1` ile `requirements.txt` bağımlılıklarını kurun. Paket yeniden eğitim gerektirmez. Başka bir bilgisayar/işletim sistemi kurulumu bu oturumda denenmedi.

Uygulama: `http://127.0.0.1:8501`. **Eşik AI → Tek satırda şirket sorgula** bölümünde şu örneği çalıştırın:

> ESIK-05511 için en önemli 3 risk nedir?

Beklenen kayıt: **sıra 135/1170**, ilk %10 listesinde **Hayır**, eksik girdi **0/64**. Bu örnek hata analizindeki kaçırılan vakayı da tartışmaya uygundur. İlk %10 içinde bir kayıt göstermek için `ESIK-05817` kullanılabilir. `ESIK-99999` mevcut değildir ve açık bir “kayıt bulunamadı” yanıtı vermelidir.

“AI yorumu da ekle” seçilmezse doğrudan kayıt özeti sunulur. Seçilirse ilgili kaydın bağlamı mevcut n8n sohbet akışına gönderilir; hizmet ulaşılamazsa özet korunur ve bağlantı hatası açıkça belirtilir. Bu özet serbest soruyu dil modeliyle yanıtladığı iddiası taşımaz.

## n8n bağlantıları

Bu bilgisayarda dört akış zaten içe aktarıldı ve OpenAI bağlantısı tamamlandı. Aynı kurulumu yeniden yapmayın; aşağıdaki kurulum adımları başka bir bilgisayar veya hesap için geçerlidir. n8n: `http://127.0.0.1:5678`. İlk açılış biraz sürebilir. Arayüzde oturum açın; parolanızı veya servis anahtarlarınızı sohbet mesajına yazmayın.

Başka bir kurulumda proje klasöründen `python scripts/prepare_connections.py` çalıştırın. Ardından yerel n8n CLI varsa:

```powershell
n8n import:credentials --input=.runtime/credentials.json
n8n import:workflow --separate --input=.runtime/workflows
```

Bu komutlar dört taslağı pasif olarak içe aktarır. **Bağlantıları elle tamamladıktan sonra tekrar içe aktarmayın:** aynı kimlikli taslakları yeniler. `.runtime/credentials.json` ve `.streamlit/secrets.toml` yerel gizli değer içerir; paylaşmayın. Paylaşılabilir akış şablonları `n8n/` klasöründedir ve kullanıcı bağlantısı içermez.

### 1. Mikrofon ve AI

1. n8n'de bir **OpenAI** bağlantısı oluşturun. Anahtar yalnızca n8n bağlantı formuna girilmeli.
2. **EŞİK v3 - Finansal Risk Asistanı** akışındaki **OpenAI Chat Model** düğümüne bu bağlantıyı atayın.
3. **Eşik AI - Sesli Soruyu Yazıya Çevir** akışındaki **Türkçe Metne Çevir** düğümüne aynı OpenAI bağlantısını atayın.
4. İki akışın Webhook düğümlerindeki **EŞİK v3 - Uygulama Webhook** Header Auth bağlantısı yerel içe aktarmada hazırdır. Elle kurulumda header adı `X-ESIK-WEBHOOK-TOKEN`, değeri yerel `N8N_WEBHOOK_TOKEN` ile aynı olmalı.
5. İki akışı yayımlayın. Ardından `python scripts/prepare_connections.py --enable-ai` çalıştırıp uygulamayı ve şirket sorgu servisini yeniden başlatın. Bu adım yerel sohbet/ses URL'lerini açar; akışların çalıştığını otomatik olarak kanıtlamaz.
6. Eşik AI'da önce kısa bir yazılı soru deneyin. Sonra mikrofonla kaydedin, ses kaydını gönderin, gelen metni kontrol edip **ayrıca gönderin**. Metin kendiliğinden AI sorusu olarak gönderilmez.

Bu uygulama **kayıt tamamlandıktan sonra** yazıya çevirir; konuşurken kelime kelime canlı dikte değildir. Streamlit soru kutusunun yerleşik ses desteği kullanılır. [Streamlit belgesi](https://docs.streamlit.io/develop/api-reference/chat/st.chat_input).

Kurulu n8n 2.36.9'un yerleşik ses düğümü `whisper-1` kullanır; sohbet şablonundaki model adı `gpt-5.6-luna` olarak korunmuştur. Bu hesapta sohbet modeline erişim ve gerçek API yanıtı doğrulandı. Sohbet düğümü düşük düşünme düzeyi ve 1.536 token çıktı sınırıyla çalışır. Ses n8n üzerinden OpenAI'a iletilir ve API hesabı kullanılır. [n8n ses işlemleri](https://docs.n8n.io/integrations/builtin/app-nodes/n8n-nodes-langchain.openai/audio-operations/), [OpenAI ses rehberi](https://developers.openai.com/api/docs/guides/speech-to-text).

### 2. Telegram

1. Telegram hesabınızda **BotFather** üzerinden `/newbot` ile bot oluşturun. Bot token'ını n8n'de **Telegram API** bağlantısına kaydedin. [Resmî n8n bağlantı rehberi](https://docs.n8n.io/integrations/builtin/credentials/telegram/).
2. **EŞİK v3 - Telegram Şirket Sorgusu** akışında **Telegram Mesajı** ve **Yanıtı Gönder** düğümlerine aynı Telegram bağlantısını atayın.
3. n8n'in Telegram tarafından erişilebilir bir HTTPS webhook adresi olmalı. Mevcut `127.0.0.1` adresi dışarıdan erişilemez. Kendi n8n sunucusu veya uygun HTTPS reverse proxy kurulumu için [webhook URL yapılandırması](https://docs.n8n.io/hosting/configuration/configuration-examples/webhook-url/) kullanılmalı. Bu oturumda tünel veya herkese açık sunucu kurulmadı.
4. Kendi kullanıcı kimliğinizi Telegram Trigger testindeki `message.from.id` alanından alın. Yalnız kendi botunuza bir mesaj gönderin; bu aşamada uygulama yanıt vermeyebilir.
5. `python scripts/prepare_connections.py --telegram-id 123456789` komutunda örnek sayıyı kendi kimliğinizle değiştirin. Şirket sorgu servisini yeniden başlatın; akışları yeniden içe aktarmayın.
6. Akışı yayımlayın, botla özel konuşmada örnek şirket sorusunu gönderin ve doğru şirkete ait yanıtı görün. Aynı bot için test/production webhook'u çakışabilir; son kontrolü yayımlanmış akışta yapın. [Telegram Trigger](https://docs.n8n.io/integrations/builtin/trigger-nodes/n8n-nodes-base.telegramtrigger/).

Grup mesajları, bot mesajları ve ses/görsel ekleri bu Telegram taslağında işlenmez. Telefon üzerinden sesli mesaj transkripsiyonu bu sürüme dahil değildir; mikrofon özelliği web uygulamasındadır.

### 3. WhatsApp

1. Meta developer uygulaması ve WhatsApp Business Cloud varlıklarını hazırlayın. Test numarası kullanılıyorsa Meta'nın izin verdiği test alıcılarını ayarlayın.
2. n8n'de **WhatsApp Trigger API** bağlantısına uygulamanın Client ID/Client Secret bilgilerini; **WhatsApp API** bağlantısına Access Token/Business Account ID bilgilerini girin. [n8n WhatsApp bağlantı rehberi](https://docs.n8n.io/integrations/builtin/credentials/whatsapp/).
3. **EŞİK v3 - WhatsApp Şirket Sorgusu** akışındaki **WhatsApp Mesajı** düğümüne trigger bağlantısını, **Yanıtı Gönder** düğümüne mesaj gönderme bağlantısını atayın. Yerleşik trigger webhook doğrulamasını yönetir; rastgele bir verify token uydurmayın.
4. Telegram için olduğu gibi n8n'in HTTPS webhook adresi dışarıdan erişilebilir olmalı. Aynı Meta uygulamasında tek WhatsApp webhook'u bulunabilir; başka bir akış varsa çakışmayı kontrol edin. [WhatsApp Trigger](https://docs.n8n.io/integrations/builtin/trigger-nodes/n8n-nodes-base.whatsapptrigger/).
5. `python scripts/prepare_connections.py --whatsapp-id 905551234567` komutundaki örneği kendi numaranızla değiştirin; ülke kodu kullanın. Şirket sorgu servisini yeniden başlatıp yayımlanmış akışa gelen metin mesajını deneyin.

Akış gelen mesaja yanıt verir; toplu gönderim veya kendiliğinden bildirim yapmaz. Teslim/okundu durumları ve metin dışındaki ekler sorgu başlatmaz. Hesap izinleri ve API kısıtları nedeniyle canlı WhatsApp demosunun hazır olduğu henüz söylenemez.

## Mimari ve doğrulama

```mermaid
flowchart LR
    T[Telegram metni] --> N[n8n yerleşik trigger]
    W[WhatsApp metni] --> N
    N --> Q[Kimlik doğrulamalı yerel şirket servisi]
    Q --> D[Kaydedilmiş skor ve SHAP]
    Q --> A[İsteğe bağlı n8n AI yorumu]
    D --> R[Aynı konuşmaya yanıt]
    A --> R
```

Servis yalnız `127.0.0.1:8765` üzerinde dinler. `X-ESIK-QUERY-TOKEN` gerekir; Telegram/WhatsApp izin listesi boşsa yanıt gönderilmez. Kullanıcı sorusu içindeki tek şirket kodu kesin eşleşmeyle seçilir; birden fazla şirket kodu veya bilinmeyen kod açıklama ister. Gerçek test etiketi AI bağlamına alınmaz. Telefon/kullanıcı kimliği LLM oturum kimliğine düz olarak aktarılmaz. İstekler arasında önceki şirketin sohbet geçmişi taşınmaz.

**n8n ve Python servisi aynı bilgisayarda çalışacak şekilde yapılandırıldı.** n8n Cloud veya Docker'a taşınırsa HTTP Request düğümünün `127.0.0.1` adresi o ortamı işaret eder; servis erişimi ayrıca düzenlenmeden bu şablon çalışmaz. Mevcut yerel servisi doğrudan internete açmayın.

73 Python testi ve 5 JavaScript testi geçti. Sekiz sayfa AppTest ile açıldı; tek satırlık sorgu gerçek tarayıcıda da denendi. Kontroller: doğru kayıt/SHAP bağlamı, bilinmeyen/çoklu kod, HTTP kimlik doğrulama, izin listesi, hatalı istekler, AI kesintisi, şirketler arası bağlam ayrımı, ses taslağı, Telegram metin kaçışları ve WhatsApp çoklu mesajları. Altı eğitim protokolü testi için eksik Optuna bağımlılıkları kullanıcı arşivinden ayrı test klasörüne alındı. Ana model yeniden eğitilmedi.

**Test kapsamı sınırı:** Yerel testler, gerçek OpenAI sohbet yanıtı ve sentetik sesin transkripsiyon servisine ulaşması doğrulandı. Kullanıcı kendi gerçek mikrofon denemesinin çalıştığını bildirdi. Bağımsız bir Türkçe konuşma doğruluğu ölçümü yapılmadı. Canlı Telegram/WhatsApp teslimi ve harici HTTPS erişimi henüz doğrulanmadı. Bu bir sunum prototipidir; kuyruk, kalıcı mesaj tekrarını önleme, yük testi ve üretim işletimi ayrıca gerekir.

## 15 dakikalık sunumda kullanımı

Yeni özelliklere **45–60 saniye** ayırın: bir şirket koduyla özeti açın, veri kaynağını gösterin; hesap bağlantıları önceden denenmişse tek bir mikrofon veya Telegram örneği ekleyin. Model seçimi, dengesiz sınıf, veri sızıntısını önleme, kapasite metriği, hata analizi ve SHAP anlatımını koruyun.

Sunum cümlesi: “EŞİK yalnızca risk sıralaması üretmiyor; analistin aynı kaydı uygulamada veya mesajla sorgulayabileceği bir erişim katmanı da sunuyor. Burada anlık olan yanıt süresi; veri setimiz tarihsel.”

Gerçek şirket adı yazarak güncel Türkiye şirket verisini getirme özelliği bu pakette yoktur. PCA/özellik mühendisliği deneyleri ayrı raporlardadır; sırf gösteriş için ana modele eklenmedi. Henüz bağlanmamış bir kanalı canlı çalışan özellik olarak sunmayın.
