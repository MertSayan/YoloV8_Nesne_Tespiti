# Diş Fırçası ve Diş Macunu Nesne Tespiti  
**YOLOv8 (CNN) + PyQt5 Masaüstü Uygulaması**

🖼 **Proje Önizlemesi**  
(PyQt5 tabanlı masaüstü arayüz üzerinden görüntü yükleme ve nesne tespiti)

<img width="1918" height="995" alt="image" src="https://github.com/user-attachments/assets/164756f3-d642-4881-88ca-713f30b0e772" />


## 🎯 Projenin Amacı
Bu projede, derin öğrenme tabanlı nesne tespiti algoritmalarından **YOLOv8** kullanılarak, gerçek görüntüler üzerinde **diş fırçası** ve **diş macunu** nesnelerinin tespit edilmesi amaçlanmıştır.

Proje kapsamında:

- İki sınıflı (diş fırçası – diş macunu) özel bir görüntü veri seti oluşturulmuştur  
- Görüntüler YOLO formatında etiketlenmiştir  
- YOLOv8 modeli Google Colab ortamında eğitilmiştir  
- Eğitilen model, PyQt5 tabanlı bir masaüstü uygulamasına entegre edilmiştir  
- Kullanıcı, arayüz üzerinden görüntü seçerek modeli test edebilmektedir  

Bu sayede, uçtan uca bir **nesne tespiti + masaüstü uygulama entegrasyonu** gerçekleştirilmiştir.

---

## 1️⃣ Veri Seti Hazırlığı

### 🔹 Sınıflar
Bu projede iki adet sınıf bulunmaktadır:

- dis_fircasi  
- dis_macunu  

### 🔹 Veri Seti Özellikleri
- Görüntüler tarafımca toplanmıştır  
- Dosya formatı: `.jpg` / `.png`  
- Etiketleme işlemi **LabelImg** aracı kullanılarak yapılmıştır  
- YOLO formatında `.txt` etiket dosyaları oluşturulmuştur  

### 🔹 Sınıf İndeksleri
- **0 → dis_fircasi**  
- **1 → dis_macunu**

### 🔹 Veri Bölünmesi
Veri seti aşağıdaki şekilde ayrılmıştır:

- **Train (Eğitim)**
- **Val (Doğrulama)**

Bu ayrım, modelin genelleme başarısını ölçmek amacıyla yapılmıştır.

---

## 2️⃣ YOLO Formatı ve YAML Dosyası
Model eğitimi için `data.yaml` dosyası oluşturulmuştur.

Bu dosyada:

- Eğitim ve doğrulama veri yolları  
- Sınıf sayısı (`nc`)  
- Sınıf isimleri (`names`)  

tanımlanmıştır.

Bu yapı, YOLOv8 modelinin veri setini doğru şekilde okuyabilmesi için zorunludur.

---

## 3️⃣ Model Eğitimi (YOLOv8)

### 🔹 Kullanılan Model
- **Model:** YOLOv8n (Nano)  
- **Framework:** Ultralytics YOLOv8  
- **Eğitim Ortamı:** Google Colab (GPU)

### 🔹 Eğitim Parametreleri
- Epoch: 50  
- Görüntü boyutu: 640 × 640  
- Batch size: 8  

### 🔹 Eğitim Kodu
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="/content/drive/MyDrive/dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device=0,
    name="dis_fircasi_dis_macunu_yolo"
)
```

Grafikler:
<img width="1635" height="816" alt="image" src="https://github.com/user-attachments/assets/005ffddb-29fb-49a0-ac91-03890c2bf979" />


🔹 Eğitim Sonuçları

Eğitim süresince loss değerlerinde düzenli düşüş gözlemlenmiştir

Model, diş fırçası ve diş macunu nesnelerini başarılı şekilde tespit edebilmiştir

En iyi performansa sahip model ağırlıkları best.pt dosyası olarak kaydedilmiştir

---

4️⃣ PyQt5 Masaüstü Uygulaması

Eğitilen YOLOv8 modeli, PyQt5 kullanılarak geliştirilen bir masaüstü uygulamasına entegre edilmiştir.

🔹 Uygulama Özellikleri

Görüntü yükleme

YOLOv8 ile nesne tespiti

Bounding box çizimi

Tespit edilen nesnelerin liste halinde gösterimi

Sonuç görüntüsünü kaydetme

Modern ve kullanıcı dostu arayüz

🔹 Kullanıcı Akışı

Kullanıcı Select Image butonu ile görüntüyü seçer

Test Image butonuna basılır

Model görüntüyü analiz eder

Tespit edilen nesneler bounding box ile işaretlenir

Sonuçlar arayüzde listelenir ve istenirse kaydedilir


---

📁 Proje Dosya Yapısı
YOLO_GUI/
├── gui_app.py
├── best.pt
├── README.md

---

▶️ Uygulamayı Çalıştırma

Uygulama terminale yazılacak olan aşağıdaki komut ile çalıştırılır:

python gui_app.py

---

🛠️ Kullanılan Teknolojiler

Python 3.10

YOLOv8 (Ultralytics)

PyTorch

OpenCV

PyQt5

Google Colab (GPU)

---

📊 Genel Değerlendirme

Bu projede, CNN tabanlı YOLOv8 algoritması kullanılarak iki sınıflı bir nesne tespit sistemi geliştirilmiştir.
Eğitilen modelin PyQt5 tabanlı bir masaüstü arayüzü ile sunulması, projenin uygulama odaklı, kullanıcı dostu ve gerçek hayat senaryolarına uygun olmasını sağlamıştır.

---

👤 Geliştirici

Muhammed Mert Sayan
Okul No : 2212721028
