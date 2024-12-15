import os
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from PIL import Image
import random
import matplotlib.pyplot as plt

print("Etiket dosyası yükleniyor...")
# Etiket dosyasını yükleme
mat = loadmat('imagelabels.mat')
labels = mat['labels'][0]  # Etiketlerin array'e dönüştürülmesi

print("Resim dosyaları yükleniyor ve işleniyor...")
# Resim dosyalarının bulunduğu dizin
image_dir = '102flowers/jpg'  # Resim dosyalarının bulunduğu dizin

# Resim dosyalarını yükleme ve işleme
image_files = sorted(os.listdir(image_dir))  # Resim dosyalarını sıralama

images = []
for idx, image_file in enumerate(image_files):
    if idx % 1000 == 0:
        print(f"{idx} resim işlendi...")
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path)
    image = image.resize((64, 64))  # Resimleri 64x64 piksele yeniden boyutlandırma
    image = np.array(image).flatten()  # Resmi düzleştirme
    images.append(image)

print("Veriler numpy arraylerine dönüştürülüyor...")
# Numpy arraylerine dönüştürme
X = np.array(images)
y = np.array(labels)

# Veri setinin boyutunu yazdırma
print(f"Orijinal veri kümesinin boyutu: {X.shape}")

print("Veriler eğitim ve test setlerine ayrılıyor...")
# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Veriler normalize ediliyor...")
# Veriyi standartlaştırma
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("PCA ile özellik sayısı azaltılıyor...")
# PCA ile boyut indirgeme
pca = PCA(n_components=5000)  # Özellik sayısını 5000'e indirgeme
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("Veri boyutu azaltılıyor...")
# Yeni veri kümesi için örnek sayısı
sample_size = 5000  # Daha küçük bir örnekleme boyutu

# Veri kümesinden rastgele örneklem alma
random.seed(42)  # Tekrarlanabilirlik için seed ayarı
sampled_indices = random.sample(range(len(X_train_pca)), sample_size)
X_train_sampled = X_train_pca[sampled_indices]
y_train_sampled = y_train[sampled_indices]

print("Yeni veri kümesinin boyutu:", X_train_sampled.shape)

print("Lojistik regresyon modeli eğitiliyor...")
# Lojistik regresyon modeli ile sınıflandırma
model = LogisticRegression(solver="lbfgs", C=0.1, multi_class='ovr', random_state=0, max_iter=1000)
model.fit(X_train_sampled, y_train_sampled)

print("Tahmin yapılıyor ve doğruluk hesaplanıyor...")
# Tahmin yapma ve doğruluk hesaplama
y_pred = model.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk: {accuracy}")

# Örnek üzerinde tahmin yapma ve sonucu görme
def predict_example(index):
    # Test verisindeki örnek
    example = X_test[index].reshape(1, -1)
    # PCA dönüşümünü uygulama
    example_pca = pca.transform(example)
    # Tahmin yapma
    prediction = model.predict(example_pca)
    return prediction[0]

# Örneğin tahminini yazdırma
test_index = 0  # Test setindeki 0. örneği seç
predicted_label = predict_example(test_index)
true_label = y_test[test_index]

print(f"Test örneği index: {test_index}")
print(f"Gerçek etiket: {true_label}")
print(f"Tahmin edilen etiket: {predicted_label}")

# Görselleştirme
def plot_example(index):
    # Orijinal resmi geri yükleme
    image = scaler.inverse_transform(X_test[index].reshape(1, -1)).reshape(64, 64, 3)
    # Verilerin 0-255 aralığında olmasını sağlama
    image = np.clip(image, 0, 255).astype(np.uint8)

    plt.imshow(image)
    plt.title(f"Gerçek: {true_label}, Tahmin: {predicted_label}")
    plt.axis('off')
    plt.show()

plot_example(test_index)

