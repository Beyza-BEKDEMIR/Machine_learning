import os
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras import regularizers
import matplotlib.pyplot as plt

# Verilerin yüklenmesi
mat = loadmat('imagelabels.mat')
labels = mat['labels'][0]

image_dir = '102flowers/jpg'
image_files = sorted(os.listdir(image_dir))

images = []
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path)
    image = image.resize((64, 64))  # Resimleri 64x64 piksele yeniden boyutlandırma
    image = np.array(image).reshape((64, 64, 3)) 
    images.append(image)

X = np.array(images)
y = np.array(labels)

# Verilerin eğitim ve test setlerine ayrılması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Etiketlerin kodlanması
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# Veri artırma için ImageDataGenerator oluşturma
datagen = ImageDataGenerator(
    rotation_range=45,  # Rastgele döndürme
    width_shift_range=0.2,  # Genişlik kaydırma
    height_shift_range=0.2,  # Yükseklik kaydırma
    #shear_range=0.2,  # Kayma dönüşümü
    zoom_range=0.2,  # Rastgele yakınlaştırma
    #horizontal_flip=True,  # Yatay simetri ekleme
    fill_mode='nearest'  # Doldurma modu
)

# Modelin oluşturulması
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.001)),  # L2 regularizasyonu uygulama
    Dense(102, activation='softmax') #102 sınıf var
])

# Modelin derlenmesi
model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modelin eğitimi
model.fit(datagen.flow(X_train, y_train_encoded, batch_size=64), epochs=100, validation_data=(X_test, y_test_encoded))

# Modelin değerlendirilmesi
loss, accuracy = model.evaluate(X_test, y_test_encoded)
print(f"Kayıp: {loss}, Doğruluk: {accuracy}")

# Test veri setinden tahminler
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Etiketleri geri dönüştürme
y_test_decoded = encoder.inverse_transform(y_test_encoded)
predicted_labels_decoded = encoder.inverse_transform(predicted_labels)

# Plot için örnek görüntüler ve tahminler
num_samples = 10
sample_indexes = np.random.choice(len(X_test), num_samples, replace=False)
sample_images = X_test[sample_indexes]
sample_true_labels = y_test_decoded[sample_indexes]
sample_predicted_labels = predicted_labels_decoded[sample_indexes]

# Plotting
plt.figure(figsize=(15, 5))
for i in range(num_samples):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_images[i])
    true_label = sample_true_labels[i]
    predicted_label = sample_predicted_labels[i]
    plt.title(f'True: {true_label}\nPred: {predicted_label}')
    plt.axis('off')

plt.tight_layout()
plt.show()