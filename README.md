<img src="https://github.com/enricochandran/Hands-on-Machine-Learning-with-Scikit-Learn-Keras-TensorFlow/blob/ebe6a3b60edc8129acbe1144bec6b71711f00cb7/Hands-on%20Machine%20Learning%20with%20Scikit-Learn%2C%20Keras%20%26%20TensorFlow%20Cover.jpg" width="250" alt="Cover">

## Bagian I: Fondasi Machine Learning

Bagian pertama buku ini membangun dasar yang kokoh dalam Machine Learning, menjelaskan prinsip-prinsip inti dan berbagai jenis algoritma.

### Bab 1: Gambaran Umum Machine Learning

Bab ini menyajikan pengenalan menyeluruh tentang Machine Learning (ML), menjelaskan definisi, signifikansi, dan taksonomi sistem ML. Pembaca akan memahami beragam jenis ML berdasarkan:

* **Tingkat Supervisi**: Membedakan antara **Supervised Learning** (dengan label data), **Unsupervised Learning** (tanpa label data), **Semi-supervised Learning** (kombinasi keduanya), dan **Reinforcement Learning** (belajar melalui interaksi lingkungan).
* **Metode Pembelajaran**: Kontras antara **Batch Learning** (model dilatih sekali secara *offline*) dan **Online Learning** (model diperbarui secara inkremental saat data baru tiba).
* **Pendekatan Generalisasi**: Perbandingan antara **Instance-based Learning** (memprediksi berdasarkan kesamaan dengan contoh yang sudah ada) dan **Model-based Learning** (membangun model dari data untuk membuat prediksi).

Selain itu, bab ini juga menguraikan tantangan krusial dalam ML, seperti pentingnya **kualitas dan kuantitas data**, serta isu umum **overfitting** (model terlalu kompleks untuk data pelatihan, gagal bergeneralisasi) dan **underfitting** (model terlalu sederhana, tidak menangkap pola data).

---

### Bab 2: Proyek Machine Learning dari Awal hingga Akhir

Bab ini memberikan panduan praktis melalui studi kasus proyek ML lengkap menggunakan *dataset* harga rumah California. Langkah-langkah esensial yang dibahas meliputi:

1.  **Akuisisi Data**: Proses mengunduh dan memuat *dataset* awal.
2.  **Pembentukan *Test Set***: Penekanan pada pembuatan *test set* yang representatif, seringkali melalui **stratified sampling**, untuk menghindari **data snooping bias** yang dapat mengarah pada penilaian performa model yang terlalu optimis.
3.  **Eksplorasi dan Visualisasi Data**: Analisis awal data untuk menemukan korelasi, distribusi, dan pola visual yang membantu dalam pemahaman *dataset*.
4.  **Persiapan Data (Preprocessing)**: Tahap krusial yang mencakup pembersihan data, penanganan nilai yang hilang (imputasi), transformasi fitur kategorikal, dan **feature scaling** (normalisasi atau standardisasi) untuk memastikan model bekerja optimal. Penggunaan `Pipeline` dan `ColumnTransformer` dari Scikit-Learn dijelaskan untuk merampingkan proses ini.
5.  **Pelatihan Model**: Melatih berbagai algoritma dasar seperti Regresi Linier, *Decision Tree*, dan *Random Forest* untuk mendapatkan *baseline* kinerja.
6.  **Penyempurnaan Model (*Fine-Tuning*)**: Mengoptimalkan **hyperparameter** model terbaik menggunakan teknik seperti **GridSearchCV** untuk menemukan kombinasi parameter yang menghasilkan kinerja optimal.
7.  **Evaluasi Akhir**: Menguji model yang telah disetel pada *test set* yang belum pernah dilihat sebelumnya untuk mendapatkan estimasi kinerja yang tidak bias.

---

### Bab 3: Klasifikasi

Bab ini mendalami tugas klasifikasi, dengan fokus pada *dataset* MNIST (angka tulisan tangan). Penekanan diberikan pada pentingnya metrik kinerja yang tepat, karena akurasi saja seringkali tidak memadai.

* **Metrik Kinerja Utama**:
    * **Confusion Matrix**: Ringkasan visual kinerja model klasifikasi, menunjukkan *True Positives*, *True Negatives*, *False Positives*, dan *False Negatives*.
    * **Precision**: Proporsi prediksi positif yang benar ($TP / (TP + FP)$).
    * **Recall (Sensitivitas)**: Proporsi instance positif yang terdeteksi dengan benar ($TP / (TP + FN)$).
    * **F1-Score**: Rata-rata harmonik dari *precision* dan *recall*, berguna ketika ada *trade-off* antara keduanya.
    * **Kurva Precision-Recall**: Memvisualisasikan *trade-off* antara *precision* dan *recall* pada berbagai *threshold*.
* **Kurva ROC (Receiver Operating Characteristic)**: Alat grafis untuk mengevaluasi kinerja pengklasifikasi biner dengan memplot *True Positive Rate* (Recall) terhadap *False Positive Rate* ($FP / (FP + TN)$) pada berbagai *threshold*. Area di bawah kurva ROC (AUC ROC) sering digunakan sebagai metrik ringkasan.
* **Klasifikasi Multikelas**: Strategi untuk menangani masalah klasifikasi dengan lebih dari dua kelas, termasuk **One-vs-Rest (OvR)** atau **One-vs-All (OvA)**, di mana satu pengklasifikasi biner dilatih untuk setiap kelas, dan **One-vs-One (OvO)**, yang melatih pengklasifikasi biner untuk setiap pasangan kelas.
* **Analisis Error**: Menganalisis *confusion matrix* secara mendalam untuk mengidentifikasi jenis kesalahan yang paling sering dilakukan model, membantu dalam strategi perbaikan.

---

### Bab 4: Pelatihan Model

Bab ini menyelami mekanisme internal pelatihan model linier, memperkenalkan konsep optimisasi yang fundamental.

* **Regresi Linier**: Dibahas melalui dua pendekatan utama:
    * **Normal Equation**: Solusi analitis langsung untuk menemukan parameter model yang meminimalkan *Mean Squared Error*.
    * **Gradient Descent**: Algoritma optimisasi iteratif yang secara bertahap menyesuaikan parameter model ke arah penurunan gradien fungsi biaya.
* **Jenis-jenis Gradient Descent**:
    * **Batch Gradient Descent**: Menghitung gradien menggunakan seluruh *dataset* pada setiap langkah.
    * **Stochastic Gradient Descent (SGD)**: Menghitung gradien menggunakan hanya satu instance data yang dipilih secara acak pada setiap langkah, memungkinkan pembaruan lebih cepat tetapi dengan lebih banyak fluktuasi.
    * **Mini-batch Gradient Descent**: Kompromi antara Batch dan SGD, menggunakan subset kecil dari data pada setiap langkah.
* **Regresi Polinomial**: Memperluas Regresi Linier untuk menangani data non-linier dengan menambahkan pangkat fitur sebagai fitur baru.
* **Regularisasi**: Teknik untuk mencegah *overfitting* pada model linier dengan menambahkan penalti pada ukuran parameter model:
    * **Ridge Regression**: Menambahkan penalti $L_2$ (kuadrat magnitude koefisien).
    * **Lasso Regression**: Menambahkan penalti $L_1$ (nilai absolut koefisien), yang juga dapat menghasilkan pemilihan fitur (beberapa koefisien menjadi nol).
    * **Elastic Net**: Kombinasi penalti Ridge dan Lasso.
* **Regresi Logistik & Softmax**: Model linier yang disesuaikan untuk tugas klasifikasi, masing-masing untuk klasifikasi biner dan multikelas, dengan mengaplikasikan fungsi *sigmoid* atau *softmax* ke output linier.

---

### Bab 5: Support Vector Machines (SVM)

Bab ini memperkenalkan *Support Vector Machines* (SVM), algoritma klasifikasi dan regresi yang kuat, bertujuan untuk menemukan *hyperplane* dengan *margin* terbesar yang memisahkan kelas.

* **Klasifikasi Margin Besar**: Konsep inti SVM adalah menemukan *hyperplane* yang memaksimalkan jarak ke titik data terdekat dari setiap kelas (*support vectors*).
* **Hard vs. Soft Margin**:
    * **Hard Margin Classification**: Membutuhkan semua instance berada di luar *margin* dan di sisi yang benar dari *hyperplane* (rentan terhadap *outlier*).
    * **Soft Margin Classification**: Memungkinkan beberapa pelanggaran *margin* atau *misclassification* untuk meningkatkan robustitas, dikendalikan oleh **hyperparameter `C`** (semakin rendah C, semakin luas margin, semakin banyak pelanggaran diizinkan).
* **Kernel Trick**: Teknik revolusioner yang memungkinkan SVM menangani data non-linier secara efisien. Dengan menggunakan **kernel function** (seperti **Polynomial Kernel** atau **Radial Basis Function (RBF) Kernel**), SVM secara implisit memetakan data ke ruang dimensi yang lebih tinggi di mana pemisahan linier mungkin, tanpa secara eksplisit melakukan transformasi tersebut, sehingga menghindari biaya komputasi yang tinggi.
* **Regresi SVM**: Mengadaptasi prinsip SVM untuk tugas regresi dengan mencari *hyperplane* yang dapat memasukkan sebanyak mungkin instance ke dalam *margin* tertentu, meminimalkan jumlah *support vectors* di luar *margin*.

---

### Bab 6: Decision Trees

Bab ini membahas *Decision Tree*, model ML yang intuitif dan mudah diinterpretasikan (*white-box model*) karena membuat prediksi berdasarkan serangkaian aturan if/else sederhana.

* **Pelatihan dan Visualisasi**: Decision Tree dapat dilatih dengan cepat dan struktur pohonnya dapat divisualisasikan dengan jelas, memungkinkan pemahaman langsung tentang proses pengambilan keputusan model.
* **Algoritma CART (Classification and Regression Tree)**: Algoritma *greedy* yang digunakan oleh Scikit-Learn untuk membangun *Decision Tree*. Ia membagi data secara rekursif menjadi dua *subset* pada setiap *node* berdasarkan fitur yang menghasilkan kemurnian tertinggi (untuk klasifikasi) atau penurunan *MSE* terbesar (untuk regresi).
* **Regularisasi**: Untuk mencegah *overfitting* yang ekstrem pada *Decision Tree*, batasan diterapkan selama pelatihan menggunakan **hyperparameter** seperti `max_depth` (kedalaman maksimum pohon), `min_samples_split` (jumlah minimum sampel untuk memecah *node*), atau `min_samples_leaf` (jumlah minimum sampel di *leaf node*).
* **Kelemahan**: *Decision Tree* sangat sensitif terhadap variasi kecil dalam data pelatihan (*high variance*), yang dapat menyebabkan perubahan signifikan pada struktur pohon. Ini menjadi motivasi kuat untuk menggunakan teknik *Ensemble Learning* seperti *Random Forest*.

---

### Bab 7: Ensemble Learning dan Random Forests

Bab ini menjelaskan bagaimana menggabungkan kekuatan beberapa model individu (*estimators*) untuk menghasilkan prediksi yang lebih akurat dan robust daripada model tunggal. Ini adalah konsep inti dari **Ensemble Learning**.

* **Voting Classifiers**: Menggabungkan prediksi dari beberapa pengklasifikasi yang berbeda. Ini bisa berupa **Hard Voting** (memilih kelas mayoritas) atau **Soft Voting** (menjumlahkan probabilitas kelas dan memilih yang tertinggi).
* **Bagging (Bootstrap Aggregating) dan Pasting**:
    * **Bagging**: Melatih beberapa model yang sama pada *subset* data pelatihan yang berbeda, yang diambil dengan penggantian (bootstrap). Hasilnya kemudian diagregasi (rata-rata untuk regresi, mayoritas untuk klasifikasi).
    * **Pasting**: Mirip dengan Bagging, tetapi *subset* data diambil tanpa penggantian.
    * **Random Forest**: Implementasi khusus dari Bagging yang sangat populer untuk *Decision Tree*. Tidak hanya mengambil sampel data, tetapi juga mengambil sampel fitur pada setiap pemisahan *node*, lebih lanjut mengurangi korelasi antar pohon dan meningkatkan *robustness*.
* **Boosting**: Teknik di mana *estimators* dilatih secara sekuensial, dan setiap *estimator* baru mencoba memperbaiki kesalahan yang dibuat oleh *estimator* sebelumnya. Contoh populer meliputi:
    * **AdaBoost (Adaptive Boosting)**: Memfokuskan pada instance yang salah diklasifikasi pada iterasi sebelumnya dengan memberikan bobot lebih tinggi.
    * **Gradient Boosting**: Membangun *estimators* secara sekuensial untuk memprediksi residual (*error*) dari *ensemble* sebelumnya.
* **Stacking (Stacked Generalization)**: Metode *ensemble* yang lebih kompleks di mana sebuah model (*blender* atau *meta-learner*) dilatih untuk mengagregasi prediksi dari beberapa model dasar (*base estimators*). Model *blender* belajar cara terbaik untuk menggabungkan output dari model-model dasar.

---

### Bab 8: Reduksi Dimensi

Bab ini membahas fenomena "kutukan dimensi" (*curse of dimensionality*), di mana kinerja algoritma ML menurun drastis ketika jumlah fitur (*dimensi*) meningkat, dan memperkenalkan berbagai teknik untuk menguranginya.

* **Pendekatan Utama**:
    * **Proyeksi**: Memetakan data dari ruang dimensi tinggi ke ruang dimensi rendah dengan mempertahankan informasi penting (misalnya, varians).
    * **Manifold Learning**: Mengasumsikan bahwa data dimensi tinggi terletak pada *manifold* dimensi rendah yang tersembunyi, dan mencoba menemukan *manifold* tersebut.
* **PCA (Principal Component Analysis)**: Teknik proyeksi linier yang paling populer. PCA mengidentifikasi **Principal Components** (sumbu-sumbu ortogonal) yang menjelaskan varians terbesar dalam data, kemudian memproyeksikan data ke sumbu-sumbu ini untuk mengurangi dimensi.
* **Kernel PCA**: Ekstensi non-linier dari PCA yang menggunakan **kernel trick** (mirip dengan SVM) untuk melakukan proyeksi non-linier, memungkinkan deteksi struktur non-linier dalam data.
* **LLE (Locally Linear Embedding)**: Teknik *manifold learning* non-linier yang bertujuan untuk mempertahankan hubungan linier lokal antar instance. LLE bekerja dengan merekonstruksi setiap titik data sebagai kombinasi linier dari tetangga terdekatnya, dan kemudian mencari representasi dimensi rendah yang mempertahankan bobot rekonstruksi ini.

---

### Bab 9: Teknik Pembelajaran Tanpa Supervisi

Bab ini berfokus pada algoritma *unsupervised learning* yang belajar pola dan struktur dari data yang tidak memiliki label (*unlabeled data*).

* **Clustering**: Tugas mengelompokkan instance serupa ke dalam *cluster* atau kelompok.
    * **K-Means**: Algoritma *clustering* berbasis *centroid* yang relatif cepat. Ia bekerja dengan secara iteratif menetapkan titik data ke *cluster* terdekat dan kemudian memperbarui *centroid* *cluster* tersebut. Batasannya termasuk kebutuhan untuk menentukan jumlah *cluster* (`k`) di awal dan sensitivitas terhadap bentuk *cluster* non-globular.
    * **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Algoritma *clustering* berbasis kepadatan yang dapat menemukan *cluster* dengan bentuk arbitrer dan mengidentifikasi *noise* atau *outlier*. Ia bekerja dengan mengidentifikasi "titik inti" (memiliki tetangga yang cukup padat) dan membangun *cluster* di sekitarnya.
* **Gaussian Mixture Models (GMM)**: Model probabilistik yang mengasumsikan bahwa data dihasilkan dari campuran beberapa distribusi Gaussian (normal). GMM dapat digunakan untuk *clustering* (dengan menetapkan setiap instance ke komponen Gaussian yang paling mungkin), **density estimation** (memperkirakan fungsi kerapatan probabilitas data), dan **anomaly detection** (mengidentifikasi instance dengan probabilitas rendah di bawah model).

---

## Bagian II: Jaringan Saraf dan Deep Learning

Bagian kedua buku ini beralih ke dunia *Neural Networks* dan *Deep Learning* yang revolusioner, mencakup arsitektur modern dan praktik implementasi menggunakan Keras dan TensorFlow.

### Bab 10: Pengantar Jaringan Saraf Buatan dengan Keras

Bab ini berfungsi sebagai gerbang ke dunia *deep learning*, memperkenalkan konsep dasar jaringan saraf buatan dan bagaimana membangunnya dengan Keras.

* **MLP (Multilayer Perceptron)**: Arsitektur jaringan saraf dasar yang terdiri dari lapisan *input*, satu atau lebih **hidden layers**, dan lapisan *output*. Setiap *neuron* di lapisan terhubung ke semua *neuron* di lapisan berikutnya.
* **Backpropagation**: Algoritma pelatihan fundamental untuk jaringan saraf. Ia menghitung gradien fungsi *loss* terhadap bobot jaringan menggunakan aturan rantai, kemudian menggunakan gradien tersebut untuk memperbarui bobot melalui **Gradient Descent** atau turunannya.
* **Keras API**: Memperkenalkan API tingkat tinggi TensorFlow yang memungkinkan pembangunan dan pelatihan jaringan saraf dengan mudah.
    * **Sequential API**: Ideal untuk membangun model sebagai tumpukan lapisan sederhana, di mana setiap lapisan memiliki tepat satu *input tensor* dan satu *output tensor*.
    * **Functional API**: Memberikan fleksibilitas lebih besar untuk membangun arsitektur model yang lebih kompleks, seperti model dengan banyak *input*, banyak *output*, atau lapisan yang berbagi bobot.
* **Praktik Implementasi**: Panduan praktis tentang cara membangun pengklasifikasi dan peramal regresi, menyimpan model terlatih, memuatnya kembali, dan menggunakan **callbacks** (fungsi yang dijalankan pada titik-titik tertentu selama pelatihan, misalnya untuk *early stopping* atau menyimpan *checkpoint* model).

---

### Bab 11: Pelatihan Jaringan Saraf Dalam

Bab ini membahas tantangan spesifik yang muncul saat melatih jaringan saraf yang sangat dalam dan menyediakan solusi efektif.

* **Masalah Gradien**: Mengatasi fenomena **vanishing gradients** (gradien menjadi sangat kecil, menyebabkan pembaruan bobot lambat di lapisan awal) dan **exploding gradients** (gradien menjadi sangat besar, menyebabkan pembaruan bobot yang tidak stabil). Solusi meliputi:
    * **Inisialisasi Bobot yang Lebih Baik**: Strategi seperti **He initialization** (untuk aktivasi ReLU) dan **Glorot (Xavier) initialization** (untuk aktivasi *sigmoid* atau *tanh*) membantu menjaga skala varians aktivasi dan gradien.
    * **Fungsi Aktivasi Non-Saturasi**: Penggunaan fungsi aktivasi seperti **ReLU (Rectified Linear Unit)**, **ELU (Exponential Linear Unit)**, dan **SELU (Scaled Exponential Linear Unit)** yang tidak mengalami masalah *saturasi* gradien seperti *sigmoid* atau *tanh* pada input besar.
    * **Batch Normalization**: Teknik yang menormalkan *input* ke setiap lapisan dalam *mini-batch*, mengurangi masalah *internal covariate shift* dan mempercepat konvergensi.
    * **Gradient Clipping**: Membatasi gradien agar tidak melebihi nilai ambang tertentu, efektif dalam mengatasi *exploding gradients*.
* **Optimizer Lanjutan**: Penggunaan algoritma optimisasi yang lebih canggih dan cepat daripada SGD murni:
    * **Adam (Adaptive Moment Estimation)**: Menggabungkan keunggulan AdaGrad dan RMSProp, menggunakan rata-rata bergerak dari gradien dan kuadrat gradien.
    * **Nadam (Nesterov-accelerated Adaptive Moment Estimation)**: Versi Adam dengan momentum Nesterov.
    * **RMSProp (Root Mean Square Propagation)**: Menggunakan rata-rata bergerak kuadrat gradien untuk menyesuaikan *learning rate* secara adaptif untuk setiap parameter.
* **Regularisasi**: Teknik untuk mencegah *overfitting* pada *deep neural networks*:
    * **Dropout**: Secara acak menonaktifkan sejumlah *neuron* selama pelatihan, memaksa jaringan untuk belajar fitur yang lebih robust dan tidak terlalu bergantung pada *neuron* tertentu.
    * **Max-Norm Regularization**: Membatasi norma bobot koneksi yang masuk ke *neuron* untuk mencegah bobot menjadi terlalu besar.
* **Transfer Learning**: Strategi yang sangat efektif di mana lapisan-lapisan dari model yang sudah dilatih sebelumnya pada tugas serupa (misalnya, model yang dilatih pada *ImageNet*) digunakan kembali sebagai titik awal untuk tugas baru, menghemat waktu pelatihan dan sumber daya komputasi.

---

### Bab 12: Model Kustom dan Pelatihan dengan TensorFlow

Bab ini menyelami API tingkat rendah TensorFlow, memberikan fleksibilitas maksimum untuk membangun komponen *deep learning* kustom dan mengontrol proses pelatihan secara mendalam.

* **Tensor & Variable**: Pengenalan struktur data inti TensorFlow: **Tensors** (array multidimensi yang tidak dapat diubah) dan **Variables** (Tensor yang dapat diubah dan digunakan untuk menyimpan parameter model seperti bobot dan *bias*).
* **Komponen Kustom**: Cara membuat blok bangunan *deep learning* yang disesuaikan:
    * **Custom Loss Functions**: Mendefinisikan fungsi *loss* yang spesifik untuk masalah Anda.
    * **Custom Metrics**: Membuat metrik evaluasi yang disesuaikan.
    * **Custom Layers**: Mendesain lapisan jaringan saraf yang unik.
    * **Custom Models**: Mengimplementasikan arsitektur model secara penuh dari awal.
* **`tf.GradientTape`**: Alat esensial untuk menghitung gradien secara otomatis (**autodiff**). Ini merekam operasi yang dilakukan selama *forward pass* dan kemudian digunakan untuk menghitung gradien selama *backward pass*.
* **Custom Training Loop**: Memberikan kontrol penuh atas proses pelatihan, memungkinkan implementasi algoritma pelatihan yang tidak standar atau penyesuaian perilaku pelatihan secara rinci.
* **TF Functions (`@tf.function`)**: Sebuah *decorator* yang mengonversi fungsi Python biasa menjadi grafik TensorFlow berperforma tinggi. Ini mengoptimalkan eksekusi kode dengan mengompilasi ulang fungsi Python menjadi grafik yang dapat dieksekusi secara efisien oleh TensorFlow.

---

### Bab 13: Memuat dan Preprocessing Data dengan TensorFlow

Bab ini membahas cara membangun *data pipeline* yang efisien, skalabel, dan robust menggunakan TensorFlow, yang krusial untuk melatih model *deep learning* pada *dataset* besar.

* **Data API (`tf.data`)**: Alat utama TensorFlow untuk memuat, mentransformasi, mengacak (*shuffle*), mem-batch, dan melakukan *prefetching* data. API ini dirancang untuk menangani *dataset* yang tidak muat di memori dan mengoptimalkan *throughput* data ke GPU/TPU.
* **Format TFRecord**: Format biner efisien dari TensorFlow untuk menyimpan data besar. TFRecord mengemas data ke dalam *protobuf* dan dapat diakses secara *streaming*, ideal untuk *dataset* yang sangat besar.
* **Lapisan Preprocessing Keras**: Lapisan seperti `TextVectorization` (untuk mengubah teks menjadi representasi numerik) dan `Normalization` (untuk menormalisasi fitur numerik) yang memungkinkan langkah *preprocessing* menjadi bagian integral dari model. Ini menyederhanakan proses *deployment* karena *preprocessing* tidak perlu dilakukan secara terpisah di lingkungan produksi.
* **TFDS (TensorFlow Datasets)**: Sebuah *library* yang menyediakan ratusan *dataset* umum (gambar, teks, video) yang sudah dipersiapkan dan siap digunakan dengan `tf.data.Dataset` API, memudahkan eksperimen dan pengembangan.

---

### Bab 14: Computer Vision Mendalam dengan CNN

Bab ini secara khusus berfokus pada **Convolutional Neural Networks (CNNs)**, arsitektur yang sangat efektif untuk tugas-tugas pemrosesan gambar dan telah merevolusi bidang Computer Vision.

* **Lapisan Konvolusional & Pooling**: Blok pembangun fundamental dari CNN.
    * **Lapisan Konvolusional**: Menerapkan filter (*kernel*) yang belajar untuk mendeteksi fitur lokal (misalnya, tepi, tekstur) dalam gambar.
    * **Lapisan Pooling (misalnya, Max Pooling)**: Mengurangi dimensi spasial representasi fitur, mengurangi jumlah parameter, dan memberikan *invariance* terhadap sedikit pergeseran atau distorsi.
* **Arsitektur CNN Klasik dan Modern**: Pembahasan tentang arsitektur CNN yang terkenal dan berpengaruh:
    * **LeNet-5**: Salah satu CNN pertama yang sukses, untuk pengenalan angka.
    * **AlexNet**: Pemenang ImageNet 2012, menandai dimulainya era *deep learning* dalam *Computer Vision*.
    * **GoogLeNet (Inception)**: Memperkenalkan modul *Inception* yang menggabungkan beberapa ukuran filter dalam satu lapisan.
    * **ResNet (Residual Network)**: Memperkenalkan koneksi *skip* atau *residual* yang memungkinkan pelatihan jaringan yang sangat dalam.
* **Transfer Learning untuk Computer Vision**: Praktik umum dan sangat efektif untuk menggunakan model CNN yang telah dilatih sebelumnya pada *dataset* besar seperti ImageNet. Bobot yang sudah dilatih dapat digunakan sebagai *feature extractor* atau di-fine-tune untuk tugas *Computer Vision* yang baru.
* **Tugas Lanjutan dalam Computer Vision**: Pengenalan konsep-konsep seperti **Object Detection** (mengidentifikasi lokasi dan kelas objek dalam gambar, mis. dengan YOLO, Faster R-CNN) dan **Semantic Segmentation** (mengklasifikasikan setiap *pixel* dalam gambar ke dalam kategori objek, mis. dengan U-Net).

---

### Bab 15: Memproses Urutan dengan RNN dan CNN

Bab ini memperkenalkan arsitektur jaringan saraf yang dirancang khusus untuk memproses data sekuensial, di mana urutan informasi sangat penting.

* **Recurrent Neural Networks (RNNs)**: Jaringan saraf yang memiliki "memori" dan mampu memproses urutan input, di mana output saat ini bergantung pada input dan state sebelumnya. Ini dicapai dengan *loop* internal yang memungkinkan informasi untuk bertahan dari satu langkah waktu ke langkah waktu berikutnya.
* **Masalah Memori Jangka Pendek**: RNN sederhana kesulitan mengingat informasi dari urutan yang sangat panjang (**vanishing gradients**) yang terpisah oleh banyak langkah waktu.
* **LSTM (Long Short-Term Memory) dan GRU (Gated Recurrent Unit)**: Sel rekuren yang lebih canggih yang mengatasi masalah memori jangka pendek RNN sederhana. Mereka menggunakan mekanisme **gerbang (*gates*)** (misalnya, *input gate*, *forget gate*, *output gate* pada LSTM) yang mengatur aliran informasi, memungkinkan mereka untuk menyimpan atau melupakan informasi penting dari urutan panjang.
* **CNN untuk Urutan**: Meskipun awalnya dirancang untuk gambar, lapisan `Conv1D` (konvolusi 1D) dan arsitektur seperti **WaveNet** dapat digunakan secara efisien untuk memproses data sekuensial. CNN dapat menangkap pola lokal dalam urutan, dan dengan menumpuk lapisan konvolusional atau menggunakan *dilated convolutions*, mereka dapat menangkap dependensi jangka panjang. CNN seringkali lebih cepat dalam pelatihan dibandingkan RNN untuk urutan panjang.

---

### Bab 16: Pemrosesan Bahasa Alami dengan RNN dan Attention

Bab ini menerapkan model sekuensial untuk tugas-tugas **Natural Language Processing (NLP)**, bidang yang berfokus pada interaksi antara komputer dan bahasa manusia.

* **Analisis Sentimen & Text Generation**: Contoh aplikasi umum dari RNN dalam NLP.
    * **Analisis Sentimen**: Mengklasifikasikan polaritas emosional dari teks (positif, negatif, netral).
    * **Text Generation**: Membuat teks baru yang koheren dan relevan, seringkali pada level kata atau karakter.
* **Encoder-Decoder & Attention Mechanism**: Arsitektur yang umum digunakan untuk tugas-tugas *sequence-to-sequence* seperti penerjemahan mesin.
    * **Encoder**: Memproses urutan input dan mengkompresnya menjadi representasi vektor kontekstual.
    * **Decoder**: Mengambil representasi ini dan menghasilkan urutan output.
    * **Attention Mechanism**: Sebuah inovasi penting yang memungkinkan *decoder* untuk "memperhatikan" atau fokus pada bagian-bagian yang relevan dari *input sequence* saat menghasilkan setiap elemen output. Ini mengatasi keterbatasan representasi vektor tunggal pada *encoder* untuk urutan panjang.
* **Transformer**: Arsitektur revolusioner yang sepenuhnya mengandalkan mekanisme **self-attention** dan meninggalkan rekurensi (seperti RNN) atau konvolusi (seperti CNN). Transformer telah menjadi standar baru dan *state-of-the-art* dalam banyak tugas NLP karena kemampuannya untuk memproses urutan secara paralel dan menangkap dependensi jangka panjang secara efektif.
* **Model Bahasa Modern**: Pengenalan singkat tentang model *pretrained* besar yang telah mengubah lanskap NLP:
    * **BERT (Bidirectional Encoder Representations from Transformers)**: Model *encoder-only* yang belajar representasi kontekstual dari kata-kata dengan mempertimbangkan konteks dua arah.
    * **GPT-2 (Generative Pre-trained Transformer 2)**: Model *decoder-only* yang sangat besar dan kuat untuk *text generation*.

---

### Bab 17: Representation Learning dan Generative Learning

Bab ini membahas model *unsupervised* yang tidak hanya belajar representasi data yang efisien tetapi juga memiliki kemampuan untuk menghasilkan data baru yang menyerupai *dataset* pelatihan.

* **Autoencoders**: Jaringan saraf yang dilatih untuk merekonstruksi inputnya sendiri. Mereka terdiri dari **encoder** (yang memetakan input ke representasi dimensi rendah di **latent space**) dan **decoder** (yang merekonstruksi input dari *latent space*). Autoencoder berguna untuk:
    * **Reduksi Dimensi**: Representasi di *latent space* dapat menjadi versi data yang lebih ringkas.
    * **Deteksi Anomali**: Instance yang memiliki *reconstruction error* tinggi kemungkinan adalah anomali.
    * **Unsupervised Pretraining**: Bobot *encoder* dapat digunakan sebagai inisialisasi untuk model *supervised learning*.
* **Variational Autoencoder (VAE)**: Sebuah ekstensi dari autoencoder yang generatif dan probabilistik. VAE tidak hanya belajar representasi *latent* tetapi juga memastikan bahwa *latent space* terstruktur dengan baik (misalnya, mengikuti distribusi Gaussian). Ini memungkinkan pengambilan sampel dari *latent space* untuk menghasilkan data baru yang realistis.
* **Generative Adversarial Networks (GANs)**: Salah satu inovasi paling menarik dalam *generative modeling*. GAN terdiri dari dua jaringan saraf yang saling bersaing:
    * **Generator**: Bertanggung jawab untuk menghasilkan data baru yang realistis (misalnya, gambar).
    * **Discriminator**: Bertanggung jawab untuk membedakan antara data nyata dari *dataset* pelatihan dan data palsu yang dihasilkan oleh Generator.
    Kedua jaringan dilatih secara bersamaan dalam permainan *zero-sum*, di mana Generator mencoba menipu Discriminator dan Discriminator mencoba untuk tidak tertipu, menghasilkan kemampuan Generator untuk menghasilkan data yang sangat realistis.

---

### Bab 18: Reinforcement Learning

Pengenalan ke **Reinforcement Learning (RL)**, paradigma pembelajaran di mana agen (*agent*) belajar bagaimana bertindak di lingkungan (*environment*) untuk memaksimalkan *reward* kumulatif melalui interaksi coba-coba.

* **Konsep Inti RL**:
    * **Agent**: Entitas yang belajar dan bertindak dalam lingkungan.
    * **Environment**: Dunia tempat *agent* berinteraksi.
    * **Action**: Tindakan yang dapat diambil *agent*.
    * **Reward**: Sinyal umpan balik numerik yang diterima *agent* dari lingkungan.
    * **Policy**: Strategi *agent* untuk memilih tindakan berdasarkan *state* lingkungan saat ini.
* **OpenAI Gym**: Sebuah *toolkit* yang menyediakan berbagai lingkungan (*environment*) simulasi untuk mengembangkan dan membandingkan algoritma RL, seperti bermain game atau mengendalikan robot.
* **Algoritma RL Utama**:
    * **Policy Gradients (PG)**: Keluarga algoritma yang secara langsung mengoptimalkan *policy* *agent* untuk memaksimalkan *reward* yang diharapkan.
    * **Deep Q-Networks (DQN)**: Pendekatan *value-based* yang menggabungkan *Q-Learning* (belajar untuk mengestimasi nilai dari setiap pasangan *state-action*) dengan *deep neural networks* untuk menangani *state space* yang besar.
* **TF-Agents**: *Library* tingkat tinggi yang dibangun di atas TensorFlow untuk membangun, melatih, dan mengevaluasi sistem RL yang kompleks dan dapat diskalakan, menyediakan komponen yang dapat digunakan kembali untuk berbagai algoritma RL.

---

### Bab 19: Melatih dan Mengimplementasikan Model TensorFlow dalam Skala

Bab ini membahas aspek-aspek praktis dan tantangan dalam membawa model *deep learning* dari tahap eksperimen ke produksi, termasuk *deployment* dan pelatihan skala besar.

* **Menyimpan dan Melayani Model**:
    * **SavedModel**: Format standar TensorFlow untuk menyimpan model terlatih, termasuk arsitektur, bobot, dan graf komputasi.
    * **TensorFlow Serving**: Sistem *flexible*, performa tinggi untuk melayani model Machine Learning dalam produksi. Ia mendukung penyebaran model melalui Docker dan dapat melayani beberapa versi model secara bersamaan.
* **Deploy ke Cloud**: Mengimplementasikan model di platform *cloud* seperti **Google Cloud AI Platform** yang menyediakan layanan untuk melatih, menyetel, dan mendeploy model ML.
* **Deploy di Perangkat Terbatas**: Menggunakan **TensorFlow Lite (TFLite)**, *toolkit* yang dioptimalkan untuk menjalankan model TensorFlow pada perangkat seluler dan *embedded* (seperti Android, iOS, Raspberry Pi). TFLite menyediakan konverter model dan *interpreter* yang efisien.
* **Training Skala Besar**: Menggunakan **Distribution Strategies API (`tf.distribute`)** TensorFlow untuk melatih model secara paralel di beberapa GPU (pada satu mesin) atau di beberapa mesin (komputasi terdistribusi). Ini penting untuk melatih model yang sangat besar atau pada *dataset* yang masif.

---
