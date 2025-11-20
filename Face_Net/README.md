# FaceNet Face Verification and Identification
Proyek ini menerapkan FaceNet untuk deteksi wajah, alignment, ekstraksi embedding 512 dimensi, verifikasi wajah satu lawan satu, dan identifikasi multi-kelas menggunakan SVM. Struktur dan alur kerja mengikuti modul praktikum FaceNet.
## Struktur Folder
``` 
FaceNet/
│
├── data/
│   ├── train/
│   │   ├── arya/
│   │   │   ├── a1.jpg
│   │   │   └── a2.jpg 
│   │   └── dimas/
│   │       ├── b1.jpg 
│   │       └── b2.jpg 
│   │
│   └── val/
│       ├── arya/
│       │   └── a1.jpg 
│       └── dimas/
│           └── b2.jpg 
│
├── build_embeddings.py
├── eval_folder.py
├── facenet_svm.joblib
├── predict_one.py
├── train_classifier.py
├── train_knn.py
├── utils_facenet.py
├── verify_cli.py
├── verify_pair.py

## Tujuan Praktikum
* Memahami deteksi dan alignment wajah dengan MTCNN
* Menghasilkan embedding FaceNet dimensi 512
* Melakukan verifikasi wajah satu lawan satu
* Melakukan identifikasi wajah menggunakan SVM
* Mengevaluasi akurasi model pada data validasi

## Analisis Setiap Kode File
### build_embeddings.py
Program ini untuk mengambil semua gambar yang ada di dalam folder dataset, mendeteksi wajahnya, lalu mengubahnya menjadi embedding berukuran 512 dimensi. Cara kerjanya cukup rapi karena langsung membaca folder sebagai label, jadi tidak perlu memberi label manual. Prosesnya dibuat lebih ringan karena gambar diproses satu per satu, bukan sekaligus. Ada progress bar juga supaya kita tahu sampai mana prosesnya. Secara keseluruhan sudah enak dipakai, meski nanti mungkin bisa ditambah pengecekan format gambar dan pengaturan path yang lebih fleksibel..

### eval_folder.py
Program ini digunakan untuk mengecek apakah model yang sudah dilatih benar-benar bisa mengenali wajah dengan baik. Ia mengambil gambar-gambar di folder validasi, menghitung embedding-nya, lalu mencocokkannya dengan model SVM. Di akhir proses kita bisa melihat akurasinya, baik secara keseluruhan maupun per orang. Kelemahannya, file ini sangat bergantung pada struktur folder tertentu. Kalau nama foldernya beda-beda atau tidak terstruktur, evaluasinya bisa salah atau gagal.

### predict_one.py
program predict_one.py bekerja sebagai alat sederhana untuk menebak identitas wajah dari satu gambar. Script ini memuat model SVM yang sudah dilatih, lalu mengambil embedding wajah dari gambar menggunakan fungsi embed_from_path. Jika wajah tidak terdeteksi, hasilnya langsung dianggap "NO_FACE". Setelah embedding berhasil dibuat, model akan menghitung probabilitas tiap kelas dan memilih kemungkinan tertinggi sebagai prediksi. Agar lebih aman, script juga menyediakan ambang batas kepercayaan (threshold) sehingga jika nilai confidence terlalu rendah, wajah tersebut diberi label "UNKNOWN". Bagian akhirnya hanya berfungsi sebagai contoh penggunaan, tempat kita bisa mengganti path gambar dan langsung melihat hasil prediksi dan tingkat kepercayaannya.

### train_classifier.py
Progran ini bertugas melatih model SVM untuk pengenalan wajah dengan menggunakan embedding yang sudah dibuat sebelumnya. Script memuat data X_train.npy dan y_train.npy, menampilkan jumlah sampel dan daftar kelas, lalu membangun pipeline yang terdiri dari proses standardisasi dan SVM ber-kernel RBF. Karena jumlah data biasanya tidak terlalu besar, pelatihan dilakukan langsung tanpa cross-validation agar cepat dan sederhana. Setelah model selesai dilatih, file ini menyimpannya sebagai facenet_svm.joblib sehingga bisa digunakan pada proses prediksi atau verifikasi. Di akhir, script juga menghitung akurasi training hanya untuk mengetahui seberapa baik model mempelajari data yang diberikan.

###  train_knn.py
Digunakan untuk melatih model KNN sebagai alternatif dari SVM dalam sistem pengenalan wajah. Script ini memuat embedding dan label dari file .npy, menampilkan jumlah data yang digunakan, lalu membuat pipeline yang berisi proses standardisasi dan model KNN dengan 3 tetangga terdekat menggunakan jarak Euclidean. Setelah model dilatih, hasilnya disimpan dalam file facenet_knn.joblib agar bisa langsung digunakan untuk prediksi. Pendekatan ini sederhana dan cocok untuk dataset kecil karena KNN tidak membutuhkan proses training yang rumit seperti SVM.

### utils_facenet.py
Program ini berisi kumpulan fungsi inti yang menangani seluruh proses penting dalam pengenalan wajah, mulai dari membaca gambar, mendeteksi wajah, melakukan alignment, hingga menghasilkan embedding berukuran 512 dimensi menggunakan model FaceNet. Script ini otomatis memilih GPU jika tersedia agar proses lebih cepat, sekaligus tetap kompatibel dengan CPU. MTCNN digunakan untuk mendeteksi dan merapikan wajah, sementara model InceptionResnetV1 dari VGGFace2 dipakai untuk menghasilkan embedding. Fungsi-fungsinya dibuat terpisah agar mudah digunakan kembali: ada fungsi untuk membaca gambar dalam format BGR, mengubah ke PIL, mendeteksi wajah, menghitung embedding, dan bahkan menyediakan fungsi cosine similarity untuk mengukur kemiripan dua embedding. Secara keseluruhan, file ini adalah “mesin utama” yang membuat modul-modul lain bisa melakukan prediksi dan verifikasi wajah dengan lancar.

###  verify_cli.py
Program ini dibuat sebagai alat sederhana berbasis command-line untuk melakukan verifikasi wajah 1:1, yaitu membandingkan apakah dua gambar menunjukkan orang yang sama. Program menerima dua path gambar dan sebuah nilai threshold melalui argumen terminal, lalu memproses masing-masing gambar menggunakan fungsi embed_from_path untuk menghasilkan vektor embedding wajah. Jika salah satu gambar tidak terdeteksi wajahnya, program langsung memberi tahu pengguna. Jika keduanya berhasil diproses, program menghitung tingkat kemiripan menggunakan cosine_similarity, kemudian memutuskan apakah kedua wajah tersebut dianggap cocok berdasarkan nilai threshold yang ditetapkan (default 0.85). Hasil akhirnya berupa angka similarity dan status “MATCH” atau “NO MATCH” sehingga pengguna bisa langsung memahami seberapa mirip kedua wajah yang diuji.

### verify_pair.py
Program ini berfungsi sebagai skrip cepat untuk menguji apakah dua gambar wajah termasuk orang yang sama tanpa perlu argumen command-line. Program mengambil dua path gambar yang sudah ditentukan, lalu memprosesnya menggunakan embed_from_path untuk menghasilkan embedding wajah. Jika salah satu gambar tidak berhasil mendeteksi wajah, skrip langsung memberi peringatan. Jika keduanya valid, kode menghitung tingkat kemiripan menggunakan cosine similarity dan menampilkan angka similarity tersebut. Terakhir, script membandingkan hasil similarity dengan threshold 0.85 untuk menentukan apakah kedua gambar dianggap cocok atau tidak, sehingga memudahkan pengguna mengecek kualitas data atau memastikan bahwa model bekerja sesuai harapan.

## Cara Menjalankan
* Jalankan build_embeddings.py untuk membuat embedding training.
* Jalankan train_classifier.py untuk melatih model SVM.
* Jalankan predict_one.py untuk mengenali satu gambar.
* Jalankan verify_pair.py untuk membandingkan dua wajah.
* Jalankan verify_cli.py untuk verifikasi cepat lewat terminal.
* Jalankan eval_folder.py untuk mengevaluasi performa model pada folder val.
## Fungsi Output
* X_train.npy berisi embedding wajah untuk training.
* y_train.npy berisi label wajah untuk training.
* facenet_svm.joblib berisi model klasifikasi identitas.
* Training accuracy menunjukkan performa model pada data train.
* Hasil prediksi menunjukkan identitas wajah dan tingkat kepercayaan.
* Hasil verifikasi menunjukkan kekuatan kemiripan dua wajah.
* Akurasi evaluasi menunjukkan kemampuan model pada data uji.
