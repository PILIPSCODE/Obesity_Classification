# Submission 1: Spam Sms Classification
Nama:Pilipus Kuncoro Wismoady

Username dicoding:Pilcotech

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) |
| Masalah |Pesan spam yang masuk melalui SMS merupakan salah satu masalah yang sering mengganggu pengguna ponsel, bahkan dapat membahayakan karena berisi penipuan atau tautan berbahaya. Untuk mengatasi hal ini, dibutuhkan sistem yang mampu secara otomatis membedakan antara pesan spam dan pesan yang sah (ham). Dengan berkembangnya teknologi pemrosesan bahasa alami atau Natural Language Processing (NLP), kita dapat membangun model klasifikasi yang dapat mendeteksi pesan spam secara akurat. Dataset Spam SMS Classification Using NLP dari Kaggle menyediakan data pesan SMS yang telah diberi label, sehingga sangat bermanfaat sebagai bahan pelatihan dan pengujian model klasifikasi berbasis teks. |
| Solusi machine learning |Sebagai solusi untuk mengatasi pesan spam, digunakan pendekatan machine learning yang memungkinkan sistem belajar dari data historis pesan SMS yang telah diberi label sebagai spam atau ham. Dengan memanfaatkan teknik Natural Language Processing (NLP), setiap pesan diubah menjadi representasi numerik agar dapat dianalisis oleh algoritma. Model machine learning kemudian dilatih untuk mengenali pola kata, struktur kalimat, dan ciri khas dari pesan spam. Setelah model dilatih, sistem dapat secara otomatis memprediksi apakah sebuah pesan termasuk spam atau bukan, sehingga proses penyaringan menjadi lebih cepat, akurat, dan efisien tanpa perlu pengecekan manual. |
| Metode pengolahan | Di Dataset ini terdapat 2 column yaitu v1 dan v2. Langkah pertama mengubah semua nama column menjadi lebih mudah dipahami. setelah itu mengubah column v1 menjadi numerik 1 dan 0. 1 untuk spam dan 0 untuk tidak, kemudian membagi dataset menjadi 80/20. 80 untuk train 20 untuk eval.|
| Arsitektur model | Model ini menggunakan arsitektur yang terdiri dari lapisan Embedding, Bidirectional LSTM, Dense (Fully Connected), dan Output Layer dengan aktivasi sigmoid. Optimizer yang digunakan adalah Adam, dan fungsi loss-nya adalah BinaryCrossentropy.|
| Metrik evaluasi |Metrik evaluasi yang digunakan yaitu ExampleCount, AUC, FalsePositives, TruePositives, FalseNegatives, TrueNegatives, dan BinaryAccuracy |
| Performa model | Berdasarkan nilai metrik yang dihasilkan dari Evaluator, diperoleh nilai keseluruhan AUC sebesar 98%, Binary Accuracy sebesar 98%, FN, FP, TN, TP masing-masing 11, 11, 939, dan 130 dari 1091 Example Count. Secara keseluruhan, performa model ini sangat baik. Nilai AUC dan akurasi 97% menunjukkan model sangat mampu membedakan spam dan bukan spam. |
