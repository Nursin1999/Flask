/*'messages' tablosunu varsa siler*/
DROP TABLE IF EXISTS messages;

CREATE TABLE messages ( /*'messages' tablosunu oluşturur.*/
id INTEGER PRIMARY KEY AUTOINCREMENT, /*'id' INTEGER türünde ve otomatik artan bir PRIMARY KEY olarak belirlenir.*/
message TEXT NOT NULL, /*'message' alanı, zorunlu olarak girilmesi gereken TEXT tipinde olacaktır.*/
predicted_label INTEGER NOT NULL /*'predicted_label' INTEGER olarak belirlenir.*/
);