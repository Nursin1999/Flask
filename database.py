import sqlite3  # sqlite3 kütüphanesi eklenir
from flask import g  # Flask framework'ünün g değişkeni import edilir

DATABASE = 'database.db'  # veritabanı dosyasının ismi tanımlanır

def get_db():
    db = getattr(g, '_database', None)  # g değişkeninde "_database" adında bir değişken varsa, onu alır.
    if db is None:  # db yoksa
        db = g._database = sqlite3.connect(DATABASE)  # _database adında bir değişken oluşturur ve bu değişkene DATABASE'deki veritabanını bağlar.
    return db  # veritabanını döndürür

def close_db(e=None):
    db = g.pop('_database', None)  # g değişkeninde "_database" adında bir değişken varsa, onu alır ve g değişkeninden siler.
    if db is not None:  # db varsa
        db.close()  # veritabanını kapatır

def init_db():
    with open('E:/python/database.sql', 'r') as f:  # "database.sql" dosyasını okur
        database = f.read()  # dosyadaki içeriği "database" değişkenine atar
    db = get_db()  # veritabanını alır
    db.executescript(database)  # "database" değişkenindeki SQL komutlarını veritabanında çalıştırır.
    db.commit()  # işlemleri kaydeder ve veritabanını kapatır
