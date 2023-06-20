#region Temel String Islemleri

# Toplama/Çarpma islemleri yapilabilir.
isim = "Kadir"
3 * isim
isim = isim + "Can"
"X" + isim[1:]

# Listenin elemanlarini yazdirma
isimler = ["Ali", "Veli", "Mehmet"]
for isim in isimler:
    print("Isim:", isim, sep=" ")

# Her elemanin basina karakter ekleme
for isim in isimler:
    print("_", isim[0:], sep="")


print(isimler)
# ['Ali', 'Veli', 'Mehmet']
# Listenin basina * karakterin getirildiğinde tek bir eleman olarak cikti verir.
print(*isimler)
# Ali Veli Mehmet

# Liste elemanlarini numaralandirma
for isim in enumerate(isimler):
    print(isim)

# Indeksi 1den baslatarak numaralandirma
for isim in enumerate(isimler, 1):
    print(isim)
#endregion

#region Dizi Icinde Tip Sorgulamalari
# Alfabetik mi?
"kcu".isalpha() # True
"kcu97".isalpha() # False

# Numerik mi?
"123".isnumeric() # True
"123".isdigit() # True

# Alfa-numerik mi?
"kcu97".isalnum() # True
#endregion

#region Elemanlarina ve Eleman Indekslerine Erismek

# String degiskeni liste gibi kullanabiliriz. Slice islemi yapilabilir.
isim = "kcu"
isim[0]
isim[0:2]

# String degiskenlerin her elemaninin indeksi bulunur.
isim = "kadircanuzumcu"
isim.index("r")
isim.index("a")
isim.index("a", 2)
#endregion

#region Baslangic ve Bitis Karakterlerini Sorgulamak

isim = "Kadir Can UZUMCU"
# Baslangic ve bitis harflerini sorgulama. Buyuk/kucuk harf hassasiyeti vardir.
isim.startswith("K") #True
isim.startswith("k") # False
isim.endswith("U") # True
isim.endswith("u") # False

# String icinde karakter frekansi
isim.count("a") # 2

# Alfabetik siralama
sorted(isim)

# Alfabetik siralayip liste elemanlarini birlestirme.
print(*sorted(isim), sep="") # "  CCKMUUUZaadinr"
#endregion

#region Karakterleri Bolmek

# String degeri parcalara ayirma, varsayilan degeri bosluktur.
isim = "Kadir Can UZUMCU"
isim.split()
isim.split("U")
#endregion

#region Buyuk-Kucuk Harf Islemleri

# String degerlerin butun karakterlerini buyuk/kucuk yapma.
isim = "Kadir Can UZUMCU"
isim.upper()
isim.lower()

# Buyuk/kucuk harf sorgulama.
isim = isim.upper()
isim.islower() # False

# Degerin yalnizce ilk harfini buyuk yapma.
isim = isim.lower()
isim.capitalize()

# Degerin her kelimesinin ilk harfini buyuk yapma
isim = isim.title()

# Buyuk harfleri kucuk, kucuk harfleri buyuk yapma
isim.swapcase()
#endregion

#region Karakter Kirpmak veya Birlestirmek

# String degerlerin bas ve sonundan karakter cikartma. Varsayilan deger bosluktur.
isim = " Kadir Can "
isim.strip()
isim = " Kadir Can*"
isim.strip("*")

# Sol ve sag olarak, yalnizca bastan veya sondan karakter cikartma.
isim.rstrip("*")
isim.lstrip()

# Liste elemanlarinin arasina string bir deger getirerek birlestirme.
isim = "Kadir Can UZUMCU"
ayrik = isim.split()
joiner = "#"
joiner.join(ayrik)
#endregion

#region Karakter Degistirme

# Istemedigimiz karakterleri yeni karakterlerle degistirmek
isim = "Kadir Can UZUMCU"
isim.replace("a", "e")
isim.replace("Can", "John")

# Ayni islemi kapsamli ve fonksiyon haline getirerek yapmak.
metin = "a b c ç d e f g ğ"
turkce_harfler = "çÇğĞıİöÖüÜşŞ"
yeni_harfler = "cCgGiIoOuUsS"
yeni_metin = str.maketrans(turkce_harfler, yeni_harfler)
metin.translate(yeni_metin)


import pandas as pd
isimler = ["Ali", "Veli", "Ahmet", "Mehmet", "Hasan", "Huseyin", "ali", "aali"]
v = pd.Series(isimler)

# Vektor icinde istedigimiz karakterlerin bulundugu elemanlari yakalamak.
v.str.contains("li") # True/False

# Elemanlari getirmek
v[v.str.contains("H")]

# Sayisini ogrenmek
v[v.str.contains("H")].count()

# Sayisini ogrenmek
v.str.contains("[aA]li").sum()
#endregion