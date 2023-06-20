#region Temel String Islemleri

isim = "Kadir"
3 * isim

isim = isim + "Can"
"X" + isim[1:]

isimler = ["Ali", "Veli", "Mehmet"]
for isim in isimler:
    print("Isim:", isim, sep=" ")

for isim in isimler:
    print("_", isim[0:], sep="")

print(isimler)
# ['Ali', 'Veli', 'Mehmet']
print(*isimler)
# Ali Veli Mehmet

for isim in enumerate(isimler):
    print(isim)

for isim in enumerate(isimler, 1):
    print(isim)
#endregion

#region Dizi Icinde Tip Sorgulamalari
# Alfabetik mi?
"kcu".isalpha() # True
"kcu97".isalpha() # False

"123".isnumeric() # True
"123".isdigit() # True

"kcu97".isalnum() # Alfa-numerik mi? - True
#endregion

#region Elemanlarina ve Eleman Indekslerine Erismek
isim = "kcu"
isim[0]
isim[0:2]

isim = "kadircanuzumcu"
isim.index("r")
isim.index("a")
isim.index("a", 2)
#endregion

#region Baslangic ve Bitis Karakterlerini Sorgulamak
isim = "Kadir Can UZUMCU"
isim.startswith("K") #True
isim.startswith("k") # False
isim.endswith("U") # True
isim.endswith("u") # False

isim.count("a") # 2
sorted(isim) # Alfabetik siralama

print(*sorted(isim), sep="") # "  CCKMUUUZaadinr"
#endregion

#region Karakterleri Bolmek
isim = "Kadir Can UZUMCU"
isim.split()
isim.split("U")
#endregion

#region Buyuk-Kucuk Harf Islemleri
isim = "Kadir Can UZUMCU"
isim.upper()
isim.lower()

isim = isim.upper()
isim.islower() # False

isim = isim.lower()
isim.capitalize()
isim = isim.title()
isim.swapcase()
#endregion

#region Karakter Kirpmak veya Birlestirmek
isim = " Kadir Can "
isim.strip()
isim = " Kadir Can*"
isim.strip("*")

isim.rstrip("*")
isim.lstrip()

isim = "Kadir Can UZUMCU"
ayrik = isim.split()

joiner = "#"
joiner.join(ayrik)
#endregion

#region Karakter Degistirme
isim = "Kadir Can UZUMCU"
isim.replace("a", "e")
isim.replace("Can", "John")

metin = "a b c ç d e f g ğ"
turkce_harfler = "çÇğĞıİöÖüÜşŞ"
yeni_harfler = "cCgGiIoOuUsS"

yeni_metin = str.maketrans(turkce_harfler, yeni_harfler)
metin.translate(yeni_metin)

import pandas as pd
isimler = ["Ali", "Veli", "Ahmet", "Mehmet", "Hasan", "Huseyin", "ali", "aali"]
v = pd.Series(isimler)
v.str.contains("li")
v.str.contains("et").sum()

v[v.str.contains("H")]
v[v.str.contains("H")].count()

v.str.contains("[aA]li").sum()
#endregion