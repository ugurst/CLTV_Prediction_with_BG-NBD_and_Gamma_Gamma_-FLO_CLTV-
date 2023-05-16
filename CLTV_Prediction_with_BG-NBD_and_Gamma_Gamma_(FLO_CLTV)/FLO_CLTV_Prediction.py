##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################


###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from lifetimes.plotting import plot_period_transactions

pd.set_option("display.max_columns", 20)
# pd.set_option("display.max_rows", 20)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

###############################################################
# GÖREVLER
###############################################################
# GÖREV 1: Veriyi Hazırlama
# 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()
df.head()
df.describe().T


# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.


# outlier'ı hesaplamak için yüzde 25 ve 75lik eşik değerleri hesaplanır
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit)


# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
# aykırı değerleri varsa baskılayanız.

columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
           "customer_value_total_ever_online"]

df.skew()
# order_num_total_ever_online         10.4885
# order_num_total_ever_offline        20.3296
# customer_value_total_ever_offline   16.3007
# customer_value_total_ever_online    20.0858


# belirlediğimiz sütunları baskılıyoruz
for col in columns:
    replace_with_thresholds(df, col)

df.skew()
# order_num_total_ever_online         4.5002
# order_num_total_ever_offline        3.2979
# customer_value_total_ever_offline   3.6581
# customer_value_total_ever_online    4.3270


# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df["order_num_total"] = (df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]).astype(int)

df["total_purchase"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

df.dtypes
# first_order_date                      object
# last_order_date                       object
# last_order_date_online                object
# last_order_date_offline               object


df = df.apply(lambda x: pd.to_datetime(x) if "date" in x.name else x)
# first_order_date                     datetime64[ns]
# last_order_date                      datetime64[ns]
# last_order_date_online               datetime64[ns]
# last_order_date_offline              datetime64[ns]

# GÖREV 2: CLTV Veri Yapısının Oluşturulması
# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

df["last_order_date"].max()  # Timestamp('2021-05-30')
analysis_date = dt.datetime(2021, 6, 1)

# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
# Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.

cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).astype("timedelta64[D]")) / 7
cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype('timedelta64[D]')) / 7
cltv_df["frequency"] = df["order_num_total"]
cltv_df["monetary_cltv_avg"] = df["total_purchase"] / df["order_num_total"]

cltv_df.set_index("customer_id", inplace=True)
cltv_df.columns = ["recency", "T", "frequency", "monetary"]
cltv_df.head()

monthly_purchase = df.groupby(df["last_order_date"].dt.to_period("M")).sum()[["order_num_total_ever_online", "order_num_total_ever_offline"]]
monthly_purchase.plot(kind='line', figsize=(10, 6))
plt.title("Zaman İçindeki Alışveriş Trendleri")
plt.xlabel("Tarih")
plt.ylabel("Alışveriş Sayısı")
plt.show(block=True)


# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
# 1. BG/NBD modelini fit ediniz.

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])

# a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.


cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                           cltv_df["frequency"],
                                           cltv_df["recency"],
                                           cltv_df["T"])

# b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                           cltv_df["frequency"],
                                           cltv_df["recency"],
                                           cltv_df["T"])

# 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"], cltv_df["monetary"])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary"])
cltv_df.head()

# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time=6,  # ay periyodu
                                   freq="W",  # haftalık frekans T bilgisi
                                   discount_rate=0.01)

cltv_df["cltv"] = cltv
cltv_df.head()

plt.figure(figsize=(10, 6))
sns.histplot(cltv_df["cltv"], kde=True, bins=20)
plt.title("Müşteri Ömür Değeri (CLTV) Dağılımı")
plt.xlabel("CLTV")
plt.ylabel("Müşteri Sayısı")
plt.show(block=True)


# b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.

cltv_df.sort_values("cltv", ascending=False)[:20]



# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
# 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.
cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()



