#########importy danych, czasu, ML
import numpy as np
import pandas as pd
import yfinance as yf
yf.pdr_override()
from pandas_datareader import data as pdr
import socket
import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
###########importy apki
from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.menu import MDDropdownMenu
from kivy.uix.screenmanager import ScreenManager
from kivymd.uix.screen import MDScreen
from kivy.uix.screenmanager import NoTransition
from kivy.uix.screenmanager import FadeTransition
from kivy.clock import Clock
from kivy.storage.jsonstore import JsonStore
from kivymd.uix.button import MDButton
from kivymd.uix.textfield import MDTextField, MDTextFieldMaxLengthText, MDTextFieldHintText, MDTextFieldLeadingIcon, MDTextFieldTrailingIcon
################################################################
################################################################
#######################################################funkcje
#funkcja spr internetu
def connection():
    try:
        socket.create_connection(("1.1.1.1", 53))
        return True
    except:
        pass

#funkcja do aplikacji -> dane pobrane i zestawy uczące, testujące, próbka z dnia kolejnego
def data_from_web(sym, n):
    start=str(datetime(datetime.now().year-10,datetime.now().month,datetime.now().day).date())
    global data
    data=[]
    while len(data)==0:
        df=pdr.get_data_yahoo(sym, start=start)
        data=df.filter(['Close']).values
    global X
    global Y
    X=[]
    Y=[]
    for i in range(60+n-1,len(data)):
        X.append(data[i-n+1-60:i-n+1,0])
        Y.append(data[i,0])

    X=np.array(X)
    Y=np.array(Y)
    X_akt=data[-60:]
    X_akt=X_akt.reshape(1,len(X_akt))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)
    Y_train=Y_train.reshape(len(Y_train),1)
    Y_test=Y_test.reshape(len(Y_test),1)
    return X_train, Y_train, X_test, Y_test, X_akt

#funkcja błędu
def error(y_true,y_pred):
    MAPE = 100*mean_absolute_percentage_error(y_true, y_pred)
    return MAPE

#funkcja przypisania symbolu
def choosen_symbol(x):
    global out
    out=''
    match x:
        case 'EUR/USD':
            out='EURUSD=X'
        case 'GBP/USD':
            out='GBPUSD=X'
        case 'AUD/USD':
            out='AUDUSD=X'
        case 'NZD/USD':
            out='NZDUSD=X'
        case 'EUR/GBP':
            out='EURGBP=X'
        case 'USD/CHF':
            out='CHF=X'
        case 'USD/JPY':
            out='JPY=X'
        case 'EUR/JPY':
            out='EURJPY=X'
        case 'USD/CAD':
            out='CAD=X'
        case 'USD/CNY':
            out='CNY=X'
        case 'TSLA':
            out='TSLA'
        case 'AAPL':
            out='AAPL'
        case 'GOOGL':
            out='GOOGL'
        case 'AMZN':
            out='AMZN'
        case 'INTC':
            out='INTC'
        case 'META':
            out='META'
        case 'MSFT':
            out='MSFT'
        case 'NVDA':
            out='NVDA'
        case 'XOM':
            out='XOM'
        case 'S&P 500':
            out='^GSPC'
        case 'DOW JONES':
            out='^DJI'
        case 'NASDAQ':
            out='^IXIC'
        case 'FTSE 100':
            out='^FTSE'
        case 'DAX':
            out='^GDAXI'
        case 'GOLD':
            out='GC=F'
        case 'BRENT OIL':
            out='BZ=F'
        case 'BTC/USD':
            out='BTC-USD'
        case 'ETH/USD':
            out='ETH-USD'
        case 'XRP/USD':
            out='XRP-USD'
        case 'DOGE/USD':
            out='DOGE-USD'
    return out

#funkcja do sprawdzani flagi
def check_flag(s,n):
    d= str(datetime.now().day)
    store = JsonStore('flags.json')
    if store.exists('d'):
        if store.get('d')['day']==str(d):
            pass
        else:
            store.clear()
    if store.exists(f'{s} {n}'):
        return str(store.get(f'{s} {n}')['p']), str(store.get(f'{s} {n}')['e'])
    else:
        return ('0','0')
        
#funkcja zapisu flagi
def store_flag(s,n,p,e):
    d= str(datetime.now().day)
    store = JsonStore('flags.json')
    store.put('d', day=str(d))
    store.put(f'{s} {n}', p=p, e=e)

#funckja dla emaila
def check_mail():
    store_m= JsonStore('flags3.json')
    if store_m.exists('m'):
        return str(store_m.get('m')['mail'])
    else:
        return ""

#funkcja premium
def check_vip():
    store_v = JsonStore('flags2.json')
    m=datetime.now(timezone.utc).month
    d=datetime.now(timezone.utc).day
    y=datetime.now(timezone.utc).year
    if store_v.exists('v'):
        if datetime.fromtimestamp((store_v.get('v')['pr'])/179).year==y:
            if datetime.fromtimestamp((store_v.get('v')['pr'])/179).month==m:
                return 1
            else:
                if datetime.fromtimestamp((store_v.get('v')['pr'])/179).month==m-1:
                    if datetime.fromtimestamp((store_v.get('v')['pr'])/179).day>d:
                        return 1
                    else:
                        return 0
                else:
                    return 0
        else:
            if datetime.fromtimestamp((store_v.get('v')['pr'])/179).year==y-1:
                if datetime.fromtimestamp((store_v.get('v')['pr'])/179).day>d:
                    return 1
                else:
                    return 0
            else:
                return 0
    else:
        store_v.put('v', pr=1234567890)
        return 0
 
#funkcja kodu premium:
def prem_code(m, code):
    store_v = JsonStore('flags2.json')
    mail=m.upper()
    m=datetime.now(timezone.utc).month
    d=datetime.now(timezone.utc).day
    y=datetime.now(timezone.utc).year
    a=str(datetime(y,m,d).replace(tzinfo=timezone.utc).timestamp())[:8]
    b=str(int(a)+m*10000000)
    a=b
    fir=(chr(ord(mail[0]) + int(d)+1)).upper()
    sec=(chr(ord(mail[1]) + int(d)+ 4)).upper()
    thi=(chr(ord(mail[2]) + int(d)+ 7)).upper()
    fou=(chr(ord(mail[3]) + int(d)+ 10)).upper()
    k=f'{a[3]}{sec}{a[0]}-{fir}{a[7]}{a[4]}-{a[5]}{a[2]}{thi}-{a[1]}{a[6]}{fou}'
    if k==code:
        store_v.put('v', pr=int(str(datetime.timestamp(datetime.now(timezone.utc)))[:10])*179)
    else:
        pass

#funkcja do uczenia modeli i predykcji
def learn():
    sym = MainApp.get_running_app().p
    l= MainApp.get_running_app().e
    s=choosen_symbol(sym)
    flag1, flag2 = check_flag(sym,int(l))
    if flag1=='0' and flag2=='0':
        X_train, Y_train, X_test, Y_test, X_akt=data_from_web(s, int(l))

        #skalery definicje
        In = MinMaxScaler()
        In.fit(X_train)
        Out = StandardScaler()
        Out.fit(Y_train)

        #skalowanie
        S_X_train = In.transform(X_train)
        S_X_test = In.transform(X_test) 
        S_X_akt = In.transform(X_akt)
        S_Y_train = Out.transform(Y_train)
        S_Y_test = Out.transform(Y_test) 
    ##########################################################
    ##################################################prognozowanie
        #modele, uczenie, testowanie
        R3=KNeighborsRegressor(n_neighbors = 3, weights = 'distance')
        R5=MLPRegressor(hidden_layer_sizes=43, activation='relu',
                            solver='lbfgs',alpha=0.1,max_iter=15000,
                            tol=0.000001,verbose=False, n_iter_no_change=5)
        R6=SVR(gamma='auto',C=100,epsilon=10**-6,verbose=False)
        VR=VotingRegressor([ ('KNR', R3),('MLP', R5),('SVM', R6)],verbose=False)
        VR.fit(S_X_train,S_Y_train.reshape(len(S_Y_train),))
        PredVR=VR.predict(S_X_test)
        PredVR = Out.inverse_transform(PredVR.reshape(len(PredVR),1))
        PredVR_E = error(Y_test,PredVR.reshape(len(PredVR),1))
        prognoza=VR.predict(S_X_akt)
        prognoza=Out.inverse_transform(prognoza.reshape(1,1)).reshape(1,)
        result=str(round(float(prognoza[0]),4))
        mean_error=str(round(PredVR_E,4))+"%"
        store_flag(sym,int(l),result,mean_error)
    else:
        result=flag1
        mean_error=flag2
        
    MainApp.get_running_app().root.current ='prediction'
    MainApp.get_running_app().root.get_screen('prediction').ids.pred_title.text = f"Prediction for {sym}, {l} day(s) ahead:"
    MainApp.get_running_app().root.get_screen('prediction').ids.prediction.text = f"{result}*"
    MainApp.get_running_app().root.get_screen('prediction').ids.mean_error.text = f"AI model's mean prediction error: {mean_error}"
######################################################################
#####################################################################

symbol=['EUR/USD', 'GBP/USD','AUD/USD','NZD/USD', 'EUR/GBP', 'USD/CHF', 'USD/JPY', 'EUR/JPY',
        'USD/CAD', 'USD/CNY', 'AAPL', 'AMZN', 'GOOGL', 'INTC', 'META', 'MSFT', 'NVDA', 'TSLA',
        'XOM', 'S&P 500', 'DOW JONES', 'NASDAQ', 'FTSE 100', 'DAX', 'GOLD', 'BRENT OIL',
        'BTC/USD','ETH/USD', 'XRP/USD', 'DOGE/USD']
liczba=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]


######################################################################
#####################################################################
###apka

class Screen1(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

class Screen2(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
class Screen3(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Screen4(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Screen5(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

class ScreenM(ScreenManager):
    pass


class MainApp(MDApp):
    p='EUR/USD'
    e='1'
    premium=check_vip()
    if premium==0:
        dis=True
        dis_color="#757575"
    if premium==1:
        dis=False
        dis_color="#A4E7FE"
    
    def open_drop_item_menu(self, item):
        menu_items = [
            {
                "text": f"{symbol[0]}",
                "theme_text_color": "Custom",
                "text_color": "#A4E7FE",
                "on_release": lambda x=f"{symbol[0]}": self.menu_callback(x),
                "disabled": False
            },
            {
                "text": f"{symbol[1]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[1]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[2]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[2]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[3]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[3]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[4]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[4]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[5]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[5]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[6]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[6]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[7]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[7]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[8]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[8]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[9]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[9]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[10]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[10]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[11]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[11]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[12]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[12]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[13]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[13]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[14]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[14]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[15]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[15]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[16]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[16]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[17]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[17]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[18]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[18]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[19]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[19]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[20]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[20]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[21]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[21]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[22]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[22]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[23]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[23]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[24]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[24]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[25]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[25]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[26]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[26]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[27]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[27]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[28]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[28]}": self.menu_callback(x),
                "disabled": self.dis
            },
            {
                "text": f"{symbol[29]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{symbol[29]}": self.menu_callback(x),
                "disabled": self.dis
            }
            
        ]
        
        self.drop_item_menu = MDDropdownMenu(
            caller=item, items=menu_items, position="bottom",ver_growth="down"
        )
        self.drop_item_menu.open()

    def menu_callback(self, text_item):
        self.root.get_screen('selection').ids.drop_text.text = text_item
        self.p=text_item
        self.drop_item_menu.dismiss()

    def open_drop_item_menu2(self, item2):
        menu_items2 = [
            {
                "text": f"{liczba[0]}",
                "theme_text_color": "Custom",
                "text_color": "#A4E7FE",
                "on_release": lambda x=f"{liczba[0]}": self.menu_callback2(x),
                "disabled": False
            },
            {
                "text": f"{liczba[1]}",
                "theme_text_color": "Custom",
                "text_color": "#A4E7FE",
                "on_release": lambda x=f"{liczba[1]}": self.menu_callback2(x),
                "disabled": False
            },
            {
                "text": f"{liczba[2]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{liczba[2]}": self.menu_callback2(x),
                "disabled": self.dis
            },
            {
                "text": f"{liczba[3]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{liczba[3]}": self.menu_callback2(x),
                "disabled": self.dis
            },
            {
                "text": f"{liczba[4]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{liczba[4]}": self.menu_callback2(x),
                "disabled": self.dis
            },
            {
                "text": f"{liczba[5]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{liczba[5]}": self.menu_callback2(x),
                "disabled": self.dis
            },
            {
                "text": f"{liczba[6]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{liczba[6]}": self.menu_callback2(x),
                "disabled": self.dis
            },
            {
                "text": f"{liczba[7]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{liczba[7]}": self.menu_callback2(x),
                "disabled": self.dis
            },
            {
                "text": f"{liczba[8]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{liczba[8]}": self.menu_callback2(x),
                "disabled": self.dis
            },
            {
                "text": f"{liczba[9]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{liczba[9]}": self.menu_callback2(x),
                "disabled": self.dis
            },
            {
                "text": f"{liczba[10]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{liczba[10]}": self.menu_callback2(x),
                "disabled": self.dis
            },
            {
                "text": f"{liczba[11]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{liczba[11]}": self.menu_callback2(x),
                "disabled": self.dis
            },
            {
                "text": f"{liczba[12]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{liczba[12]}": self.menu_callback2(x),
                "disabled": self.dis
            },
            {
                "text": f"{liczba[13]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{liczba[13]}": self.menu_callback2(x),
                "disabled": self.dis
            },
            {
                "text": f"{liczba[14]}",
                "theme_text_color": "Custom",
                "text_color": self.dis_color,
                "on_release": lambda x=f"{liczba[14]}": self.menu_callback2(x),
                "disabled": self.dis
            }
        ]

        self.drop_item_menu2 = MDDropdownMenu(
            caller=item2, items=menu_items2, position="bottom",ver_growth="down"
        )
        self.drop_item_menu2.open()

    def menu_callback2(self, text_item2):
        self.root.get_screen('selection').ids.drop_text2.text = text_item2
        self.e=text_item2
        self.drop_item_menu2.dismiss()

    def button_fun1(self):
        if (connection()):
            self.root.get_screen('selection').ids.internet.text = ""         
            self.root.transition = FadeTransition()
            self.root.current='learning'
            def button_fun2(self):
                learn()
            Clock.schedule_once(button_fun2,1)
        else:
            self.root.get_screen('selection').ids.internet.text = "No Internet connection!!!"

    def restart(self):
        self.premium=check_vip()
        if self.premium==0:
            self.dis=True
            self.dis_color="#757575"
        if self.premium==1:
            self.dis=False
            self.dis_color="#A4E7FE"
        self.root.transition = FadeTransition()
        self.root.current='selection'
        
    def prem_butt(self):
        self.root.get_screen('premium').ids.mail.text = check_mail()
        self.root.get_screen('premium').ids.code_err.text = ""
        if self.premium==0:
            self.root.transition = FadeTransition()
            self.root.current='premium'
        if self.premium==1:
            pass

    def conf_mail(self):
        store_m = JsonStore('flags3.json')
        store_m.put('m', mail=str(self.root.get_screen('premium').ids.mail.text))

    def apply_code(self):
        store_m= JsonStore('flags3.json')
        if len(store_m.get('m')['mail'])>4:
            prem_code(store_m.get('m')['mail'],self.root.get_screen('premium').ids.code.text)
            self.premium=check_vip()
            if self.premium==0:
                self.root.get_screen('premium').ids.code_err.text = "Wrong premium code!"
            if self.premium==1:
                self.root.get_screen('premium').ids.code_err.text = ""
                self.dis=False
                self.dis_color="#A4E7FE"
                self.root.transition = FadeTransition()
                self.root.current='selection'
        else:
            self.root.get_screen('premium').ids.code_err.text = "Write your email!"     
            
    def build(self):
        self.theme_cls.theme_style="Dark"
        self.theme_cls.primary_palette = "Deepskyblue"
        self.theme_cls.accent_palette = "Gold"
        return Builder.load_file('app.kv')
        #return Builder.load_string(KV)

MainApp().run()
