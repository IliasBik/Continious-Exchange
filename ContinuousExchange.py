import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rd
import time as time_modul
from matplotlib.patches import Rectangle



data = pd.read_csv("SBER.txt")
seed = rd.randint(0, 35000)
print(seed)
prices = data["open"][seed: seed + 50].tolist()
time = np.arange(len(prices)).tolist()
values = []
print(prices[-1])


plt.ion() #позволяет обновлять график онлайн
fig, ax = plt.subplots()
scatter = ax.scatter(time, prices,s=5)
ax.set_xlim(0, 200) #размеры графика
ax.set_ylim(prices[-1] * 0.8, prices[-1] * 1.2)


epsilon = 0.0000001 #нужная точность определения
price_max = 99999 #максимальная цена, нужна для бинпоиска

class trader:
    def __init__(self, id_trader,category ,balance, stock_count):
        self.balance = balance
        self.stock_count = stock_count
        self.id_trader = id_trader
        self.category = category
        self.value = prices[-1] * stock_count + balance
        self.id_order_trader = 0
        self.order_buy = []
        self.order_sell = []
    def make_easy_order(self, type, risk, size):
        if type == "skip":
            return
        time_max = min(100,len(time))
        time_check = round(min(time_max, risk ** 3 * time_max + 10)) #в зависимости от риска мы считаем сколько последних итераций расмотреть
        max_price_check = max(prices[-time_check:])
        min_price_check = min(prices[-time_check:]) #смотрим на макс и мин цену достигувшую за последние time_check итераций
        
        p_l = min_price_check #/ (1 + rd.random() / 50)
        p_h = max_price_check #* (1 + rd.random() / 100)
        
                
        if type == "b":
            if self.balance > 0:
                money = min(size * self.value, self.balance)
                t_max = risk ** 5 * time_max #считаем, что таких цен как p_l и p_h может достигнуть цена, за то же время назад
                q_max =  money / p_h
                u_max = q_max / t_max * (risk + 1) * 2 #чем больше риск, тем больше u_max

                self.order_buy.append(buyer(p_l=p_l, p_h=p_h, u_max=u_max, q_max=q_max, t_max=t_max, money=money))
                buyers[(self.id_trader, self.id_order_trader)] = self.order_buy[-1]
                self.id_order_trader += 1
                self.balance -= money
                #print("b",u_max)
        if type == "s":
            if self.stock_count > 0: #смотрим, чтобы акций было зашорчено меньше, чем размер портфеля
                money = min(size * self.value, self.stock_count * p_l)
                t_max = risk ** 5 * time_max #считаем, что таких цен как p_l и p_h может достигнуть цена, за то же время назад
                q_max =  money / p_h
                u_max = q_max / t_max * (risk + 1) * 2 #чем больше риск, тем больше u_max

                self.order_sell.append(seller(p_l=p_l, p_h=p_h, u_max=u_max, q_max=q_max, t_max=t_max, q=q_max))
                sellers[(self.id_trader, self.id_order_trader)] = self.order_sell[-1]
                self.id_order_trader += 1
                self.stock_count -= q_max
                #print("s", u_max)

    def make_order(self, predictions, risk, size): 
        p_min = min(predictions)
        p_max = max(predictions)
        
        if prices[-1] - p_min < p_max - prices[-1] and p_max - prices[-1] > prices[-1] * 0.01: #проверка то что профит > 1%
            t_max = predictions.index(p_max) + 1                                            #покупка
            p_l = min(prices[-1], p_min)
            p_h = p_max

            money = min(size * self.value, self.balance)
            q_max = money / p_h
            u_max = q_max / t_max * (risk + 1) * 2 #чем больше риск, тем больше u_max

            self.order_buy.append(buyer(p_l=p_l, p_h=p_h, u_max=u_max, q_max=q_max, t_max=t_max, money=money))
            buyers[(self.id_trader, self.id_order_trader)] = self.order_buy[-1]

            self.id_order_trader += 1
            self.balance -= money

        elif prices[-1] - p_min > p_max - prices[-1] and prices[-1] - p_min > prices[-1] * 0.01: #продажа
            t_max = predictions.index(p_min) + 1 
            p_l = p_min
            p_h = max(prices[-1], p_max)                                       
        
            money = min(size * self.value, self.stock_count * p_l)
            q_max =  money / p_h
            u_max = q_max / t_max * (risk + 1) * 2 #чем больше риск, тем больше u_max

            self.order_sell.append(seller(p_l=p_l, p_h=p_h, u_max=u_max, q_max=q_max, t_max=t_max, q=q_max))
            sellers[(self.id_trader, self.id_order_trader)] = self.order_sell[-1]

            self.id_order_trader += 1
            self.stock_count -= q_max
        else:
            print("SKIP!!")





class buyer:
    def __init__(self, p_l, p_h, u_max, q_max, t_max, money):
        self.p_l = p_l
        self.p_h = p_h
        self.u_max = u_max
        self.q_max = q_max
        self.t_max = t_max
        self.t = 0 #время действия заявки
        self.q = 0 #исполненый объем заявки
        self.money = money #объем денег выделенный на покупку, уменьшается при увеличение q
    def u_demand(self, p):
        if p >= self.p_h:
            return 0
        if p <= self.p_l:
            return min(self.u_max, self.q_max - self.q) #не можем исполнить заявку больше, чем на q_max
        u = self.u_max * (self.p_h - p) / (self.p_h - self.p_l) #считаем объем по специальной формуле, которая должна быть линейна на участке, где мы что-то покупаем
        return min(u, self.q_max - self.q)
    
class seller:
    def __init__(self, p_l, p_h, u_max, q_max, t_max, q):
        self.p_l = p_l
        self.p_h = p_h
        self.u_max = u_max
        self.q_max = q_max
        self.t_max = t_max
        self.t = 0
        self.q = q
        self.money = 0 #деньги полученные при продаже, должна увеличивается при уменьшение q
    def u_supply(self, p):
        if p <= self.p_l:
            return 0
        if p >= self.p_h:
            return min(self.u_max, self.q) #не может продать акций, чем у нас осталось q
        u = self.u_max * (p - self.p_l) / (self.p_h - self.p_l)
        return min(u, self.q)
    
def random_predict(change_size, forward_time, price):
    pred = []
    last_price = price
    
    for i in range(forward_time):
        delta = rd.choices([change_size, 1 / change_size], k=1)[0]

        last_price *= delta 
        pred.append(last_price)
    #print(pred)
    return pred

def overage(price, buyers, sellers): #считаем избыточное предложение
    sum_overage = 0

    for id in buyers:
        sum_overage -= buyers[id].u_demand(price)
        
    for id in sellers:
        sum_overage += sellers[id].u_supply(price)
    
    return sum_overage

def best_round(number, delta):
    number /= delta
    number = round(number)
    return number * delta #функция, которая округляет с точность до delta

def update(price): #пересчитываем исполненный объем и время сделки
    for id in buyers:
        buy = buyers[id].u_demand(price)

        buyers[id].q += buy #увеличиваем счетчик, который следит за исполнением сделки
        buyers[id].t += 1 #счетчик

        (id_trader, id_order_trader) = id
        traders[id_trader].stock_count += buy #добавляем акции на счет трейдера
        buyers[id].money -= buy * price #забираем из резерва деньги

    for id in sellers:
        sell = sellers[id].u_supply(price)

        sellers[id].money += sell * price #изменяем счетчик
        sellers[id].t += 1

        (id_trader, id_order_trader) = id
        traders[id_trader].balance += sell * price #сразу добавляем деньги на счет трейдера
        sellers[id].q -= sell #резерв
        
    

buyers = {} #список покупателей
sellers = {} #список продавцов

creator = trader(category="creator", id_trader=0, balance=9999999, stock_count=99999)
traders = []#creator]

for id_trader in range(0, 100):
    trader_i = trader(id_trader=id_trader, category="random", balance=rd.randint(10000, 11000), stock_count=rd.randint(10000, 11000) / prices[-1])
    traders.append(trader_i)

id_order = 0 #номер заявки, общий для продавцов и покупателей

first_iteration = True



while True:

    time_pred = time_modul.time()

    to_del = [] #создаем массив на удаление элементов, чтобы после обработки удалить их
    for id in buyers:
        (id_trader, id_order_trader) = id
        if buyers[id].q >= buyers[id].q_max or buyers[id].t >= buyers[id].t_max:
            traders[id_trader].balance += buyers[id].money #оставшиеся деньги возвращаем на счет
            to_del.append(id) #добавляем на удаление, если он кончилась по какому-то параметру
    for id in to_del:
        del buyers[id]
    
    to_del = []
    for id in sellers:
        (id_trader, id_order_trader) = id
        if sellers[id].q <= 0 or sellers[id].t >= sellers[id].t_max:
            traders[id_trader].stock_count += sellers[id].q
            to_del.append(id)
    for id in to_del:    
        del sellers[id]
    quests = 0#int(input()) #количество новых операций


    time_now = time_modul.time()
    print("Время на удаление " + str(time_now - time_pred))
    time_pred = time_now


    for i in range(quests):
        quest = list(map(str, input().split())) #информацию о операции разделяется 
        op1 = quest[0] #тип операции
        if op1 == "+": #добавить новую заявку
            typ = quest[1]
            if typ == "b": #заявка на покупку
                [p_l, p_h, u_max, q_max, t_max] = quest[2:] #достаем информацию из заявки
                p_l = float(p_l) #переводим в численные значения т.к. была строка
                p_h = float(p_h)
                u_max = float(u_max)
                q_max = float(q_max)
                t_max = float(t_max)
                buyers[(0, id_order)] = buyer(p_l=p_l, p_h=p_h, u_max=u_max, q_max=q_max, t_max=t_max, money=q_max * p_h) #добавляем по индексу заявку
                print(id_order, "b", p_l, p_h, u_max, q_max, t_max)
                id_order += 1 #обязательно увеличиваем счетчик
            if typ == "s": #аналогично для продавца
                [p_l, p_h, u_max, q_max, t_max] = quest[2:]
                p_l = float(p_l)
                p_h = float(p_h)
                u_max = float(u_max)
                q_max = float(q_max)
                t_max = float(t_max)
                sellers[(0, id_order)] = seller(p_l=p_l, p_h=p_h, u_max=u_max, q_max=q_max, t_max=t_max, q=q_max)
                print(id_order, "s", p_l, p_h, u_max, q_max, t_max)
                id_order += 1

        #удаление заявки очень не работает, надо дописывать до обновления!!!
        if op1 == "-": #если операция требует досрочно убрать заявку
            id2del = int(quest[1]) #достаем индекс заявки, которую надо убрать

            b_del = buyers.pop(id2del, None) #пытаемся удалить из покупателей
            if b_del != None: #выводим информацию о успешном удаление заявки
                print("del buyer:" ,b_del.p_l, b_del.p_h, b_del.u_max, b_del.q_max, b_del.t_max, b_del.t)
            else: 
                s_del = sellers.pop(id2del) #удалим из продавцов если не получилось из покупателей
                print("del seller:", s_del.p_l, s_del.p_h, s_del.u_max, s_del.q_max, s_del.t_max, s_del.t)
    

    for trader_i in traders:
        size = 1 / 5
        if trader_i.category == "easy_random":
            trader_i.make_easy_order(type=rd.choices(["s","b",], weights=[0.5, 0.5], k=1)[0], risk = rd.random(), size = 1 / 5)
        if trader_i.category == "random":
            pred = random_predict(1.02, rd.randint(10,40), prices[-1])
            trader_i.make_order(predictions=pred, risk=rd.random(),size=size)
    left_p = 0
    right_p = price_max


    time_now = time_modul.time()
    print("Время на создание заявок " + str(time_now - time_pred))
    time_pred = time_now


    while right_p - left_p >= epsilon: #пишем бинпоиск для определения равновесной цены
        m = (right_p + left_p) / 2

        if overage(price = m, buyers = buyers, sellers=sellers) > 0:
            right_p = m
        else:
            left_p = m
    
    price = (left_p + right_p) / 2

    price = best_round(price, epsilon) #округляем с нужной точностью

    value = 0
    for id in buyers:
        value += buyers[id].u_demand(price)
    
    print(price, value)


    time_now = time_modul.time()
    print("Время на поиск равновесной цены " + str(time_now - time_pred))
    time_pred = time_now


    values.append(value)

    value_rect = Rectangle(
        xy= (time[-1], prices[49] * 0.8),
        width = 1,
        height = value / (10000/prices[49]) / 1.5 * prices[49] * 0.05 / 5,
        color = "red" if price < prices[-1] else "green",
        alpha = 0.5
    )
    ax.add_patch(value_rect)


    time_now = time_modul.time()
    print("Время на создание прямоугольников " + str(time_now - time_pred))
    time_pred = time_now


    #отвечает за красивые графики
    demand = [] 
    supply = []
    prices1 = []
    for price100 in range(int(prices[-1] * 0.9) * 50, int(prices[-1] * 1.1) * 50):
        price1 = price100 / 50
        demand1 = 0
        for id in buyers:
            demand1 += buyers[id].u_demand(price1)
        supply1 = 0
        for id in sellers:
            supply1 += sellers[id].u_supply(price1)
        demand.append(demand1)
        supply.append(supply1)
        prices1.append(price1)

    plt.figure(2)
    plt.clf()
    plt.scatter(prices1, demand, c='green',s=2)
    plt.scatter(prices1, supply, c='red',s=2)


    time_now = time_modul.time()
    print("Время на спрос/предложение " + str(time_now - time_pred))
    time_pred = time_now


    money = []
    stock_count = []
    for trader_i in traders:
        money.append(trader_i.balance)
        stock_count.append(trader_i.stock_count)
    plt.figure(3)
    plt.clf()
    plt.scatter(money, stock_count, s=6)
    plt.xlim(0,12500)
    plt.ylim(0,12500 / prices[49])
    

    time_now = time_modul.time()
    print("Время на состав портфеля " + str(time_now - time_pred))
    time_pred = time_now


    update(price)

    time.append(time[-1] + 1) #добавляем новую точку
    prices.append(price) 



    scatter.set_offsets(np.column_stack((time, prices))) #обновляем график
    fig.canvas.draw()
    fig.canvas.flush_events()
    

    time_now = time_modul.time()
    print("Время на отрисовку " + str(time_now - time_pred))
    time_pred = time_now


    if first_iteration:
        str1 = str(input())
        first_iteration = False
    
