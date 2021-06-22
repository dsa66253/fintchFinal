import pandas as pd
from pandas.core.frame import DataFrame

# dfStockInfo = pd.read_pickle("./stock_info_all.pkl")
# print(dfStockInfo)
dfCusInfo = pd.read_csv('./cust_info_1.csv')
dfStockInfo = pd.read_pickle("./stock_info_all.pkl")
dfNew = pd.read_pickle("./new.pkl")
# print(dfNew)
# print(float.hex(dfNew.iloc[0, 0]))
# print(dfStockInfo)
# print(dfCusInfo)
# data = {'DATE_RANK':df.iloc[:,0],
#         'STOCK_NO':df.iloc[:,1],
#         'OPEN_PRICE':df.iloc[:,2],
#         'MAX_PRICE':df.iloc[:,3],
#         'MIN_PRICE':df.iloc[:,4],
#         'CLOSE_PRICE':df.iloc[:,5],
#         'VOLUME':df.iloc[:,6],
#         'AMONT':df.iloc[:,7],
#         'CAPITAL_TYPE':df.iloc[:,8],
#         'Alpha':df.iloc[:,9],
#         'BETA_21D':df.iloc[:,10],
#         'BETA_65D':df.iloc[:,11],
#         'BETA_250D':df.iloc[:,12]}
# df2 = pd.DataFrame(data)

# df3 = df1.append(df2)
# print(df3)
# df3.to_pickle("stock_info_all.pkl")
# # print(test)
# test.to_pickle("./stock_info.pkl")

# pd.set_option("max_rows", 100)
# for j in range(0, 1):
#     for i in range(0, len(df.columns)):
#         print(i, df.columns[i], df.iloc[j][i])
#     print("\n")

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# print(df.head(3))


mapUser = {
    "1":"0xDB5D3FF6B7FE584CAE62A6C482194282E627EE8F9BBB37D0BBA43B012950E3D0",
    "2":"0x1879388A362A9CE3DB27F651C333F9BF4B66460538CFD835F4F77E45604CFD4F",
    "3":"0x78D069BCC23EAD3DF21154E5C04838C7F942ACD422BBD1D902A0065A68B90238",
    "4":"0x2E5D5882D3FA49E0B552A4C481F43307F854246B918BFF2399EAA997A6211BFB",
} 
mapStock = {
    "1":"0xEECBB51D8482648B5A0ADF7D6F246A46",
    "2":"0xB380046890B1103F1CB9AE0E801DC070",
    "3":"0x1F6B361C75F3EF3ABD232C1B32B902C1",
    "4":"0x54D555B8812E41DEB72465C28833BA79",
}
stockIndexInStockInfo = {
    "1":[0, 22431, 44849, 67336, 90045],
    "2":[1, 22432, 44850, 67337, 90046],
    "3":[2, 22433, 44851, 67338, 90047],
    "4":[3, 22434, 44852, 67339, 90048,],
}

# linebot call 這個
# id is a string. eg "1" "2" "3" "4"
# buyOrSell 1 for buy , 0 for sell
# stockNo is a string. eg "1" "2" "3" "4"

# shares is a float. 股數 eg 1000.0 2000.0 3000.0 4000.0

# return 一個probability or TF看會不會違約，再回給linebot
def startPredict(id, buyOrSell, stockNo, shares):
    print("startPredict() was called...")
    toTin = makeInputToTin(id, buyOrSell, stockNo, shares)
    print("toTin:", toTin, sep="\n")
    res = callTin(toTin)
    return True


def calAvg(stockIndexInStockInfo, dfStockInfo, stockNo, col):
    sum = 0.0
    for i in stockIndexInStockInfo[stockNo]:
        sum = sum*4/5.0 + dfStockInfo.iloc[i][col]/5.0
    return sum/len(stockIndexInStockInfo[stockNo])

#做出一個15維 list準備要給model吃 不過好像還要call亭臻的function
def makeInputToTin(id, buyOrSell, stockNo, shares):
    print("makeInputToTin() was called...")
    follow = dfNew.loc[mapUser[id]]
    avgROI = follow["ROI"].mean()
    avgOpenPrice = calAvg(stockIndexInStockInfo, dfStockInfo, stockNo, "OPEN_PRICE")
    avgMaxPrice = calAvg(stockIndexInStockInfo, dfStockInfo, stockNo, "MAX_PRICE")
    avgMinPrice = calAvg(stockIndexInStockInfo, dfStockInfo, stockNo, "MIN_PRICE")
    avgClosePrice = calAvg(stockIndexInStockInfo, dfStockInfo, stockNo, "CLOSE_PRICE")

    a = list(range(0, 17))
    data = {"BS_CODE": [buyOrSell],
        "COMMISION_TYPE_CODE":[0],
        "PRICE":[dfStockInfo.iloc[int(stockNo)-1, 5]],
        "STOCKS":[shares],
        "ROI":[avgROI],
        "AGE_LEVEL":[dfCusInfo.iloc[int(id)-1, 1]],
        "OPEN_ACCT_YEAR":[dfCusInfo.iloc[int(id)-1, 2]],
        "INVESTMENT_TXN_CODE":[dfCusInfo.iloc[int(id)-1, 7]],
        "BUY_COUNT":[dfCusInfo.iloc[int(id)-1, 8]],
        "SELL_COUNT":[dfCusInfo.iloc[int(id)-1, 9]],
        "NONTXN_COUNT":[dfCusInfo.iloc[int(id)-1, 10]],
        "OPEN_PRICE":[avgOpenPrice],
        "MAX_PRICE":[avgMaxPrice],
        "MIN_PRICE":[avgMinPrice],
        "CLOSE_PRICE":[avgClosePrice],
        "VOLUME":[dfStockInfo.iloc[int(stockNo)-1, 6]],
        "AMONT":[dfStockInfo.iloc[int(stockNo)-1, 7]]

    }
    # a[0] = buyOrSell #不知道這個是用 0 1 or B S #BS_CODE
    # a[1] = 0 # 直接都是現股買賣 #COMMISION_TYPE_CODE
    # a[2]# = price# PRICE  
    # a[3] = shares # STOCKS
    # a[4] = 0# ROI
    # a[5] = dfCusInfo.iloc[int(id)-1, 1]# AGE_LEVEL
    # a[6] = dfCusInfo.iloc[int(id)-1, 2]# OPEN_ACCT_YEAR
    # a[7] = dfCusInfo.iloc[int(id)-1, 7]# INVESTMENT_TXN_CODE
    # a[8] = dfCusInfo.iloc[int(id)-1, 8]# BUY_COUNT
    # a[9] = dfCusInfo.iloc[int(id)-1, 9]# SELL_COUNT
    # a[10] = dfCusInfo.iloc[int(id)-1, 10]# NONTXN_COUNT
    # a[11] = dfStockInfo.iloc[int(stockNo)-1, 2]# OPEN_PRICE
    # a[12] = dfStockInfo.iloc[int(stockNo)-1, 3]# MAX_PRICE
    # a[13] = dfStockInfo.iloc[int(stockNo)-1, 4]# MIN_PRICE
    # a[14] = dfStockInfo.iloc[int(stockNo)-1, 5]# CLOSE_PRICE
    # a[15] = dfStockInfo.iloc[int(stockNo)-1, 6]# VOLUME
    # a[16] = dfStockInfo.iloc[int(stockNo)-1, 7]# AMONT

    columnList = ["BS_CODE", "COMMISION_TYPE_CODE", "PRICE", "STOCKS", "ROI", "AGE_LEVEL", "OPEN_ACCT_YEAR", "INVESTMENT_TXN_CODE", "BUY_COUNT", "SELL_COUNT", "NONTXN_COUNT", "OPEN_PRICE", "MAX_PRICE", "MIN_PRICE", "CLOSE_PRICE", "VOLUME", "AMONT"]
    a = pd.DataFrame(data)
    a.index = [mapUser[id]]

    a = a.append(follow)

    return a



def callTin(a):
    pass
def printRow(df, num):
    # for i in range(0,num):
    #     print(df.iloc[i][1:])
    for i in range(0,num):
        # print(df.iloc[i][1:])
        print(df.iloc[i,:])
        print("==================================")

def findAlpha(df, num, alpha = -0.0366, BETA_21D = 0.7038):
    # for i in range(0,num):
    #     print(df.iloc[i][1:])
    for i in range(0,num):
        # print(df.iloc[i][1:])
        if df.iloc[i,11] == alpha and df.iloc[i,12] == BETA_21D:
            print(i, df.iloc[i,0:])
            print("==================================")


def printAllStock(df):
    l = []
    for i in range(0,200):
        if not(str(df.iloc[i][2]) in l):
            l.append(df.iloc[i][2])
            print(i, df.iloc[i][2])
    print("l:\n", l)

def printAllRow(df, num):
    l = []
    for i in range(0,num):
        print(i, df.iloc[i][0]==0xEECBB51D8482648B5A0ADF7D6F246A46)
    # print("l:\n", l)


def findStockIndex(df, STOCK_NO):
    indexList = []
    for i in range(0, len(df)):
        if str(df.iloc[i, 1]) == STOCK_NO:
            # print(i, df.iloc[i, 1])
            indexList.append(i)
    return indexList

def printCertainRow(df, indexList):
    for i in indexList:
        print(i, df.iloc[i, 0:])

def printAllUser(df):
    l = []
    count = 0
    print(len(df))
    for i in range(0,1000):
        if not(str(df.iloc[i][0]) in l):
            count = count + 1
            l.append(df.iloc[i][0])
            print(i, df.iloc[i][0])
    # print("l:\n", l)
    print("total user:", count)

def findCustIndex(cust):
    index = 0
    for i in range(0, len(cust)):
        if dfCusInfo.iloc[i, 0]==cust:
            # print(i, dfCusInfo.iloc[i,0:])
            index = i
    return index
def findCustIndexNew(dfNew, cust):
    index = 0
    for i in range(0, len(cust)):
        if dfNew.iloc[i, 0]==cust:
            # print(i, dfCusInfo.iloc[i,0:])
            index = i
    return index

if __name__ == '__main__':
    # printAllUser(df)
    #rintAllStock(df)
    # printRow(df)
    id = "1"
    stockNo = ""
    shares = 1000
    # makeInputToMode(id, stockNo, shares)
    # indexList = findStockIndex(df, mapStock["4"])
    # print(indexList)
    # findCustIndex(mapUser["4"])
    # printRow(dfCusInfo, 4)
    startPredict("1", "1", "1", 1000)
    # startPredict("1", "1", "2", 1000)
    # startPredict("1", "1", "3", 1000)
    # startPredict("1", "1", "4", 1000)
    # printRow(dfNew, 2)
    follow = dfNew.loc["0xDB5D3FF6B7FE584CAE62A6C482194282E627EE8F9BBB37D0BBA43B012950E3D0"]
    # print(dfStockInfo)
    # for i in stockIndexInStockInfo[id]:
    #     print(dfStockInfo.iloc[i,0:])
    # print(dfStockInfo.iloc[0]["OPEN_PRICE"])
    # print(dfStockInfo.iloc[22431]["OPEN_PRICE"])
    # print(dfStockInfo.iloc[44849]["OPEN_PRICE"])
    # print(dfStockInfo.iloc[67336]["OPEN_PRICE"])
    
    # print(dfNew.loc["0xDB5D3FF6B7FE584CAE62A6C482194282E627EE8F9BBB37D0BBA43B012950E3D0"])
    # print(hex(dfNew.iloc[0, 0]))

