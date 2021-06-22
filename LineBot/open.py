import pandas as pd
from pandas.core.frame import DataFrame
from trans_data import parse_data
from sklearn.preprocessing import StandardScaler

dfCusInfo = pd.read_csv('./cust_info_1.csv')
dfStockInfo = pd.read_pickle("./stock_info_all.pkl")
dfNew = pd.read_pickle("./new.pkl")

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)



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
    # print("toTin:", toTin, sep="\n")
    toModel = parse_data(toTin)
    # print(toModel)

    return True




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

    a = pd.DataFrame(data)
    a.index = [mapUser[id]]
    a.index.name = "CUST_NO"
    a = a.append(follow)

    return a

def calAvg(stockIndexInStockInfo, dfStockInfo, stockNo, col):
    sum = 0.0
    for i in stockIndexInStockInfo[stockNo]:
        sum = sum*4/5.0 + dfStockInfo.iloc[i][col]/5.0
    return sum/len(stockIndexInStockInfo[stockNo])

def callTin(a):
    pass


if __name__ == '__main__':

    startPredict("1", "1", "1", 1000)


